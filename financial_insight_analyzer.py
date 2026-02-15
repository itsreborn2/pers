"""
재무제표 AI 인사이트 분석기

LLM을 활용하여 재무제표의 이상 패턴을 감지하고,
검색을 통해 원인을 파악하여 종합 보고서를 생성합니다.

아키텍처:
1. LLM 0 (Flash): 업종 파악
2. LLM 1 (Pro): 이상 감지
3. 병렬 웹 리서치 (Pro + Search): 템플릿 기반 프롬프트로 검색 실행
4. LLM 2 (Pro): 종합 보고
"""

import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# 주석 추출용
from dart_financial_extractor import DartFinancialExtractor

# Gemini API
from google import genai
from google.genai import types

# 환경변수 로드
load_dotenv()

# FY 형식 변환 함수
import re

def convert_fy_to_year(text: str) -> str:
    """
    최종 보고서 텍스트에서 FY 형식을 연도 형식으로 변환
    - FY2020-FY2024 → 2020~2024년
    - FY2020년 → 2020년 (중복 방지)
    - FY2020 → 2020년
    """
    if not text or not isinstance(text, str):
        return text

    # 1. FY2020-FY2024 또는 FY2020~FY2024 → 2020~2024년 (기간)
    text = re.sub(r'FY(\d{4})\s*[-~]\s*FY(\d{4})', r'\1~\2년', text)

    # 2. FY2020년 → 2020년 (이미 '년' 붙은 경우, 중복 방지)
    text = re.sub(r'FY(\d{4})년', r'\1년', text)

    # 3. FY2020 → 2020년 (단독)
    text = re.sub(r'FY(\d{4})', r'\1년', text)

    return text

# Gemini 클라이언트 초기화 (P1: HTTP 타임아웃 설정)
API_TIMEOUT_MS = 300_000  # 300초 (5분) — 대형 프롬프트도 충분히 처리
client = genai.Client(
    api_key=os.getenv('GEMINI_API_KEY'),
    http_options=types.HttpOptions(timeout=API_TIMEOUT_MS)
)

# 모델 설정
MODEL_PRO = "gemini-3-pro-preview"  # Pro 모델 (분석용)
MODEL_FLASH = "gemini-3-flash-preview"  # Flash 모델 (빠른 처리)
MODEL_RESEARCH = "gemini-3-pro-preview"  # 리서치 모델 (검색 + 분석)

# 재시도 설정
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 15  # 초

import time
import re


# ============================================================
# P2: 프롬프트 인젝션 방지 — 사용자 입력 새니타이징
# ============================================================
def sanitize_for_prompt(text: str, max_length: int = 200) -> str:
    """
    사용자 입력(회사명 등)을 LLM 프롬프트에 삽입하기 전에 새니타이징.
    프롬프트 인젝션 공격 방지를 위해:
    1. 길이 제한
    2. 제어 문자 제거
    3. 프롬프트 구분자/지시문 패턴 제거
    """
    if not text:
        return text
    # 길이 제한
    text = str(text)[:max_length]
    # 제어 문자 제거 (탭, 줄바꿈 허용하되 다른 제어문자 제거)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # 프롬프트 인젝션 패턴 제거 (시스템 지시문 위장 시도)
    injection_patterns = [
        r'(?i)\b(ignore|disregard|forget)\s+(previous|above|all)\s+(instructions?|prompts?|rules?)',
        r'(?i)\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be|new\s+instructions?)',
        r'(?i)\b(system\s*:?\s*prompt|<<\s*sys|<\|im_start\|>)',
        r'```\s*(system|instruction)',
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, '[FILTERED]', text)
    return text.strip()


# ============================================================
# 계정과목명 정규화 (Solution B)
# DART에서 연도별로 다른 이름 사용 → 통일
# 예: FY2023 "영업손실(이익)" = 251 → FY2024 "영업이익(손실)" = 287
# ============================================================

# (원본 이름, 정규화된 이름, 부호반전 여부)
ACCOUNT_NAME_MAPPING = {
    # 영업이익 변형
    '영업손실(이익)': ('영업이익', True),    # 값에 -1 곱하기
    '영업이익(손실)': ('영업이익', False),
    '영업손실': ('영업이익', True),
    # 법인세비용차감전 변형
    '법인세비용차감전순손실': ('법인세비용차감전이익', True),
    '법인세비용차감전순이익': ('법인세비용차감전이익', False),
    '법인세비용차감전계속사업손실': ('법인세비용차감전이익', True),
    # 당기순이익 변형
    '당기순손실': ('당기순이익', True),
    '당기순이익(손실)': ('당기순이익', False),
    '당기순손실(이익)': ('당기순이익', True),
    # 매출총이익 변형
    '매출총손실': ('매출총이익', True),
}

# 로마숫자 접두어 패턴 (Ⅰ. Ⅱ. Ⅲ. Ⅳ. Ⅴ. 등)
_ROMAN_PREFIX_RE = re.compile(r'^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.?\s*')
_ASCII_ROMAN_PREFIX_RE = re.compile(r'^[IVX]+\.?\s*')


# ============================================================
# VCM ↔ IS 네임스페이스 매핑 (Solution C-2)
# VCM 합산 카테고리와 IS 개별 항목의 관계를 명시
# ============================================================

VCM_IS_NAMESPACE_MAP = {
    '기타영업외수익': {
        'description': 'VCM 잔여 합산 카테고리 (기타수익 + 외환차익 + 외화환산이익 + 유형자산처분이익 등)',
        'is_items': ['기타수익', '외환차익', '외화환산이익', '유형자산처분이익', '투자자산처분이익',
                     '금융수익', '이자수익', '배당금수익', '지분법이익'],
    },
    '기타영업외비용': {
        'description': 'VCM 잔여 합산 카테고리 (기타비용 + 외환차손 + 외화환산손실 + 유형자산처분손실 등)',
        'is_items': ['기타비용', '외환차손', '외화환산손실', '유형자산처분손실', '유형자산손상차손',
                     '금융비용', '이자비용', '지분법손실'],
    },
}


def normalize_account_name(name: str) -> tuple:
    """
    계정과목명을 정규화하여 연도 간 일관성 확보.

    Returns:
        (normalized_name: str, sign_flip: bool)
        sign_flip이 True면 해당 값에 -1을 곱해야 함
    """
    if not name or not isinstance(name, str):
        return (name, False)

    name = name.strip()

    # 1. 로마숫자 접두어 제거: "Ⅴ.영업이익(손실)" → "영업이익(손실)"
    name = _ROMAN_PREFIX_RE.sub('', name)
    name = _ASCII_ROMAN_PREFIX_RE.sub('', name)
    name = name.strip()

    # 2. 매핑 테이블에서 정규화
    if name in ACCOUNT_NAME_MAPPING:
        return ACCOUNT_NAME_MAPPING[name]

    return (name, False)


def normalize_is_data(is_data: list) -> list:
    """
    IS 데이터의 계정과목명을 정규화하고 부호를 통일.
    연도별로 다른 계정명(영업손실 vs 영업이익)을 가진 행을 병합.

    Args:
        is_data: [{'계정과목': '...', 'FY2023': val, 'FY2024': val}, ...]

    Returns:
        정규화된 IS 데이터 (같은 구조, 계정명 통일)
    """
    if not is_data or not isinstance(is_data, list) or len(is_data) == 0:
        return is_data

    headers = list(is_data[0].keys())
    fy_cols = [h for h in headers if h.startswith('FY')]

    # 항목명 키 찾기
    item_key = '계정과목'
    if item_key not in headers:
        for h in headers:
            if h in ('계정과목', '항목', 'item', 'Item'):
                item_key = h
                break

    # 정규화된 데이터를 모으는 딕셔너리
    # key: normalized_name, value: {FY2023: val, FY2024: val, ...}
    merged = {}
    original_order = []  # 원래 순서 유지

    for row in is_data:
        raw_name = str(row.get(item_key, '')).strip()
        if not raw_name:
            continue

        normalized_name, sign_flip = normalize_account_name(raw_name)

        if normalized_name not in merged:
            merged[normalized_name] = {item_key: normalized_name}
            original_order.append(normalized_name)

        # FY 값 채우기
        for fy in fy_cols:
            val = row.get(fy)
            if val is not None:
                # 부호 반전 처리
                if sign_flip:
                    try:
                        if isinstance(val, str):
                            numeric_val = float(val.replace(',', '').replace(' ', ''))
                            val = str(int(-numeric_val)) if numeric_val == int(numeric_val) else str(-numeric_val)
                        else:
                            val = -float(val)
                    except (ValueError, TypeError):
                        pass
                # 기존 값이 없거나 None인 경우에만 채우기
                if merged[normalized_name].get(fy) is None:
                    merged[normalized_name][fy] = val

    # 원래 순서대로 리스트 복원
    result = []
    for name in original_order:
        result.append(merged[name])

    return result


def generate_with_retry(client_obj, model: str, contents, config=None, max_retries=MAX_RETRIES, step_name: str = ""):
    """
    429 Rate Limit 에러 시 자동 재시도 + 타임아웃 에러 처리 래퍼 함수.
    HTTP 타임아웃은 Client 레벨에서 설정 (API_TIMEOUT_MS).
    이 함수는 타임아웃 에러 발생 시 재시도하고, 429는 지수 백오프로 처리.
    """
    last_error = None
    label = f"[{step_name}] " if step_name else ""

    for attempt in range(max_retries):
        try:
            if config:
                return client_obj.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
            else:
                return client_obj.models.generate_content(
                    model=model,
                    contents=contents
                )
        except Exception as e:
            error_str = str(e)
            last_error = e

            # 429 Rate Limit 에러
            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                retry_match = re.search(r'retry.?in.?(\d+(?:\.\d+)?)', error_str.lower())
                if retry_match:
                    wait_time = float(retry_match.group(1)) + 1
                else:
                    wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)

                print(f"{label}[Rate Limit] 429 에러. {wait_time:.1f}초 후 재시도 (시도 {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            # 타임아웃 에러 (httpx ReadTimeout 등)
            elif 'timeout' in error_str.lower() or 'ReadTimeout' in error_str or 'ConnectTimeout' in error_str:
                wait_time = 10 * (attempt + 1)
                print(f"{label}[Timeout] API 타임아웃. {wait_time}초 후 재시도 (시도 {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e

    # 모든 재시도 실패
    raise last_error


@dataclass
class Anomaly:
    """감지된 이상 패턴"""
    period: str         # "FY2024" 또는 "FY2020-FY2024"
    item: str           # 이상 항목명
    finding: str        # 수치와 변화 (사실만)
    context: str        # 관련 항목 수치
    search_queries: List[str] = None  # 원인 추적 검색어 리스트 (검색어 생성 에이전트가 채움)


@dataclass
class SearchTask:
    """검색 태스크"""
    anomaly: Anomaly
    query_type: str  # company, industry, macro, competitor
    query: str


@dataclass
class SearchResult:
    """검색 결과"""
    task: SearchTask
    result: str
    sources: List[str]


def format_period(period: str) -> str:
    """
    기간 형식 변환
    - "FY2024" → "2024년"
    - "FY2020-FY2024" → "2020~2024년"
    """
    if not period:
        return period

    if "-" in period:
        # 기간: FY2020-FY2024 → 2020~2024년
        parts = period.split("-")
        start = parts[0].replace("FY", "")  # FY2020 → 2020
        end = parts[1].replace("FY", "")    # FY2024 → 2024
        return f"{start}~{end}년"
    else:
        # 단일 연도: FY2024 → 2024년
        year = period.replace("FY", "")  # FY2024 → 2024
        return f"{year}년"


class FinancialInsightAnalyzer:
    """재무제표 AI 인사이트 분석기"""

    def __init__(self):
        self.client = client

    async def analyze(
        self,
        financial_data: Dict[str, Any],
        company_info: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        전체 분석 파이프라인 실행

        Args:
            financial_data: 재무제표 데이터 (bs, is, vcm 등)
            company_info: 기업개황정보
            progress_callback: 진행 상태 콜백 함수 (progress, message)

        Returns:
            분석 결과 딕셔너리
        """
        def update(progress: int, message: str):
            if progress_callback:
                progress_callback(progress, message)
            print(f"[{progress}%] {message}")

        # P2: 사용자 입력 새니타이징
        company_name = sanitize_for_prompt(company_info.get('corp_name', '알 수 없음'), max_length=100)
        print(f"\n{'='*60}")
        print(f"[분석 시작] {company_name}")
        print(f"{'='*60}")

        # 1단계: 업종 파악 (Flash + Search)
        update(5, f'[1/6] 업종 파악 중 - {company_name}')
        industry_info = await self._identify_industry(company_info)
        print(f"  → 업종: {industry_info.get('industry', '파악 실패')}")

        # 2단계: 재무제표 주석 추출 (DART API)
        update(10, f'[2/6] 재무제표 주석 추출 중')
        notes_data = await self._extract_notes(company_info)
        print(f"  → 주석 추출: {notes_data.get('notes_count', 0)}개 섹션")

        # 3단계: 이상 감지 (Pro)
        update(20, '[3/6] 재무제표 이상 패턴 감지 중')
        anomalies = await self._detect_anomalies(financial_data, company_info, industry_info)
        print(f"  → 감지된 이상 패턴: {len(anomalies)}개")

        if not anomalies:
            update(100, '분석 완료 - 이상 패턴 없음')
            return {
                "success": False,
                "no_anomalies": True,
                "company_name": company_name,
                "industry_info": industry_info,
                "anomalies": [],
                "insights": "이상 패턴 감지에 실패했습니다. 다시 시도해주세요.",
                "report": None,
                "error": "이상 패턴을 감지하지 못했습니다. AI 분석을 다시 시도해주세요."
            }

        # 4단계: 원인 추적 검색어 생성 (Pro)
        update(30, f'[4/6] 원인 추적 검색어 생성 중 - {len(anomalies)}개 패턴')
        anomalies = await self._generate_search_queries(anomalies, company_info, industry_info)

        # 5단계: 이상 패턴별 원인 분석 (재무제표 + 주석 + 웹 리서치)
        update(45, f'[5/6] 원인 분석 진행 중 - {len(anomalies)}개 병렬 분석')
        search_results = await self._execute_parallel_research(
            anomalies, company_info, industry_info, financial_data, notes_data
        )
        print(f"  → 완료된 리서치: {len(search_results)}개")

        # 6단계: 종합 보고서 생성 (Pro)
        update(75, '[6/8] 종합 보고서(원본) 생성 중')
        report = await self._generate_report(
            financial_data, company_info, industry_info,
            anomalies, search_results, notes_data
        )

        # 7단계: 수치 검증 (Solution D)
        update(85, '[7/8] 수치 검증 중')
        validation_mismatches = self._validate_report_numbers(report, financial_data)
        if validation_mismatches:
            print(f"  → {len(validation_mismatches)}건 수치 불일치 감지, 요약 시 수정 지시 포함")

        # 8단계: 요약본 생성 (Pro) — 검증 결과 + 참조 데이터 전달
        update(90, '[8/9] 요약본 생성 중')
        summary_report = await self._generate_summary(report, company_name, validation_mismatches, financial_data)

        # 9단계: 요약본 수치 검증 및 보정 (Solution D 강화)
        update(93, '[9/9] 요약본 수치 검증 중')
        summary_report = await self._validate_and_fix_summary(summary_report, financial_data, company_name)

        update(95, '보고서 작성 완료')
        print(f"\n{'='*60}")
        print(f"[분석 완료] {company_name}")
        print(f"{'='*60}")

        return {
            "success": True,
            "company_name": company_name,
            "industry_info": industry_info,
            "anomalies": [asdict(a) for a in anomalies],
            "search_results": [
                {
                    "query": sr.task.query,
                    "query_type": sr.task.query_type,
                    "result": sr.result,
                    "sources": sr.sources
                }
                for sr in search_results
            ],
            "report": report,
            "summary_report": summary_report
        }

    async def _extract_notes(self, company_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        사업보고서에서 재무제표 주석 추출

        Args:
            company_info: 기업개황정보 (corp_code 필요)

        Returns:
            주석 데이터 딕셔너리
        """
        corp_code = company_info.get('corp_code', '')
        if not corp_code:
            print("  → 주석 추출 실패: corp_code 없음")
            return {'notes': [], 'notes_text': '', 'notes_count': 0}

        # 사용자 설정 기간 추출
        analysis_start_year = company_info.get('analysis_start_year')
        analysis_end_year = company_info.get('analysis_end_year')
        print(f"  → 주석 추출 기간: {analysis_start_year} ~ {analysis_end_year}")

        try:
            # DART API로 주석 추출 (별도 스레드에서 실행)
            loop = asyncio.get_event_loop()
            extractor = DartFinancialExtractor()

            notes_data = await loop.run_in_executor(
                None,
                lambda: extractor.extract_notes(
                    corp_code,
                    start_year=analysis_start_year,
                    end_year=analysis_end_year
                )
            )

            if notes_data.get('error'):
                print(f"  → 주석 추출 경고: {notes_data.get('error')}")
                return {'notes': [], 'notes_text': '', 'notes_count': 0}

            return notes_data

        except Exception as e:
            print(f"  → 주석 추출 실패: {e}")
            return {'notes': [], 'notes_text': '', 'notes_count': 0, 'error': str(e)}

    async def _identify_industry(self, company_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        업종 파악 (Google Search 활용)
        """
        company_name = sanitize_for_prompt(company_info.get('corp_name', ''), max_length=100)
        induty_code = sanitize_for_prompt(company_info.get('induty_code', ''), max_length=20)

        prompt = f"""
다음 회사의 업종과 사업 내용을 파악해주세요.

회사명: {company_name}
업종코드: {induty_code}

다음 정보를 JSON 형식으로 반환해주세요:
{{
    "industry": "주요 업종 (예: 오피스 가구 제조업)",
    "business_description": "사업 내용 간단 설명",
    "industry_keywords": ["업종 관련 키워드1", "키워드2", ...],
    "competitors": ["주요 경쟁사1", "경쟁사2", ...],
    "macro_factors": ["거시경제 영향 요인1", "요인2", ...]
}}
"""

        try:
            # Flash 모델 + Search로 빠르게 업종 파악
            response = generate_with_retry(
                self.client, MODEL_FLASH, prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                ),
                step_name="업종 파악"
            )

            # JSON 파싱 시도
            result_text = response.text
            # JSON 블록 추출
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            return json.loads(result_text.strip())

        except Exception as e:
            print(f"  [경고] 업종 파악 실패: {e}")
            return {
                "industry": "파악 실패",
                "business_description": "",
                "industry_keywords": [],
                "competitors": [],
                "macro_factors": []
            }

    async def _detect_anomalies(
        self,
        financial_data: Dict[str, Any],
        company_info: Dict[str, Any],
        industry_info: Dict[str, Any]
    ) -> List[Anomaly]:
        """
        이상 패턴 감지 (Pro 모델)
        """
        company_name = sanitize_for_prompt(company_info.get('corp_name', ''), max_length=100)

        # 재무 데이터를 문자열로 변환
        financial_summary = self._format_financial_data(financial_data)

        prompt = f"""
당신은 PE(사모펀드)의 M&A 실사 전문가입니다.
아래 재무제표를 분석하여 인수 검토 시 반드시 확인해야 할 모든 이상 징후를 찾아주세요.

## 회사 정보
- 회사명: {company_name}
- 업종: {industry_info.get('industry', '알 수 없음')}
- 사업: {industry_info.get('business_description', '')}

## 재무 데이터
**중요: 재무상태표/손익계산서는 '백만원' 단위, 현금흐름표는 '원' 단위입니다.**
- 재무상태표/손익계산서: 5400 = 54억원 = 5,400백만원 → 보고서에 "5,400백만원"으로 표기
- 현금흐름표: 5,400,000,000 = 54억원 → 보고서에 "5,400백만원"으로 변환하여 표기

{financial_summary}

## 분석 관점

M&A 실사 전문가로서 다양한 시각에서 **긍정적/부정적 변화 모두** 찾아주세요.
아래는 예시일 뿐이며, 이 외에도 발견되는 모든 유의미한 변화를 보고해주세요.

**중요: 매출/이익 급증도 반드시 보고하세요 (원인 파악 필요: 대형계약, M&A, 일회성 등)**

### A. 손익계산서(IS) 분석 예시
- **매출 급증** (±20% 이상): 신규 대형계약, 인수합병, 일회성 매출 가능성
- **매출 급감**: 주요 고객 이탈, 시장 변화, 사업부 매각
- 영업이익/당기순이익 급변동 (증가/감소 모두), 흑자↔적자 전환
- 매출원가율/판관비율 급변동 (개선/악화 모두)
- 영업외수익/비용 급증 (일회성 항목: 자산매각익, 손상차손 등)
- 특정 비용 항목 이상 (인건비, 대손상각비, 연구개발비 등)

### B. 재무상태표(BS) 분석 예시
- 자산/부채 구조 급변 (증가/감소 모두), 부채비율 변동
- 자본잠식, 누적결손금 심화
- 매출채권/재고자산 급변 (급증: 부실 징후, 급감: 사업축소)
- 충당부채/우발부채 급증 (숨겨진 리스크)
- 유형자산/투자자산 급변 (대규모 투자, 자산매각)

### C. 현금흐름표(CF) 분석 예시
- 영업현금흐름 적자 지속 또는 급변
- 투자/재무 현금흐름 이상 패턴
- 현금 급증/급감

### D. Cross-Check 분석 예시 (재무제표 간 비교)
- [IS↔BS] 매출↑ but 매출채권 더 빠르게↑ → 매출 품질 의심
- [IS↔BS] 매출↑ but 재고↑ → 과잉생산/판매부진
- [IS↔CF] 당기순이익 흑자 but 영업현금흐름 적자 → 이익의 질 의심
- [BS↔CF] 차입금↑ but 재무CF 불일치 → 숨겨진 거래
- [전체] 다년간 지속 패턴 (3년 연속 성장, 3년 연속 적자 등)

위 예시 외에도 PE 투자자 관점에서 유의미한 모든 변화를 빠짐없이 찾아주세요.
**긍정적 변화도 원인 파악이 필요하므로 반드시 포함하세요.**

## 출력 형식
JSON 배열로 반환:
[
    {{
        "period": "FY2024",
        "item": "당기순이익",
        "finding": "130억원 흑자전환 (전년 -80억원, +262%)",
        "context": "영업이익 54억원, 영업외수익 248억원"
    }},
    {{
        "period": "FY2020-FY2024",
        "item": "자본총계",
        "finding": "5년 연속 자본잠식 (-200억 → -527억)",
        "context": "누적결손금 1,200억원, 상환전환우선주 800억원"
    }}
]

주의사항:
- period: 단일 연도("FY2024") 또는 기간("FY2020-FY2024")
- item: 항목명만 기재 (예: "매출액", "당기순이익"). "(긍정적)", "(부정적)" 등 코멘트 절대 추가 금지!
- finding: 수치와 변화 사실만 기재
- context: 관련 항목 수치
- 이상 징후가 없으면 빈 배열 [] 반환

★ 변동률(%) 기재 시 필수 검증 (절대 준수) ★
1. VCM의 '기타영업외수익', '기타영업외비용'은 여러 항목을 합산한 잔여(residual) 카테고리이다. 개별 항목(기타수익, 기타비용 등)과 값이 다르다.
2. 변동률을 기재할 때는 반드시 '손익계산서 세부항목 YoY 변동' 테이블의 개별 항목 수치를 사용하라.
3. 역검증: 기준값 × (1 + 변동률/100) ≈ 당기값 이 성립하는지 확인하라.
   예) 기준값 1,898 × (1 + 1208/100) = 1,898 × 13.08 ≈ 24,820 ✓
   예) 기준값 3,100 × (1 + 684/100) = 3,100 × 7.84 ≈ 24,300 ← 실제 24,820과 불일치 ✗
"""

        # 최대 3회 재시도 (generate_with_retry에 429+타임아웃 재시도 위임)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = generate_with_retry(
                    self.client, MODEL_PRO, prompt,
                    max_retries=1,  # 내부 재시도는 1회 (외부 루프가 JSON 파싱 재시도 담당)
                    step_name="이상 감지"
                )

                result_text = response.text
                if not result_text or not result_text.strip():
                    raise ValueError("빈 응답 수신")

                # JSON 블록 추출
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]

                anomalies_data = json.loads(result_text.strip())

                return [
                    Anomaly(
                        period=a.get('period', ''),
                        item=a.get('item', ''),
                        finding=a.get('finding', ''),
                        context=a.get('context', '')
                    )
                    for a in anomalies_data
                ]

            except Exception as e:
                print(f"  [오류] 이상 감지 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # 재시도 전 2초 대기
                    continue
                return []

        return []

    async def _generate_search_queries(
        self,
        anomalies: List[Anomaly],
        company_info: Dict[str, Any],
        industry_info: Dict[str, Any]
    ) -> List[Anomaly]:
        """
        이상 패턴별 원인 추적 검색어 생성 (Pro 모델)

        각 이상 패턴에 대해 원인을 찾기 위한 다양한 검색어를 생성합니다.
        재무 수치 자체가 아닌, 그 원인이 될 수 있는 사건/뉴스를 찾는 검색어입니다.
        """
        company_name = sanitize_for_prompt(company_info.get('corp_name', ''), max_length=100)
        industry = industry_info.get('industry', '')
        competitors = industry_info.get('competitors', [])
        competitors_str = ', '.join(competitors[:3]) if competitors else ''

        # 모든 이상 패턴을 JSON으로 구성
        anomalies_json = json.dumps([
            {
                "period": a.period,
                "item": a.item,
                "finding": a.finding,
                "context": a.context
            }
            for a in anomalies
        ], ensure_ascii=False, indent=2)

        prompt = f"""당신은 M&A 실사 전문가입니다. 아래 재무제표 이상 패턴들의 **원인**을 찾기 위한 웹 검색어를 생성해주세요.

## 중요 지침
⚠️ **재무 수치 자체를 검색하지 마세요!** 우리는 이미 재무제표 데이터를 가지고 있습니다.
⚠️ **원인이 될 수 있는 사건, 뉴스, 공시를 찾는 검색어**를 생성하세요.

## 회사 정보
- 회사명: {company_name}
- 업종: {industry}
- 주요 경쟁사: {competitors_str}

## 분석 대상 이상 패턴들
{anomalies_json}

## 검색어 생성 가이드

### 잘못된 검색어 예시 (❌ 사용 금지)
- "{company_name} 2024년 재무제표" → 이미 가지고 있음
- "{company_name} 매출액" → 이미 가지고 있음
- "{company_name} 영업이익" → 이미 가지고 있음

### 올바른 검색어 예시 (✅ 이런 방향으로)
**대손상각비 급증의 경우:**
- "{company_name} 거래처 부도"
- "{company_name} 채권 회수 문제"
- "{industry} 대금 연체 증가 2024"
- "{company_name} 소송 패소"

**매출 급감의 경우:**
- "{company_name} 주요 고객 이탈"
- "{company_name} 계약 해지"
- "{industry} 수요 감소 2024"
- "{company_name} 경쟁 심화"

**유형자산 급증의 경우:**
- "{company_name} 신규 공장"
- "{company_name} 설비 투자"
- "{company_name} 인수합병"
- "{company_name} 사업 확장"

**차입금 급증의 경우:**
- "{company_name} 대출"
- "{company_name} 회사채 발행"
- "{company_name} 자금 조달"
- "{company_name} 유동성 위기"

## 출력 형식
각 이상 패턴에 대해 **최소 5개 이상**의 다양한 검색어를 생성하세요.
반드시 아래 JSON 형식으로만 출력하세요:

```json
[
    {{
        "period": "FY2024",
        "item": "대손상각비",
        "search_queries": [
            "{company_name} 거래처 부도 2024",
            "{company_name} 채권 회수 실패",
            "{company_name} 매출채권 손상",
            "{industry} 대금 연체율 2024",
            "{company_name} 소송 손해배상",
            "..."
        ]
    }},
    ...
]
```

모든 이상 패턴에 대해 빠짐없이 검색어를 생성하세요."""

        try:
            print(f"  [검색어 생성 시작] {len(anomalies)}개 이상 패턴")

            response = generate_with_retry(
                self.client, MODEL_PRO, prompt,
                step_name="검색어 생성"
            )

            result_text = response.text

            # JSON 파싱
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            queries_data = json.loads(result_text.strip())

            # 생성된 검색어를 Anomaly 객체에 매핑
            queries_map = {
                (q['period'], q['item']): q.get('search_queries', [])
                for q in queries_data
            }

            for anomaly in anomalies:
                key = (anomaly.period, anomaly.item)
                anomaly.search_queries = queries_map.get(key, [])
                print(f"    → {anomaly.item}: {len(anomaly.search_queries)}개 검색어 생성")

            total_queries = sum(len(a.search_queries or []) for a in anomalies)
            print(f"  [검색어 생성 완료] 총 {total_queries}개 검색어")

            return anomalies

        except Exception as e:
            print(f"  [오류] 검색어 생성 실패: {e}")
            # 실패 시 기본 검색어 설정
            for anomaly in anomalies:
                year = anomaly.period.replace('FY', '').split('-')[-1] if anomaly.period else ''
                anomaly.search_queries = [
                    f"{company_name} {anomaly.item} {year}",
                    f"{company_name} {year}년 주요 이슈",
                    f"{industry} {year}년 동향"
                ]
            return anomalies

    def _build_research_prompt(
        self,
        anomaly: Anomaly,
        company_info: Dict[str, Any],
        industry_info: Dict[str, Any],
        financial_data: Dict[str, Any] = None,
        notes_data: Dict[str, Any] = None
    ) -> str:
        """
        이상 패턴별 원인 분석 프롬프트 생성

        ★ 핵심: 재무제표 내부 분석 + 주석 확인 → 웹 검색으로 보완
        """
        company_name = sanitize_for_prompt(company_info.get('corp_name', ''), max_length=100)
        industry = industry_info.get('industry', '')

        # 생성된 검색어 리스트 포맷팅
        search_queries = anomaly.search_queries or []
        search_queries_str = "\n".join([f"- {q}" for q in search_queries]) if search_queries else "- (검색어 없음)"

        # ★ 재무제표 데이터 포맷팅 (원인 분석용)
        financial_context = ""
        if financial_data:
            financial_context = self._format_financial_data_for_cause_analysis(financial_data, anomaly)

        # ★ 주석 데이터 포맷팅
        notes_context = ""
        if notes_data and notes_data.get('notes_text'):
            # 주석이 너무 길면 잘라서 포함 (최대 30000자)
            notes_text = notes_data.get('notes_text', '')[:30000]
            notes_context = f"""
### 재무제표 주석 (사업보고서 원문)

★★★ 주석 활용 규칙 (필수) ★★★
1. 주석에 있는 숫자와 내용을 **있는 그대로** 인용하라
2. **절대 추론/해석/재작성 금지** - 주석 원문 그대로 가져와라
3. 숫자 흐름만 나열: "주석에 따르면 토지 150억원, 건물 100억원 매각"
4. 주석에 없는 내용은 작성 금지

{notes_text}
"""

        research_prompt = f"""[{anomaly.item}] - {company_name}

## 이상 현상
{anomaly.finding}
관련 정보: {anomaly.context}

## ★ 1단계: 재무제표 내부 분석 (필수)
아래 재무제표 데이터에서 이 이상 현상의 원인을 찾으세요.

{financial_context}

### 분석 방법: 재무제표 항목 간 연결 흐름
이상 현상의 원인을 찾을 때, 재무제표 3표는 서로 연결되어 있음을 활용하세요.

**손익계산서 ↔ 재무상태표 연결**
- 손익계산서의 수익/비용 항목은 재무상태표의 자산/부채 변동과 대응됨
- 예: 대손상각비 증가 → 매출채권 감소, 감가상각비 → 유형자산 감소

**재무상태표 ↔ 현금흐름표 연결**
- 재무상태표의 자산/부채 증감은 현금흐름표의 세부 거래로 설명됨
- 예: 유형자산 감소 → 투자활동의 "처분" 항목, 차입금 증가 → 재무활동의 "차입" 항목

**손익계산서 내부 흐름**
- 영업이익과 당기순이익의 차이 → 영업외수익/비용에서 원인 찾기
- 매출총이익과 영업이익의 차이 → 판관비에서 원인 찾기

이 연결 관계를 따라가며 이상 현상을 설명할 수 있는 구체적 항목과 금액을 찾으세요.

## ★ 2단계: 주석에서 상세 내역 확인
주석에는 재무제표 숫자의 상세 내역이 있습니다. 이상 현상과 관련된 구체적 거래 내역을 찾으세요.
{notes_context}

## 3단계: 웹 검색 (보완)
재무제표/주석 분석 후, 추가 맥락을 위해 웹 검색을 수행하세요.

### search queries
{search_queries_str}

## 출력 규칙

### ★★★ 핵심 규칙: 데이터 기반 ONLY ★★★
- **재무제표/주석/현금흐름표에 있는 숫자만 인용하라. 없는 숫자를 절대 만들지 마라.**
- **모든 숫자는 반드시 위 재무제표 데이터에서 직접 확인 가능해야 한다**
- **★ YoY 변동액 규칙 (절대 준수) ★**: "증가", "감소", "변동" 등을 쓸 때는 반드시 위에 제공된 "YoY 변동액" 테이블의 수치를 사용하라. 해당 연도의 잔액(Balance)을 변동액(Change)으로 절대 쓰지 마라.
  - ❌ 틀린 예: "매출채권 증가(97,477백만원)" ← 이것은 잔액이지 증가액이 아님
  - ✅ 맞는 예: "매출채권 증가(11,140백만원)" ← YoY 변동액 테이블에서 확인한 차이
  - ❌ 틀린 예: "매입채무 감소(72,912백만원)" ← 이것은 잔액이지 감소액이 아님
  - ✅ 맞는 예: "매입채무 감소(7,947백만원)" ← 두 연도 잔액의 차이
- **★ 현금흐름표 정확성 규칙 (절대 준수) ★**:
  - 현금흐름표 항목을 인용할 때 반드시 **정확한 계정과목명**과 **정확한 숫자**를 사용하라
  - 유사한 항목의 숫자를 혼동하지 마라. 예: "비품의처분 18백만원"을 "17,931백만원"으로 쓰면 안 됨
  - **총유출액(gross)과 순유출액(net)을 절대 혼동하지 마라**: "재무활동현금흐름"은 순액(유입-유출)이고, "재무활동으로인한현금유출액"은 총유출액이다
  - 여러 항목을 합산할 때 반드시 각 숫자를 명시하고 합계를 직접 계산하라 (예: "A 1,000 + B 2,000 = 3,000")
- **★ 근사치 사용 규칙 ★**: "약 X억원" 표현 시 실제 수치 대비 ±5% 이내여야 한다. 실제 1,836억원을 "약 1,300억원"이라고 하면 안 됨.
- **★ 퍼센트 변동률 검증 규칙 (절대 준수) ★**:
  - 전년 대비 증감률(%)을 계산할 때 반드시 **정확한 기준연도(FY) 값**을 사용하라
  - 계산 후 역검증: 기준값 × (1 + 증감률/100) ≈ 당기값이 되어야 한다. 안 되면 기준값이 틀린 것이다.
  - **서로 다른 재무제표 항목을 절대 혼동하지 마라**: 기타수익(IS 항목)과 계약부채(BS 항목)는 FY금액이 비슷해도 완전히 다른 항목이다
  - YoY 변동액 테이블에 해당 항목의 변동이 이미 계산되어 있으면 **반드시 그 값을 사용**하라
- 웹 검색에서 나온 숫자가 재무제표와 다르면 **재무제표 숫자를 우선**하라
- 금지 표현: "~로 추정됩니다", "~일 가능성", "~로 보입니다" (근거 없을 때만)
- 허용 표현: "~로 확인됩니다", "~에 기인합니다" (재무제표 근거가 있을 때)

### ★★★ 재무제표 간 연결 기반 추론 (필수) ★★★
주석이 없어도 재무제표 3표(손익계산서, 재무상태표, 현금흐름표)의 연결 관계로 원인을 추론하라.

**예시: 순이익 급증 원인 분석**
- 손익계산서: 영업외수익 '유형자산처분이익' 248억원 발생
- 재무상태표: 유형자산 전년 대비 200억원 감소
- 현금흐름표: 투자활동에 '토지 처분', '건물 처분' 기재
→ **추론**: "유형자산처분이익 248억원 발생, 재무상태표 유형자산 감소 및 현금흐름표 투자활동 처분내역과 일치하여 부동산 처분에 따른 순이익 증가로 확인됩니다"

이처럼 재무제표 간 숫자의 **흐름이 일치**하면 그것은 추측이 아니라 **확인**이다.

### 원인 분석 작성 방법
**3가지 소스(주석, 웹검색, 재무제표)를 모두 활용하여 분석하라.**

**★★★ 출처 표기 금지 ★★★**
다음과 같은 출처 표기 문구를 사용하지 마라:
- ❌ "재무제표 분석 결과,", "손익계산서에 따르면", "현금흐름표상"
- ❌ "주석에 따르면", "주석 X번에 따르면"
- ❌ "웹 검색 결과", "언론 보도에 따르면"

**출력 예시 (출처 표기 없이 바로 내용):**
"손익계산서 유형자산처분이익 248억원이 발생했으며, 재무상태표 유형자산이 감소하고 현금흐름표에 토지/건물 처분이 기록되어 부동산 처분에 따른 순이익 증가로 확인됩니다."

**웹 검색 결과 없으면 언급하지 마라** - 웹 검색에 대해 아무 말도 하지 마라.

**모두 없는 경우에만**: "특별한 내용을 찾지 못했습니다."

### 문체 규칙
- 한국어 경어체 (~했습니다, ~입니다, ~되었습니다)
- 숫자와 항목명 중심으로 간결하게 기술
- 해석/평가/의견 절대 금지

### ★★★ 숫자 단위 규칙 (필수) ★★★
프론트엔드 테이블과 동일하게 **백만원 단위**로 표기하세요:
- 재무상태표/손익계산서: 이미 백만원 단위 → 그대로 사용 (예: 5,400 → "5,400백만원")
- 현금흐름표: 원 단위 → 백만원으로 변환 (예: 5,400,000,000 → "5,400백만원")
- 억원/천만원 변환 금지 (예: ❌ "54억원", ✅ "5,400백만원")

## 출력 형식 (반드시 준수)

**현상**: {anomaly.finding}
**원인 분석**: [재무제표 데이터 기반 사실 및 추론]

- 콜론 바로 뒤에 내용 작성 (줄바꿈 금지)
- 불릿 포인트 금지
- 재무제표/주석의 숫자를 그대로 인용"""

        return research_prompt

    def _format_financial_data_for_cause_analysis(
        self,
        financial_data: Dict[str, Any],
        anomaly: Anomaly
    ) -> str:
        """
        원인 분석을 위한 재무제표 데이터 포맷팅

        ★ 프론트엔드 일관성: vcm_display(백만원 단위)를 우선 사용
        - 프론트 테이블과 AI가 동일한 숫자를 참조
        - vcm_display가 없으면 원본(bs, is, cf) 사용 (fallback)
        """
        result = []

        # 이상 항목에 따라 관련 섹션 강조
        anomaly_item = anomaly.item.lower() if anomaly.item else ""

        def format_table(data, name: str, max_rows: int = 50) -> None:
            """테이블 데이터를 문자열로 변환"""
            if data is None:
                return

            if hasattr(data, 'to_string'):
                result.append(f"\n### {name}")
                result.append(data.to_string())
            elif isinstance(data, list) and len(data) > 0:
                result.append(f"\n### {name}")
                if isinstance(data[0], dict):
                    headers = list(data[0].keys())
                    result.append(" | ".join(str(h) for h in headers))
                    result.append("-" * 80)
                    for row in data[:max_rows]:
                        values = [str(row.get(h, '')) for h in headers]
                        result.append(" | ".join(values))

        # ★ 사전 계산된 참조 데이터 구축 (Solution A+B+C-2 통합)
        ref = self._build_precomputed_reference(financial_data)

        # P2: 단년도 데이터 경고
        if ref.get('single_year'):
            result.append(f"\n{ref['single_year_warning']}")

        # ★ vcm_display 우선 사용 (프론트엔드와 동일한 데이터, 백만원 단위)
        vcm_display = financial_data.get('vcm_display')
        if vcm_display and isinstance(vcm_display, list) and len(vcm_display) > 0:
            format_table(vcm_display, '[VCM] 재무상태표/손익계산서 (단위: 백만원)', max_rows=100)

            # ★ VCM YoY (사전 계산)
            if ref.get('vcm_yoy'):
                result.append(f"\n### ★ [VCM] 주요 항목 YoY 변동액 (단위: 백만원) ★")
                result.append("★★★ 변동액을 언급할 때 반드시 아래 수치를 사용하라. 잔액을 변동액으로 오인하지 마라. ★★★")
                result.append(ref['vcm_yoy'])

            # ★ IS 원본 YoY (계정명 정규화 적용)
            if ref.get('is_yoy'):
                result.append(f"\n### ★ [IS원본] 손익계산서 개별 항목 YoY 변동 (계정명 정규화 적용, 단위: 백만원) ★")
                result.append(ref.get('namespace_warnings', ''))
                result.append(ref['is_yoy'])

            # ★ BS 원본 YoY (신규)
            if ref.get('bs_yoy'):
                result.append(f"\n### ★ [BS원본] 재무상태표 개별 항목 YoY 변동 (단위: 백만원) ★")
                result.append("★★★ BS 항목(재고자산, 매출채권 등)의 변동액/변동률은 반드시 아래 수치를 사용하라. ★★★")
                result.append(ref['bs_yoy'])

            # ★ 현금흐름표
            cf_data = financial_data.get('cf')
            if cf_data is not None:
                format_table(cf_data, '현금흐름표 (단위: 원, 투자활동/재무활동 세부항목 주목)')
        else:
            # fallback: 원본 재무제표 사용
            is_data = financial_data.get('is') if financial_data.get('is') is not None else financial_data.get('cis')
            if is_data is not None:
                format_table(is_data, '손익계산서 (영업외수익/비용 항목 주목)')

            bs_data = financial_data.get('bs')
            if bs_data is not None:
                format_table(bs_data, '재무상태표 (자산/부채 변동 주목)')

            cf_data = financial_data.get('cf')
            if cf_data is not None:
                format_table(cf_data, '현금흐름표 (투자활동/재무활동 세부항목 주목)')

        if not result:
            return "(재무제표 데이터 없음)"

        return "\n".join(result)

    def _calculate_yoy_changes(self, vcm_display: list) -> str:
        """
        vcm_display 데이터에서 주요 항목의 YoY 변동액/변동률을 미리 계산.
        LLM이 잔액(Balance)과 변동액(Change)을 혼동하는 것을 방지.
        """
        if not vcm_display or not isinstance(vcm_display, list) or len(vcm_display) == 0:
            return ""

        # FY 컬럼 추출 (FY2020, FY2021, ... 등)
        headers = list(vcm_display[0].keys())
        fy_cols = sorted([h for h in headers if h.startswith('FY')])
        if len(fy_cols) < 2:
            return ""

        # 항목명 키 찾기
        item_key = '항목'
        if item_key not in headers:
            for h in headers:
                if h in ('항목', 'item', 'Item', '계정'):
                    item_key = h
                    break

        # 주요 항목만 변동액 계산 (BS/IS 핵심 항목 + IS 세부 항목)
        key_items = [
            '매출', '매출원가', '매출총이익', '판매비와관리비', '영업이익',
            '영업외수익', '영업외비용', '금융수익', '금융비용', '기타영업외수익', '기타영업외비용',
            # ★ IS 세부 항목 (LLM이 퍼센트 계산 시 정확한 기준값을 사용하도록)
            '기타수익', '기타비용', '종업원급여', '광고선전비', '감가상각비', '무형자산상각비',
            '이자수익', '이자비용', '유형자산처분이익', '유형자산처분손실', '유형자산손상차손',
            '법인세비용차감전이익', '법인세비용', '당기순이익', 'EBITDA',
            '유동자산', '비유동자산', '자산총계',
            '매출채권및기타채권', '재고자산', '현금및현금성자산', '기타유동자산',
            '유형자산', '무형자산', '기타비유동자산',
            '유동부채', '비유동부채', '부채총계',
            '유동차입부채', '비유동차입부채', '매입채무및기타채무',
            '자본총계', '자본금', '이익잉여금', '기타자본구성요소',
            'NWC', 'Net Debt',
        ]

        lines = []
        # 마지막 두 연도 간 변동만 계산 (가장 최근 YoY)
        last_fy = fy_cols[-1]
        prev_fy = fy_cols[-2]

        lines.append(f"항목 | {prev_fy} 잔액 | {last_fy} 잔액 | 변동액 | 변동률")
        lines.append("-" * 80)

        for row in vcm_display:
            item_name = str(row.get(item_key, '')).strip()
            if not item_name:
                continue
            # % of Sales 등 비율 행은 스킵
            if item_name.startswith('%') or item_name.startswith('(단위'):
                continue

            # 주요 항목 매칭
            matched = False
            for ki in key_items:
                if item_name == ki or item_name.startswith(ki):
                    matched = True
                    break
            if not matched:
                continue

            try:
                val_prev = row.get(prev_fy)
                val_last = row.get(last_fy)
                if val_prev is None or val_last is None:
                    continue
                # 문자열이면 숫자로 변환
                if isinstance(val_prev, str):
                    val_prev = float(val_prev.replace(',', '').replace(' ', '')) if val_prev.strip() else None
                if isinstance(val_last, str):
                    val_last = float(val_last.replace(',', '').replace(' ', '')) if val_last.strip() else None
                if val_prev is None or val_last is None:
                    continue

                val_prev = float(val_prev)
                val_last = float(val_last)
                change = val_last - val_prev
                if val_prev != 0:
                    pct = (change / abs(val_prev)) * 100
                    lines.append(f"{item_name} | {val_prev:,.0f} | {val_last:,.0f} | {change:+,.0f} | {pct:+.1f}%")
                else:
                    lines.append(f"{item_name} | {val_prev:,.0f} | {val_last:,.0f} | {change:+,.0f} | N/A")
            except (ValueError, TypeError):
                continue

        if len(lines) <= 2:  # 헤더만 있으면 빈 결과
            return ""

        return "\n".join(lines)

    def _calculate_raw_yoy_changes(self, raw_data: list, unit: str = '백만원', apply_normalization: bool = False) -> str:
        """
        원본 IS/BS 데이터에서 모든 항목의 YoY 변동액/변동률을 계산.
        VCM에 포함되지 않는 세부항목(기타수익, 종업원급여 등)의 변동도 제공.

        Args:
            apply_normalization: True이면 계정과목명을 정규화
                (영업손실(이익) → 영업이익 + 부호반전 등)
        """
        if not raw_data or not isinstance(raw_data, list) or len(raw_data) == 0:
            return ""

        # ★ 계정과목명 정규화 적용 (Solution B)
        data_to_use = normalize_is_data(raw_data) if apply_normalization else raw_data

        headers = list(data_to_use[0].keys())
        fy_cols = sorted([h for h in headers if h.startswith('FY')])
        if len(fy_cols) < 2:
            return ""

        item_key = '계정과목'
        if item_key not in headers:
            for h in headers:
                if h in ('계정과목', '항목', 'item', 'Item'):
                    item_key = h
                    break

        last_fy = fy_cols[-1]
        prev_fy = fy_cols[-2]

        # 원 단위인 경우 백만원으로 변환
        divisor = 1_000_000 if unit == '원' else 1

        lines = []
        lines.append(f"항목 | {prev_fy} | {last_fy} | 변동액 | 변동률")
        lines.append("-" * 80)

        for row in data_to_use:
            item_name = str(row.get(item_key, '')).strip()
            if not item_name or item_name.startswith('%') or item_name.startswith('(단위'):
                continue
            # 소계/합계 등 제외
            if any(skip in item_name for skip in ['소계', '합계', '---']):
                continue

            try:
                val_prev = row.get(prev_fy)
                val_last = row.get(last_fy)
                if val_prev is None or val_last is None:
                    continue
                if isinstance(val_prev, str):
                    val_prev = float(val_prev.replace(',', '').replace(' ', '')) if val_prev.strip() else None
                if isinstance(val_last, str):
                    val_last = float(val_last.replace(',', '').replace(' ', '')) if val_last.strip() else None
                if val_prev is None or val_last is None:
                    continue

                val_prev = float(val_prev) / divisor
                val_last = float(val_last) / divisor

                # 0이 아닌 변동만 표시
                change = val_last - val_prev
                if change == 0:
                    continue

                if val_prev != 0:
                    pct = (change / abs(val_prev)) * 100
                    lines.append(f"{item_name} | {val_prev:,.0f} | {val_last:,.0f} | {change:+,.0f} | {pct:+.1f}%")
                else:
                    lines.append(f"{item_name} | {val_prev:,.0f} | {val_last:,.0f} | {change:+,.0f} | N/A")
            except (ValueError, TypeError):
                continue

        if len(lines) <= 2:
            return ""

        return "\n".join(lines)

    def _calculate_bs_yoy_changes(self, bs_data: list) -> str:
        """
        BS(재무상태표) 원본 데이터에서 모든 항목의 YoY 변동액/변동률을 계산.
        VCM에 없는 세부항목(재고자산, 매출채권 등 개별 계정)의 정확한 변동을 제공하여
        LLM이 잔액에서 직접 빼기/나누기를 하지 않도록 방지.
        """
        if not bs_data or not isinstance(bs_data, list) or len(bs_data) == 0:
            return ""

        headers = list(bs_data[0].keys())
        fy_cols = sorted([h for h in headers if h.startswith('FY')])
        if len(fy_cols) < 2:
            return ""

        item_key = '계정과목'
        if item_key not in headers:
            for h in headers:
                if h in ('계정과목', '항목', 'item', 'Item'):
                    item_key = h
                    break

        last_fy = fy_cols[-1]
        prev_fy = fy_cols[-2]

        lines = []
        lines.append(f"항목 | {prev_fy} | {last_fy} | 변동액 | 변동률")
        lines.append("-" * 80)

        for row in bs_data:
            item_name = str(row.get(item_key, '')).strip()
            if not item_name or item_name.startswith('%') or item_name.startswith('(단위'):
                continue
            if any(skip in item_name for skip in ['소계', '합계', '---']):
                continue

            try:
                val_prev = row.get(prev_fy)
                val_last = row.get(last_fy)
                if val_prev is None or val_last is None:
                    continue
                if isinstance(val_prev, str):
                    val_prev = float(val_prev.replace(',', '').replace(' ', '')) if val_prev.strip() else None
                if isinstance(val_last, str):
                    val_last = float(val_last.replace(',', '').replace(' ', '')) if val_last.strip() else None
                if val_prev is None or val_last is None:
                    continue

                val_prev = float(val_prev)
                val_last = float(val_last)
                change = val_last - val_prev
                if change == 0:
                    continue

                if val_prev != 0:
                    pct = (change / abs(val_prev)) * 100
                    lines.append(f"{item_name} | {val_prev:,.0f} | {val_last:,.0f} | {change:+,.0f} | {pct:+.1f}%")
                else:
                    lines.append(f"{item_name} | {val_prev:,.0f} | {val_last:,.0f} | {change:+,.0f} | N/A")
            except (ValueError, TypeError):
                continue

        if len(lines) <= 2:
            return ""

        return "\n".join(lines)

    def _build_precomputed_reference(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM에 제공할 사전 계산된 참조 데이터 구축 (Solution A+B+C-2 통합).
        모든 YoY 변동을 코드에서 미리 계산하여 LLM이 직접 계산하지 않도록 함.

        Returns:
            {
                'vcm_yoy': str,          # VCM 항목 YoY 변동 테이블
                'is_yoy': str,           # IS 원본 항목 YoY 변동 (정규화 적용)
                'bs_yoy': str,           # BS 원본 항목 YoY 변동
                'namespace_warnings': str # VCM ↔ IS 네임스페이스 경고
                'single_year': bool      # P2: 단년도 데이터 여부
            }
        """
        ref = {}

        # P2: 단년도 데이터 감지
        vcm_display = financial_data.get('vcm_display')
        fy_count = 0
        if vcm_display and isinstance(vcm_display, list) and len(vcm_display) > 0:
            headers = list(vcm_display[0].keys())
            fy_count = len([h for h in headers if h.startswith('FY')])
        ref['single_year'] = fy_count < 2

        if ref['single_year']:
            print(f"  [P2] 단년도 데이터 감지 (FY 컬럼: {fy_count}개) — YoY 비교 불가, 절대값 분석 모드")
            ref['single_year_warning'] = (
                "★★★ 주의: 이 기업은 1개년도 데이터만 존재합니다 ★★★\n"
                "- YoY(전년 대비) 변동률/변동액을 절대 언급하지 마세요\n"
                "- '증가', '감소', '전년 대비' 표현을 사용하지 마세요\n"
                "- 절대값 기준으로만 분석하세요 (예: '매출액 X백만원 기록')\n"
                "- 비율 분석은 가능합니다 (예: 영업이익률, 부채비율 등)"
            )

        # 1. VCM YoY (기존)
        if vcm_display and isinstance(vcm_display, list) and len(vcm_display) > 0:
            ref['vcm_yoy'] = self._calculate_yoy_changes(vcm_display)

        # 2. IS YoY (★ 계정과목 정규화 적용)
        is_data = financial_data.get('is') if financial_data.get('is') is not None else financial_data.get('cis')
        if is_data and isinstance(is_data, list) and len(is_data) > 0:
            ref['is_yoy'] = self._calculate_raw_yoy_changes(is_data, '백만원', apply_normalization=True)

        # 3. BS YoY (★ 신규 추가)
        bs_data = financial_data.get('bs')
        if bs_data and isinstance(bs_data, list) and len(bs_data) > 0:
            ref['bs_yoy'] = self._calculate_bs_yoy_changes(bs_data)

        # 4. VCM ↔ IS 네임스페이스 경고 (Solution C-2)
        warnings = []
        warnings.append("★★★ VCM과 IS 원본의 차이 (절대 혼동 금지) ★★★")
        for vcm_name, info in VCM_IS_NAMESPACE_MAP.items():
            warnings.append(f"- [VCM] '{vcm_name}' = {info['description']}")
            warnings.append(f"  → [IS원본] 개별 항목: {', '.join(info['is_items'][:5])} 등")
            warnings.append(f"  → VCM 합산값과 IS 개별 항목의 변동률(%)은 다르다! IS 원본 변동률을 사용하라.")
        ref['namespace_warnings'] = "\n".join(warnings)

        return ref

    def _validate_report_numbers(self, report: str, financial_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        생성된 보고서에서 핵심 수치를 추출하여 원본 데이터와 비교검증.
        불일치 항목 목록을 반환.

        Returns:
            [{'item': '재고자산', 'report_value': '-2,409', 'source_value': '-829',
              'source': 'BS YoY', 'severity': 'HIGH'}, ...]
        """
        mismatches = []

        # 사전 계산된 참조 데이터 구축
        ref = self._build_precomputed_reference(financial_data)

        # YoY 테이블들에서 항목별 정확한 값 추출
        source_values = {}  # {'항목명': {'변동액': val, '변동률': val, 'source': 'VCM/IS/BS'}}

        def parse_yoy_table(yoy_str: str, source_name: str):
            if not yoy_str:
                return
            for line in yoy_str.split('\n'):
                if '|' not in line or line.startswith('-'):
                    continue
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 5 and parts[0] != '항목':
                    item_name = parts[0]
                    try:
                        change_str = parts[3].replace(',', '').replace('+', '')
                        change_val = float(change_str)
                        pct_str = parts[4].replace('%', '').replace('+', '').strip()
                        pct_val = float(pct_str) if pct_str != 'N/A' else None
                        source_values[item_name] = {
                            '변동액': change_val,
                            '변동률': pct_val,
                            'source': source_name,
                            'prev': float(parts[1].replace(',', '')),
                            'last': float(parts[2].replace(',', '')),
                        }
                    except (ValueError, IndexError):
                        continue

        parse_yoy_table(ref.get('vcm_yoy', ''), 'VCM')
        parse_yoy_table(ref.get('is_yoy', ''), 'IS')
        parse_yoy_table(ref.get('bs_yoy', ''), 'BS')

        # 보고서에서 "X백만원" 또는 "+X%" 패턴 추출 후 검증
        # ★ False positive 방지용 제외 단어 (조사, 부사 등)
        _EXCLUDE_ITEMS = {'대비', '전년', '당기', '전기', '약', '중', '각각', '총',
                          '동', '및', '등', '간', '상', '이상', '이하', '수준', '규모',
                          '만큼', '정도', '결과', '원인', '영향', '기준', '이후'}

        # 핵심 항목명 + 변동률 패턴 검색 (최소 2글자 항목명)
        pct_pattern = re.compile(
            r'([가-힣A-Za-z]{2,}(?:및[가-힣]+)?)\s*[은는이가]*\s*'
            r'(?:전년\s*대비\s*)?'
            r'([+-]?\d+[\d,.]*)\s*%\s*'
            r'(증가|감소|급증|급감|개선|악화|상승|하락)',
            re.UNICODE
        )

        for match in pct_pattern.finditer(report):
            item = match.group(1).strip()
            report_pct = match.group(2).replace(',', '')

            # ★ 제외 단어 필터링
            if item in _EXCLUDE_ITEMS:
                continue

            # source_values에서 매칭 검색 (★ 최소 2글자 이상 매칭 요구)
            for src_item, src_data in source_values.items():
                # 매칭 조건 강화: 2글자 미만이면 건너뜀, 부분 매칭은 최소 3글자
                match_len = min(len(item), len(src_item))
                if match_len < 2:
                    continue
                if not (item in src_item or src_item in item):
                    continue
                    if src_data['변동률'] is not None:
                        try:
                            report_pct_val = float(report_pct)
                            source_pct_val = abs(src_data['변동률'])
                            # 10% 이상 차이나면 MISMATCH
                            if abs(report_pct_val - source_pct_val) > max(source_pct_val * 0.1, 5):
                                mismatches.append({
                                    'item': item,
                                    'type': '변동률',
                                    'report_value': f"{report_pct}%",
                                    'source_value': f"{src_data['변동률']:+.1f}%",
                                    'source': src_data['source'],
                                    'severity': 'HIGH' if abs(report_pct_val - source_pct_val) > source_pct_val * 0.2 else 'MEDIUM'
                                })
                        except ValueError:
                            pass
                    break

        # 핵심 항목 변동액 패턴 검색 (예: "재고자산 2,409백만원 감소", 최소 2글자 항목명)
        change_pattern = re.compile(
            r'([가-힣A-Za-z]{2,}(?:및[가-힣]+)?)\s*[은는이가]*\s*'
            r'([+-]?\d[\d,.]*)\s*백만원\s*'
            r'(증가|감소|급증|급감|순증|순감|유입|유출)',
            re.UNICODE
        )

        for match in change_pattern.finditer(report):
            item = match.group(1).strip()
            report_val_str = match.group(2).replace(',', '')
            direction = match.group(3)

            # ★ 제외 단어 필터링
            if item in _EXCLUDE_ITEMS:
                continue

            for src_item, src_data in source_values.items():
                match_len = min(len(item), len(src_item))
                if match_len < 2:
                    continue
                if not (item in src_item or src_item in item):
                    continue
                    try:
                        report_val = float(report_val_str)
                        source_val = abs(src_data['변동액'])
                        # 20% 이상 차이나면 MISMATCH
                        if source_val > 0 and abs(report_val - source_val) > max(source_val * 0.2, 100):
                            mismatches.append({
                                'item': item,
                                'type': '변동액',
                                'report_value': f"{report_val_str}백만원",
                                'source_value': f"{src_data['변동액']:+,.0f}백만원",
                                'source': src_data['source'],
                                'severity': 'HIGH' if abs(report_val - source_val) > source_val * 0.5 else 'MEDIUM'
                            })
                    except ValueError:
                        pass
                    break

        if mismatches:
            print(f"  [검증] {len(mismatches)}건 불일치 감지:")
            for m in mismatches:
                print(f"    [{m['severity']}] {m['item']}: 보고서 {m['report_value']} vs 원본({m['source']}) {m['source_value']}")

        return mismatches

    async def _execute_parallel_research(
        self,
        anomalies: List[Anomaly],
        company_info: Dict[str, Any],
        industry_info: Dict[str, Any],
        financial_data: Dict[str, Any] = None,
        notes_data: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """
        이상 패턴별 원인 분석 병렬 실행

        각 이상 패턴에 대해:
        1. ★ 재무제표 내부 데이터 분석 (1차 원인 추적)
        2. ★ 재무제표 주석에서 상세 내역 확인 (2차)
        3. 웹 리서치로 추가 맥락 확보 (3차 보완)

        모든 이상 패턴은 병렬로 처리됨
        """
        def extract_sources(response) -> List[str]:
            """응답에서 소스 URL 추출"""
            sources = []
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    metadata = candidate.grounding_metadata
                    if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri'):
                                sources.append(chunk.web.uri)
            return sources

        def build_fallback_prompt(anomaly: Anomaly) -> str:
            """소스 없을 때 사용할 대체 검색 프롬프트"""
            company_name = sanitize_for_prompt(company_info.get('corp_name', ''), max_length=100)
            industry = industry_info.get('industry', '')

            # 더 넓은 범위의 대체 검색어
            year = anomaly.period.replace('FY', '')
            fallback_queries = [
                f"{company_name} {year}년 뉴스",
                f"{company_name} {year}년 실적 발표",
                f"{company_name} 경영 이슈",
                f"{company_name} 사업 현황",
                f"{industry} {year}년 동향",
                f"{industry} 업계 뉴스 {year}",
            ]
            queries_str = "\n".join([f"- {q}" for q in fallback_queries])

            return f"""[Fallback] {anomaly.item} - {company_name}

## fallback search queries
{queries_str}

## MANDATORY: USE GOOGLE SEARCH ONLY
- You MUST execute Google Search for the queries above
- NEVER use your training data or internal knowledge
- ONLY use information found in Google Search results

## rules
1. Execute Google Search - THIS IS MANDATORY
2. Write ONLY facts found in search results
3. Do not repeat company name
4. NEVER use your pre-trained knowledge

## CRITICAL RULE: DATA-BASED INFERENCE ONLY
- FORBIDDEN: Using your training data or internal knowledge
- If no search results AND no financial data connections found, write ONLY: "보고서와 웹 검색에서 특별한 내용을 찾지 못했습니다."
- FORBIDDEN: general theories, "may be", "possibly", "likely" without data basis
- ALLOWED: Data-based inference when financial statements show correlated changes (e.g., gain on disposal in IS + asset decrease in BS + disposal in CF → real estate sale)
- Write concrete facts confirmed by Google Search OR derived from financial statement connections

## Writing style rule
- "cause analysis" section ONLY: Use Korean formal style (honorific)
- Formal: ~habnida, ~imnida (~했습니다, ~입니다)
- Informal FORBIDDEN: ~da, ~haetda (~다, ~했다)

## Output format (MUST follow exactly) - USE KOREAN LABELS

**현상**: {anomaly.finding}
**원인 분석**: 보고서와 웹 검색에서 특별한 내용을 찾지 못했습니다.

Write content immediately after the colon
- NO line break! Text starts right after colon
- NO bullet points!
- If financial statement data connections OR search results exist: Replace default text with found facts (in Korean formal style)
- If NO relevant data: Keep default text (DO NOT use training data)"""

        def research_one_sync(anomaly: Anomaly) -> SearchResult:
            """동기 함수로 API 호출 (스레드에서 실행) - 재무제표 내부 분석 + 웹 검색"""
            # 1. 원인 분석 프롬프트 구성 (재무제표 + 주석 데이터 포함)
            print(f"    [원인 분석 시작] {anomaly.period} {anomaly.item}")
            prompt = self._build_research_prompt(anomaly, company_info, industry_info, financial_data, notes_data)

            # 더미 SearchTask 생성 (기존 구조 호환용)
            task = SearchTask(
                anomaly=anomaly,
                query_type="integrated",
                query=f"{anomaly.period} {anomaly.item} 통합 분석"
            )

            try:
                # 2. Pro + Search로 실제 웹 리서치 수행 (1차 시도)
                print(f"    [웹 리서치 시작] {anomaly.period} {anomaly.item}")

                response = generate_with_retry(
                    self.client, MODEL_RESEARCH, prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())]
                    ),
                    step_name=f"리서치:{anomaly.item}"
                )

                # 소스 URL 추출
                sources = extract_sources(response)
                result_text = response.text if response.text else "결과 없음"

                # 웹 소스 없어도 1차 응답(주석/재무제표 기반 분석) 유지
                if not sources:
                    print(f"    [웹 소스 없음] {anomaly.period} {anomaly.item} - 재무제표/주석 기반 분석 유지")

                print(f"    [웹 리서치 완료] {anomaly.period} {anomaly.item}")
                print(f"    ┌─────────────────────────────────────────────────────────")
                print(f"    │ [리서치 결과] {anomaly.period} {anomaly.item}")
                print(f"    │ 소스: {sources[:3]}")
                print(f"    │ 내용 (앞 500자):")
                for line in result_text[:500].split('\n'):
                    print(f"    │   {line}")
                print(f"    └─────────────────────────────────────────────────────────")

                return SearchResult(
                    task=task,
                    result=result_text,
                    sources=sources[:5]  # 최대 5개 소스
                )

            except Exception as e:
                print(f"    [웹 리서치 실패] {anomaly.period} {anomaly.item}: {e}")
                # P1: 에러 메시지가 보고서에 누출되지 않도록 중립적 대체 텍스트 사용
                return SearchResult(
                    task=task,
                    result=f"**현상**: {anomaly.finding}\n\n**원인 분석**: 웹 검색에서 관련 정보를 찾지 못했습니다. 재무제표 데이터에 기반한 추가 분석이 필요합니다.",
                    sources=[]
                )

        # 모든 이상 패턴에 대해 완전 병렬 실행 (ThreadPoolExecutor 사용)
        print(f"  → {len(anomalies)}개 이상 패턴 병렬 웹 리서치 시작 (Pro + Search)")

        loop = asyncio.get_event_loop()
        # 최대 10개 스레드로 동시 실행 보장
        RESEARCH_TIMEOUT = 360  # 개별 리서치 6분 타임아웃
        with ThreadPoolExecutor(max_workers=min(len(anomalies), 10)) as executor:
            futures = [loop.run_in_executor(executor, research_one_sync, a) for a in anomalies]
            # P1: 개별 리서치에 타임아웃 적용
            async def safe_await(future, anomaly):
                try:
                    return await asyncio.wait_for(future, timeout=RESEARCH_TIMEOUT)
                except asyncio.TimeoutError:
                    print(f"    [타임아웃] {anomaly.period} {anomaly.item} ({RESEARCH_TIMEOUT}초 초과)")
                    task = SearchTask(anomaly=anomaly, query_type="integrated", query=f"{anomaly.period} {anomaly.item} 통합 분석")
                    return SearchResult(
                        task=task,
                        result=f"**현상**: {anomaly.finding}\n\n**원인 분석**: 분석 시간이 초과되었습니다. 재무제표 데이터에 기반한 추가 분석이 필요합니다.",
                        sources=[]
                    )
            results = await asyncio.gather(*[safe_await(f, a) for f, a in zip(futures, anomalies)])

        print(f"  → {len(results)}개 리서치 완료")
        return list(results)

    async def _generate_report(
        self,
        financial_data: Dict[str, Any],
        company_info: Dict[str, Any],
        industry_info: Dict[str, Any],
        anomalies: List[Anomaly],
        search_results: List[SearchResult],
        notes_data: Dict[str, Any] = None
    ) -> str:
        """
        종합 보고서 생성 (Pro 모델)
        """
        company_name = sanitize_for_prompt(company_info.get('corp_name', ''), max_length=100)

        # 이상 패턴 + 리서치 결과 통합 (중복 제거)
        # anomaly와 해당 search_result를 매칭
        search_map = {}
        for sr in search_results:
            key = (sr.task.anomaly.period, sr.task.anomaly.item)
            search_map[key] = sr.result

        combined_findings = ""
        for i, a in enumerate(anomalies, 1):
            key = (a.period, a.item)
            research = search_map.get(key, "**현상**: " + a.finding + "\n\n**원인 분석**: 보고서와 웹 검색에서 특별한 내용을 찾지 못했습니다.")
            # P1: 리서치 실패/에러 메시지가 보고서에 누출되지 않도록 이중 필터링
            if research and any(err_sig in research for err_sig in ["리서치 실패:", "RESOURCE_EXHAUSTED", "Exception:", "Traceback", "Error:"]):
                print(f"  [P1 필터] 리서치 에러 메시지 제거됨: {a.period} {a.item}")
                research = f"**현상**: {a.finding}\n\n**원인 분석**: 웹 검색에서 관련 정보를 찾지 못했습니다. 재무제표 데이터에 기반한 추가 분석이 필요합니다."
            period_display = format_period(a.period)
            combined_findings += f"""
### {i}. [{period_display}] {a.item}
{research}
"""

        # 업종 분석 결과를 간결하게 포맷
        industry_summary = f"""- 업종: {industry_info.get('industry', '')}
- 사업: {industry_info.get('business_description', '')}
- 경쟁사: {', '.join(industry_info.get('competitors', [])[:3])}
- 거시요인: {', '.join(industry_info.get('macro_factors', [])[:3])}"""

        # P2: 단년도 데이터 경고 (보고서 프롬프트에 포함)
        single_year_section = ""
        ref_check = self._build_precomputed_reference(financial_data)
        if ref_check.get('single_year'):
            single_year_section = f"\n{ref_check['single_year_warning']}\n"

        # 주석 요약 (있으면 포함)
        notes_summary = ""
        if notes_data and notes_data.get('notes_text'):
            notes_text = notes_data.get('notes_text', '')[:15000]  # 최대 15000자
            notes_summary = f"""
## 재무제표 주석 원문

★★★ 주석 정리 규칙 (필수) ★★★
1. **있는 그대로 정리만** - 추론/해석/의견 절대 금지
2. **숫자를 그대로 인용** - "토지 150억원, 건물 100억원" 식으로
3. **항목별로 나열만** - 특수관계자, 우발채무, 담보, 자산처분 등
4. 주석에 없는 내용은 작성 금지

{notes_text}
"""

        prompt = f"""[{company_name}] 재무 이상 패턴 정리
{single_year_section}
## 회사/업종 정보
{industry_summary}

## 이상 패턴 및 조사 결과
{combined_findings}
{notes_summary}
## 작성 규칙 (★★★ 필수 ★★★)
1. **정리만 하라** - 새로운 해석/추론/재작성 절대 금지
2. 위 데이터의 **현상**, **원인 분석**을 그대로 복사
3. 당신의 역할은 포맷 정리뿐. 내용 수정/추가 금지
4. **주석은 숫자를 그대로 나열** - 추론/해석 절대 금지
5. 주석에 없는 내용 작성 금지
6. **출처 표기 문구 제거** - "재무제표 분석 결과,", "주석에 따르면", "웹 검색 결과" 등의 출처 표기 문구는 모두 삭제하라
7. 금지 표현: "~로 분석됩니다", "~로 추정됩니다"
8. **웹 검색 결과 없으면 언급 금지** - 웹 검색에 대해 아무 말도 하지 마라
9. **주석 요약 섹션 필수** - "## 주석 요약" 섹션은 반드시 포함하라. 주석 데이터가 있으면 해당 내용을 정리하고, 없으면 "주석 데이터가 없습니다"라고 작성
10. **★ 숫자 정확성 최우선 ★** - 재무제표에 없는 숫자를 절대 만들지 마라. 모든 수치는 원본 데이터에서 직접 검증 가능해야 한다. 유사한 항목의 숫자를 혼동하지 마라 (예: 비품의처분 18을 17,931으로 쓰지 마라).
11. **★ 잔액 vs 변동액 혼동 절대 금지 ★** - "증가", "감소", "변동" 등의 표현 뒤에는 반드시 YoY 변동액(두 시점의 차이)을 써라. 해당 연도의 잔액(Balance)을 변동액처럼 쓰지 마라. 예: 매출채권 FY2024 잔액 97,477 → "매출채권 증가 97,477" (❌ 틀림). 매출채권 FY2023 86,337 → FY2024 97,477, 차이 11,140 → "매출채권 증가 11,140" (✅ 맞음)
12. **★ 총유출액 vs 순액 혼동 금지 ★** - "재무활동현금흐름"이라고 쓰면 반드시 순액(net)을 써라. 총유출액(gross)과 순액을 혼동하지 마라. "재무활동으로인한현금유출액"과 "재무활동으로인한현금흐름"은 다른 숫자다.
13. **★ 근사치 정확성 ★** - "약 X억원", "약 X백만원" 등 근사치를 쓸 때 실제 수치 대비 ±5% 이내로 정확히 표기하라. 1,836억원을 "약 1,300억원"이라 하면 안 됨.
14. **★ 퍼센트 변동률 검증 (절대 준수) ★** - 전년 대비 증감률(%)을 쓸 때 반드시 다음 공식으로 역검증하라: 기준값 × (1 + 증감률/100) ≈ 당기값. 역검증이 실패하면 기준값이 잘못된 것이므로 원본 데이터에서 정확한 기준값을 다시 확인하라. 예: "685% 증가"라 쓰면 기준값 × 7.85 = 당기값이 되어야 한다. 기타수익(IS)과 계약부채(BS)처럼 이름이 다른 항목의 숫자를 절대 혼동하지 마라.

## ★★★ 숫자 단위 규칙 (필수) ★★★
프론트엔드 테이블과 동일하게 **백만원 단위**로 표기하세요:
- 재무상태표/손익계산서: 이미 백만원 단위 → 그대로 사용 (예: 5,400 → "5,400백만원")
- 현금흐름표/주석: 원 단위 → 백만원으로 변환 (예: 5,400,000,000 → "5,400백만원")
- 억원/천만원 변환 금지 (예: ❌ "54억원", ✅ "5,400백만원")

## 출력 형식

# {company_name} 재무 분석 보고서

## 요약
(수치 변화를 문장으로 서술. 해석/평가 금지.
★ 서술 순서 (필수): 해당 내용이 있는 경우에만 아래 순서로 작성
1) 손익계산서 항목 (매출액 → 매출총이익 → 영업이익 → 당기순이익 순)
2) 재무상태표 항목 (자산, 부채, 자본, 부채비율 등)
3) 현금흐름표 항목 (영업활동, 투자활동, 재무활동 현금흐름)
해당 재무제표 영역에 이상 발견이 없으면 그 영역은 생략)

## 주요 발견사항

### 1. [항목명]
- **현상**: (위 **현상** 그대로 복사)
- **원인 분석**: (위 **원인 분석** 그대로 복사)

(모든 이상 패턴 동일한 형식으로 반복. 번호는 ### 헤더로, 현상/원인분석은 불릿으로)

## 주석 요약
(주석 원문에서 숫자만 그대로 나열. 추론/해석 금지. 각 항목은 ###로 구분)

### 1. 특수관계자 거래
(있으면 내용 작성, 없으면 이 항목 전체 생략)

### 2. 우발채무/소송
(있으면 내용 작성, 없으면 이 항목 전체 생략)

### 3. 담보/지급보증
(있으면 내용 작성, 없으면 이 항목 전체 생략)

### 4. 자산 취득/처분
(있으면 내용 작성, 없으면 이 항목 전체 생략)
"""

        try:
            response = generate_with_retry(
                self.client, MODEL_PRO, prompt,
                step_name="보고서 생성"
            )
            # FY 형식을 연도 형식으로 변환 (FY2020 → 2020년)
            return convert_fy_to_year(response.text)

        except Exception as e:
            print(f"  [오류] 보고서 생성 실패: {e}")
            return f"보고서 생성 실패: {str(e)}"

    async def _generate_summary(self, original_report: str, company_name: str, validation_mismatches: List[Dict[str, str]] = None, financial_data: Dict[str, Any] = None) -> str:
        """
        원본 보고서를 요약하여 핵심만 추출 (Pro 모델)
        - 주석 요약 제거
        - 불필요한 설명/해석 제거
        - 숫자 기반 직접적 원인→결과만 남김
        - ★ 검증 불일치 수정 지시 포함 (Solution D)
        - ★ 사전 계산된 참조 데이터 포함 (Solution D 강화)
        """
        # ★ 사전 계산된 참조 데이터 구축 — 요약 LLM이 수치를 검증할 수 있도록
        reference_section = ""
        if financial_data:
            ref = self._build_precomputed_reference(financial_data)
            ref_parts = []
            # P2: 단년도 데이터 경고
            if ref.get('single_year'):
                ref_parts.append(f"\n{ref['single_year_warning']}")
            else:
                ref_parts.append("\n## ★★★ 수치 검증 참조 테이블 (절대 기준) ★★★")
                ref_parts.append("아래는 코드로 사전 계산된 정확한 YoY 변동 수치입니다.")
                ref_parts.append("요약 시 변동률(%)과 변동액을 언급할 때 반드시 아래 테이블의 수치를 사용하세요.")
                ref_parts.append("원본 보고서의 수치와 아래 테이블이 다르면, 아래 테이블이 정확합니다.\n")
                if ref.get('vcm_yoy'):
                    ref_parts.append("### [VCM] 주요 항목 YoY 변동")
                    ref_parts.append(ref['vcm_yoy'])
                if ref.get('is_yoy'):
                    ref_parts.append("\n### [IS] 손익계산서 항목 YoY 변동")
                    ref_parts.append(ref['is_yoy'])
                if ref.get('bs_yoy'):
                    ref_parts.append("\n### [BS] 재무상태표 항목 YoY 변동")
                    ref_parts.append(ref['bs_yoy'])
            reference_section = "\n".join(ref_parts)

        # ★ 검증 불일치 수정 지시 구성
        validation_section = ""
        if validation_mismatches:
            correction_lines = []
            correction_lines.append("\n## ★★★ 수치 검증 오류 수정 (최우선) ★★★")
            correction_lines.append("아래 항목들은 원본 재무제표와 대조 검증 결과 불일치가 발견되었습니다.")
            correction_lines.append("요약 시 반드시 원본 수치(source_value)로 교체하세요:")
            for m in validation_mismatches:
                correction_lines.append(f"- **{m['item']}** [{m['type']}]: 보고서 '{m['report_value']}' → 원본({m['source']}) '{m['source_value']}' 으로 수정 [{m['severity']}]")
            validation_section = "\n".join(correction_lines)

        prompt = f"""[{company_name}] 재무 분석 보고서 요약

## 원본 보고서
{original_report}
{validation_section}
{reference_section}

## ★★★ 요약 규칙 (필수) ★★★

### 제거 대상
1. **"주석 요약" 섹션 전체 제거** - 특수관계자 거래, 우발채무, 담보/지급보증, 자산 취득/처분 등 모두 삭제
2. **불필요한 설명 제거**:
   - ❌ "이러한 재무제표 항목 간의 연관성은 ~을 보여줍니다" (결론/해석)
   - ❌ "제시된 재무제표상에서 ~은 확인되지 않았습니다" (없는 것 언급)
   - ❌ "이 현금은 ~에 사용되어 ~에 기여했습니다" (당연한 설명)
   - ❌ "이는 ~의 주된 원인이 되었습니다" (뻔한 인과 설명)
   - ❌ "~로 확인됩니다", "~임을 보여줍니다" (마무리 문구)
   - ❌ 이미 전문가가 알고 있는 일반적 회계 설명
3. **관련 없는 원인 분석 제거**: 해당 항목과 직접 관련 없는 다른 연도/항목 설명 삭제

### 유지 대상
1. **요약 섹션** - 그대로 유지
2. **각 항목의 현상** - 그대로 유지
3. **원인 분석 핵심만**:
   - ✅ 구체적 숫자와 계정과목명
   - ✅ 직접적 원인 (A 때문에 B 발생)
   - ✅ 주석에서 확인된 구체적 거래 내역
   - ✅ 웹 검색으로 확인된 사실

### ★ 숫자 정확성 (절대 규칙) ★
- 원본 보고서의 숫자를 그대로 유지하라. 축약 과정에서 숫자를 변경하거나 새로 만들지 마라.
- **잔액(Balance)과 변동액(Change)을 절대 혼동하지 마라.** "증가", "감소" 뒤에 잔액을 쓰면 안 된다. 반드시 두 시점의 차이(변동액)를 써라.
- 원본에 잔액/변동액 혼동이 있으면 **수정하라**: 예) "매출채권 증가 97,477" → 잔액이 97,477이고 증가분이 아니라면 제거하거나 올바른 변동액으로 교체하라.
- **총유출액(gross)과 순액(net) 혼동이 있으면 수정하라**: 예) "재무활동현금흐름 -14,646" → 이것이 총유출액이고 순액은 -11,618이면, "재무활동으로인한현금유출액 14,646" 또는 "재무활동현금흐름 -11,618"로 정확히 표기하라.
- **원본에 비정상적으로 큰 숫자가 있으면 의심하라**: 예) 비품의처분이 18백만원인데 "비품 처분 17,931백만원"이라고 되어있으면 삭제하라.
- 재무제표에 없는 숫자를 절대 추가하지 마라.
- **근사치가 실제 수치와 크게 차이나면 수정하라**: "약 1,300억원 순증"이 실제 1,836억원이면 "약 1,840억원 순증"으로 교체.
- **★ 퍼센트 변동률 역검증 ★**: 원본에 "X% 증가/감소"가 있으면, 기준값 × (1 + X/100) ≈ 당기값인지 역검증하라. 맞지 않으면 원본 데이터에서 정확한 기준값과 퍼센트를 다시 계산하여 수정하라. 예) "685% 증가"인데 기준값이 1,898이면 1,898 × 7.85 = 14,899 ≠ 24,820이므로 실제 증가율은 (24,820-1,898)/1,898 = 1,208%로 수정.

### 원인 분석 축약 방법
각 항목의 원인 분석을 **1-3문장**으로 축약하라.

**축약 전 (원본):**
"회사가 보유한 자기주식은 12.74억원(2,600주) 규모이며, 이는 '경영상 필요'에 의해 과거에 취득하여 계속 보유 중인 자산으로 2024년 중 취득이나 처분과 같은 변동 내역은 없습니다. 재무제표상 실제 주목할 만한 이상 현상은 2023년에 발생한 대규모 유형자산 처분입니다..."

**축약 후 (요약):**
"자기주식 12.74억원(2,600주)은 과거 취득 후 변동 없이 보유 중입니다."

**축약 전 (원본):**
"FY2023년 재무제표에서 대규모 자산 매각과 신규 임차료 발생이 동시에 확인되며, 이는 자산 매각 후 재임차(Sale and Leaseback) 거래 실행에 따른 것입니다. 손익계산서상 FY2023년 영업외수익으로 유형자산처분이익 133.7억원이 발생했으며... 이러한 재무제표 항목 간의 연관성은 회사가 보유 부동산을 유동화하여 이익과 현금을 확보하고, 동일 자산을 임차하여 사용하는 방식으로 사업구조를 변경했음을 명확히 보여줍니다."

**축약 후 (요약):**
"유형자산처분이익 133.7억원 발생(토지 60.7억원, 건물 49.5억원 처분). 동시에 판관비에 지급임차료 8.15억원 신규 계상되어 Sale & Leaseback 거래로 확인됩니다."

## 출력 형식 (원본 구조 유지, 내용만 축약)

# {company_name} 재무 분석 보고서

## 요약
(원본의 요약 내용을 유지하되, 서술 순서를 아래와 같이 재배치:
★ 서술 순서 (필수): 해당 내용이 있는 경우에만 아래 순서로 작성
1) 손익계산서 항목 (매출액 → 매출총이익 → 영업이익 → 당기순이익 순)
2) 재무상태표 항목 (자산, 부채, 자본, 부채비율 등)
3) 현금흐름표 항목 (영업활동, 투자활동, 재무활동 현금흐름)
해당 재무제표 영역에 이상 발견이 없으면 그 영역은 생략)

## 주요 발견사항

### 1. [항목명]
- **현상**: (원본 그대로 유지)
- **원인 분석**: (1-3문장으로 축약. 숫자와 직접적 원인만)

(모든 항목 동일한 형식으로 반복. 번호는 ### 헤더로, 현상/원인분석은 불릿으로)

**주의: "## 주석 요약" 섹션은 절대 포함하지 마라.**"""

        try:
            response = generate_with_retry(
                self.client, MODEL_PRO, prompt,
                step_name="요약본 생성"
            )
            # FY 형식을 연도 형식으로 변환 (FY2020 → 2020년)
            return convert_fy_to_year(response.text)

        except Exception as e:
            print(f"  [오류] 요약본 생성 실패: {e}")
            return convert_fy_to_year(original_report)  # 실패 시 원본 반환 (변환 적용)

    async def _validate_and_fix_summary(self, summary: str, financial_data: Dict[str, Any], company_name: str) -> str:
        """
        요약본 수치 검증 및 프로그래밍적 보정 (Solution D 강화).

        1단계: _validate_report_numbers()로 요약본의 수치 불일치 감지
        2단계: 불일치가 있으면 프로그래밍적으로 직접 교체 시도
        3단계: 프로그래밍적 교체 불가능한 경우 targeted LLM 보정 패스
        """
        mismatches = self._validate_report_numbers(summary, financial_data)

        if not mismatches:
            print(f"  [요약 검증] 수치 불일치 없음 ✓")
            return summary

        print(f"  [요약 검증] {len(mismatches)}건 불일치 감지, 보정 시작...")

        # 사전 계산된 참조 데이터에서 정확한 값 가져오기
        ref = self._build_precomputed_reference(financial_data)
        source_values = {}

        def parse_yoy_table(yoy_str: str, source_name: str):
            if not yoy_str:
                return
            for line in yoy_str.split('\n'):
                if '|' not in line or line.startswith('-'):
                    continue
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 5 and parts[0] != '항목':
                    item_name = parts[0]
                    try:
                        pct_str = parts[4].replace('%', '').replace('+', '').strip()
                        pct_val = float(pct_str) if pct_str != 'N/A' else None
                        change_str = parts[3].replace(',', '').replace('+', '')
                        change_val = float(change_str)
                        source_values[item_name] = {
                            '변동률': pct_val,
                            '변동액': change_val,
                            'source': source_name,
                            'prev': float(parts[1].replace(',', '')),
                            'last': float(parts[2].replace(',', '')),
                        }
                    except (ValueError, IndexError):
                        continue

        parse_yoy_table(ref.get('vcm_yoy', ''), 'VCM')
        parse_yoy_table(ref.get('is_yoy', ''), 'IS')
        parse_yoy_table(ref.get('bs_yoy', ''), 'BS')

        # 프로그래밍적 교체 시도
        fixed_summary = summary
        fixed_count = 0

        for m in mismatches:
            item = m['item']
            if m['type'] == '변동률':
                # 보고서의 잘못된 퍼센트를 정확한 값으로 교체
                wrong_pct = m['report_value'].replace('%', '')
                correct_pct = m['source_value'].replace('%', '').replace('+', '')

                # 해당 항목 근처에서 잘못된 퍼센트를 찾아 교체
                # 패턴: "항목명...wrong_pct%"
                # 항목명 주변 200자 내에서 검색
                for src_item in source_values:
                    if item in src_item or src_item in item:
                        # 항목명이 나오는 위치를 찾고, 그 근처에서 잘못된 퍼센트를 교체
                        search_start = 0
                        while True:
                            item_pos = fixed_summary.find(src_item, search_start)
                            if item_pos == -1:
                                # 축약된 이름으로도 검색
                                item_pos = fixed_summary.find(item, search_start)
                                if item_pos == -1:
                                    break

                            # 항목명 위치에서 ±200자 범위 내에서 잘못된 퍼센트 검색
                            context_start = max(0, item_pos - 50)
                            context_end = min(len(fixed_summary), item_pos + 200)
                            context = fixed_summary[context_start:context_end]

                            # 잘못된 퍼센트 패턴 매칭
                            wrong_pattern = re.compile(
                                re.escape(wrong_pct) + r'\s*%',
                                re.UNICODE
                            )
                            wrong_match = wrong_pattern.search(context)
                            if wrong_match:
                                # 절대 위치 계산
                                abs_start = context_start + wrong_match.start()
                                abs_end = context_start + wrong_match.end()
                                old_text = fixed_summary[abs_start:abs_end]
                                new_text = f"{correct_pct}%"
                                fixed_summary = fixed_summary[:abs_start] + new_text + fixed_summary[abs_end:]
                                fixed_count += 1
                                print(f"    [교체] {item}: {old_text} → {new_text}")
                                break

                            search_start = item_pos + len(item)
                        break

            elif m['type'] == '변동액':
                # 변동액 교체도 동일한 로직
                wrong_val = m['report_value'].replace('백만원', '')
                correct_val = m['source_value'].replace('백만원', '').replace('+', '').replace(',', '')
                try:
                    correct_formatted = f"{abs(float(correct_val)):,.0f}"
                except ValueError:
                    continue

                for src_item in source_values:
                    if item in src_item or src_item in item:
                        search_start = 0
                        while True:
                            item_pos = fixed_summary.find(src_item, search_start)
                            if item_pos == -1:
                                item_pos = fixed_summary.find(item, search_start)
                                if item_pos == -1:
                                    break

                            context_start = max(0, item_pos - 50)
                            context_end = min(len(fixed_summary), item_pos + 200)
                            context = fixed_summary[context_start:context_end]

                            wrong_pattern = re.compile(
                                re.escape(wrong_val) + r'\s*백만원',
                                re.UNICODE
                            )
                            wrong_match = wrong_pattern.search(context)
                            if wrong_match:
                                abs_start = context_start + wrong_match.start()
                                abs_end = context_start + wrong_match.end()
                                old_text = fixed_summary[abs_start:abs_end]
                                new_text = f"{correct_formatted}백만원"
                                fixed_summary = fixed_summary[:abs_start] + new_text + fixed_summary[abs_end:]
                                fixed_count += 1
                                print(f"    [교체] {item}: {old_text} → {new_text}")
                                break

                            search_start = item_pos + len(item)
                        break

        if fixed_count > 0:
            print(f"  [요약 검증] {fixed_count}/{len(mismatches)}건 프로그래밍적 보정 완료")

        # 남은 불일치 재검증
        remaining = self._validate_report_numbers(fixed_summary, financial_data)
        if remaining:
            print(f"  [요약 검증] {len(remaining)}건 잔여 불일치 → LLM 보정 패스 실행")

            correction_items = []
            for m in remaining:
                correction_items.append(f"- {m['item']}: 현재 '{m['report_value']}' → 정확한 값 '{m['source_value']}' ({m['source']}) 으로 수정")

            fix_prompt = f"""아래 재무 분석 요약본에서 수치 오류를 수정하세요.

## 요약본
{fixed_summary}

## 수정 대상 (코드 검증으로 확인된 오류)
{chr(10).join(correction_items)}

## 규칙
1. 위에 지정된 수치만 정확한 값으로 교체하세요
2. 나머지 텍스트는 절대 변경하지 마세요
3. 형식(마크다운 구조)을 유지하세요
4. 수정된 전체 요약본을 출력하세요"""

            try:
                response = generate_with_retry(
                    self.client, MODEL_PRO, fix_prompt,
                    step_name="요약 보정"
                )
                fixed_summary = convert_fy_to_year(response.text)
                print(f"  [요약 검증] LLM 보정 패스 완료")
            except Exception as e:
                print(f"  [요약 검증] LLM 보정 실패: {e}, 프로그래밍 보정본 사용")

        # 최종 검증
        final_check = self._validate_report_numbers(fixed_summary, financial_data)
        if final_check:
            print(f"  [요약 검증] ⚠ {len(final_check)}건 최종 잔여 불일치 (수동 확인 필요)")
            for m in final_check:
                print(f"    [{m['severity']}] {m['item']}: {m['report_value']} vs {m['source_value']}")
        else:
            print(f"  [요약 검증] 최종 검증 통과 ✓")

        return fixed_summary

    def _format_financial_data(self, financial_data: Dict[str, Any]) -> str:
        """
        재무 데이터를 분석용 문자열로 변환

        ★ 프론트엔드 일관성: vcm_display(백만원 단위)를 우선 사용
        - 프론트 테이블과 AI가 동일한 숫자를 참조
        - vcm_display가 없으면 원본(bs, is, cf) 사용 (fallback)
        """
        result = []

        print(f"[FORMAT] financial_data 키: {list(financial_data.keys())}")

        def format_table(data, name: str, max_rows: int = 100) -> None:
            """테이블 데이터를 문자열로 변환"""
            if data is None:
                return

            print(f"[FORMAT] {name} 타입: {type(data)}, 길이: {len(data) if isinstance(data, list) else 'N/A'}")

            if hasattr(data, 'to_string'):
                result.append(f"\n### {name}")
                result.append(data.to_string())
            elif isinstance(data, list) and len(data) > 0:
                result.append(f"\n### {name}")
                if isinstance(data[0], dict):
                    headers = list(data[0].keys())
                    result.append(" | ".join(str(h) for h in headers))
                    result.append("-" * 80)
                    for row in data[:max_rows]:
                        values = [str(row.get(h, '')) for h in headers]
                        result.append(" | ".join(values))

        # ★ 사전 계산된 참조 데이터 구축 (Solution A+B+C-2 통합)
        ref = self._build_precomputed_reference(financial_data)

        # ★ vcm_display 우선 사용 (프론트엔드와 동일한 데이터, 백만원 단위)
        # ★ 현금흐름표(CF)는 vcm_display에 미포함이므로 별도 추가
        vcm_display = financial_data.get('vcm_display')
        if vcm_display and isinstance(vcm_display, list) and len(vcm_display) > 0:
            print(f"[FORMAT] vcm_display 사용 (프론트와 동일, 백만원 단위)")
            format_table(vcm_display, '[VCM] 재무상태표/손익계산서 (단위: 백만원) — VCM은 합산 카테고리이며 IS 원본과 값이 다를 수 있음', max_rows=150)

            # ★ VCM YoY 변동액/변동률 (사전 계산)
            if ref.get('vcm_yoy'):
                result.append(f"\n### ★ [VCM] 주요 항목 YoY 변동액 (직전년도 대비, 단위: 백만원) ★")
                result.append("★★★ 변동액을 언급할 때 반드시 아래 수치를 사용하라. 잔액을 변동액으로 오인하지 마라. ★★★")
                result.append(ref['vcm_yoy'])

            # ★ IS 원본 세부항목 YoY 변동 (계정과목 정규화 적용 — Solution B)
            if ref.get('is_yoy'):
                result.append(f"\n### ★ [IS원본] 손익계산서 개별 항목 YoY 변동 (계정명 정규화 적용, 단위: 백만원) ★")
                result.append(ref.get('namespace_warnings', ''))
                result.append("★★★ 개별 항목의 변동률(%)은 반드시 아래 수치를 사용하라. VCM 합산값과 혼동하지 마라. ★★★")
                result.append(ref['is_yoy'])

            # ★ BS 원본 세부항목 YoY 변동 (신규 추가 — Solution A 강화)
            if ref.get('bs_yoy'):
                result.append(f"\n### ★ [BS원본] 재무상태표 개별 항목 YoY 변동 (단위: 백만원) ★")
                result.append("★★★ BS 항목(재고자산, 매출채권 등)의 변동액/변동률은 반드시 아래 수치를 사용하라. 직접 계산하지 마라. ★★★")
                result.append(ref['bs_yoy'])

            # ★ 현금흐름표는 vcm_display에 없으므로 원본 사용 (원 단위)
            cf_data = financial_data.get('cf')
            if cf_data is not None:
                print(f"[FORMAT] 현금흐름표 별도 추가 (원 단위)")
                format_table(cf_data, '현금흐름표 (단위: 원)', max_rows=100)
        else:
            # fallback: 원본 재무제표 사용
            print(f"[FORMAT] vcm_display 없음, 원본 bs/is/cf 사용")
            # 재무상태표 (BS) - 전체
            format_table(financial_data.get('bs'), '재무상태표', max_rows=100)

            # 손익계산서 (IS 또는 CIS) - 전체
            is_data = financial_data.get('is') if financial_data.get('is') is not None else financial_data.get('cis')
            format_table(is_data, '손익계산서', max_rows=100)

            # 현금흐름표 (CF) - 전체
            format_table(financial_data.get('cf'), '현금흐름표', max_rows=100)

        formatted = "\n".join(result) if result else "재무 데이터 없음"
        print(f"[FORMAT] 최종 데이터 길이: {len(formatted)} 문자")
        print(f"[FORMAT] 데이터 미리보기:\n{formatted[:500]}...")
        return formatted


# ============================================================
# 테스트 실행
# ============================================================
async def main():
    """테스트 실행"""
    analyzer = FinancialInsightAnalyzer()

    # 테스트용 더미 데이터
    company_info = {
        "corp_name": "테스트기업",
        "induty_code": "32091"
    }

    financial_data = {
        "vcm": [
            {"항목": "매출", "FY2020": 100, "FY2021": 70, "FY2022": 120},
            {"항목": "영업이익", "FY2020": 10, "FY2021": 5, "FY2022": 15},
        ]
    }

    result = await analyzer.analyze(financial_data, company_info)
    print("\n결과:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
