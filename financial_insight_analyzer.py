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

# Gemini 클라이언트 초기화
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# 모델 설정
MODEL_PRO = "gemini-2.5-pro"  # Pro 모델 (분석용)
MODEL_FLASH = "gemini-2.5-flash"  # Flash 모델 (빠른 처리)
MODEL_RESEARCH = "gemini-2.5-pro"  # 리서치 모델 (검색 + 분석)

# 재시도 설정
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 15  # 초

import time
import re

def generate_with_retry(client_obj, model: str, contents, config=None, max_retries=MAX_RETRIES):
    """
    429 Rate Limit 에러 시 자동 재시도하는 래퍼 함수
    """
    last_error = None

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

            # 429 Rate Limit 에러인지 확인
            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                # 에러 메시지에서 재시도 대기 시간 추출
                retry_match = re.search(r'retry.?in.?(\d+(?:\.\d+)?)', error_str.lower())
                if retry_match:
                    wait_time = float(retry_match.group(1)) + 1  # 여유분 1초 추가
                else:
                    wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)  # 지수 백오프

                print(f"[Rate Limit] 429 에러 발생. {wait_time:.1f}초 후 재시도... (시도 {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # 다른 에러는 즉시 발생
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

        company_name = company_info.get('corp_name', '알 수 없음')
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
        update(80, '[6/7] 종합 보고서(원본) 생성 중')
        report = await self._generate_report(
            financial_data, company_info, industry_info,
            anomalies, search_results, notes_data
        )

        # 7단계: 요약본 생성 (Pro)
        update(90, '[7/7] 요약본 생성 중')
        summary_report = await self._generate_summary(report, company_name)

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

        try:
            # DART API로 주석 추출 (별도 스레드에서 실행)
            loop = asyncio.get_event_loop()
            extractor = DartFinancialExtractor()

            notes_data = await loop.run_in_executor(
                None,
                lambda: extractor.extract_notes(corp_code)
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
        company_name = company_info.get('corp_name', '')
        induty_code = company_info.get('induty_code', '')

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
            response = self.client.models.generate_content(
                model=MODEL_FLASH,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
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
        company_name = company_info.get('corp_name', '')

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
**중요: 모든 금액의 단위는 '원(KRW)'입니다. 예: 1000000000 = 10억원, 100000000 = 1억원**

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
"""

        # 최대 3회 재시도
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=MODEL_PRO,
                    contents=prompt
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
        company_name = company_info.get('corp_name', '')
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

            response = self.client.models.generate_content(
                model=MODEL_PRO,
                contents=prompt
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
        company_name = company_info.get('corp_name', '')
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

### ★★★ 핵심 규칙: 데이터 기반 추론 + 출처 명시 ★★★
- **근거 없는 추측 금지** - 그러나 재무제표 데이터 기반 추론은 **반드시** 수행하라
- 재무제표와 주석에 있는 **숫자를 그대로 인용**
- **웹 검색 내용은 반드시 출처 명시**: "웹 검색 결과에 따르면 ~라고 보도되었습니다"
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
원단위 숫자를 그대로 쓰지 말고, 가독성 있게 변환하라:
- 1억원 이상: **X.X억원** (예: 133,721,400원 → 133.7억원, 100,606,100원 → 100.6억원)
- 1천만원~1억원: **X천만원** (예: 60,807,908원 → 6천만원, 15,296,394원 → 1.5천만원)
- 1천만원 미만: **X백만원** 또는 그대로 (예: 3,637,305원 → 364만원)
절대 60,807,908원처럼 원단위 숫자를 그대로 쓰지 마라.

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
        이상 패턴과 관련된 항목을 중심으로 추출
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

        # 손익계산서 (영업외수익/비용 포함)
        is_data = financial_data.get('is') if financial_data.get('is') is not None else financial_data.get('cis')
        if is_data is not None:
            format_table(is_data, '손익계산서 (영업외수익/비용 항목 주목)')

        # 재무상태표
        bs_data = financial_data.get('bs')
        if bs_data is not None:
            format_table(bs_data, '재무상태표 (자산/부채 변동 주목)')

        # 현금흐름표 (투자/재무활동 포함)
        cf_data = financial_data.get('cf')
        if cf_data is not None:
            format_table(cf_data, '현금흐름표 (투자활동/재무활동 세부항목 주목)')

        if not result:
            return "(재무제표 데이터 없음)"

        return "\n".join(result)

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
            company_name = company_info.get('corp_name', '')
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

                response = self.client.models.generate_content(
                    model=MODEL_RESEARCH,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())]
                    )
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
                return SearchResult(
                    task=task,
                    result=f"리서치 실패: {str(e)}",
                    sources=[]
                )

        # 모든 이상 패턴에 대해 완전 병렬 실행 (ThreadPoolExecutor 사용)
        print(f"  → {len(anomalies)}개 이상 패턴 병렬 웹 리서치 시작 (Pro + Search)")

        loop = asyncio.get_event_loop()
        # 최대 10개 스레드로 동시 실행 보장
        with ThreadPoolExecutor(max_workers=min(len(anomalies), 10)) as executor:
            futures = [loop.run_in_executor(executor, research_one_sync, a) for a in anomalies]
            results = await asyncio.gather(*futures)

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
        company_name = company_info.get('corp_name', '')

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

## ★★★ 숫자 단위 규칙 (필수) ★★★
원단위 숫자를 그대로 쓰지 말고, 가독성 있게 변환하라:
- 1억원 이상: **X.X억원** (예: 133,721,400원 → 133.7억원)
- 1천만원~1억원: **X천만원** (예: 60,807,908원 → 6천만원)
- 1천만원 미만: **X백만원** 또는 그대로 (예: 3,637,305원 → 364만원)
절대 60,807,908원처럼 원단위 숫자를 그대로 쓰지 마라.

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

(모든 이상 패턴 반복. 재작성 금지, 복사만)

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
            response = self.client.models.generate_content(
                model=MODEL_PRO,
                contents=prompt
            )
            # FY 형식을 연도 형식으로 변환 (FY2020 → 2020년)
            return convert_fy_to_year(response.text)

        except Exception as e:
            print(f"  [오류] 보고서 생성 실패: {e}")
            return f"보고서 생성 실패: {str(e)}"

    async def _generate_summary(self, original_report: str, company_name: str) -> str:
        """
        원본 보고서를 요약하여 핵심만 추출 (Pro 모델)
        - 주석 요약 제거
        - 불필요한 설명/해석 제거
        - 숫자 기반 직접적 원인→결과만 남김
        """
        prompt = f"""[{company_name}] 재무 분석 보고서 요약

## 원본 보고서
{original_report}

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

(모든 항목 반복)

**주의: "## 주석 요약" 섹션은 절대 포함하지 마라.**"""

        try:
            response = self.client.models.generate_content(
                model=MODEL_PRO,
                contents=prompt
            )
            # FY 형식을 연도 형식으로 변환 (FY2020 → 2020년)
            return convert_fy_to_year(response.text)

        except Exception as e:
            print(f"  [오류] 요약본 생성 실패: {e}")
            return convert_fy_to_year(original_report)  # 실패 시 원본 반환 (변환 적용)

    def _format_financial_data(self, financial_data: Dict[str, Any]) -> str:
        """재무 데이터를 분석용 문자열로 변환 (원본 재무제표 사용)"""
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
