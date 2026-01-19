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

# Gemini API
from google import genai
from google.genai import types

# 환경변수 로드
load_dotenv()

# Gemini 클라이언트 초기화
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# 모델 설정
MODEL_PRO = "gemini-2.5-pro"  # Pro 모델 (분석용)
MODEL_FLASH = "gemini-2.5-flash"  # Flash 모델 (빠른 처리)
MODEL_RESEARCH = "gemini-2.5-pro"  # 리서치 모델 (검색 + 분석)


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
        update(10, f'[1/5] 업종 파악 중 - {company_name}')
        industry_info = await self._identify_industry(company_info)
        print(f"  → 업종: {industry_info.get('industry', '파악 실패')}")

        # 2단계: 이상 감지 (Pro)
        update(20, '[2/5] 재무제표 이상 패턴 감지 중')
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

        # 3단계: 원인 추적 검색어 생성 (Pro)
        update(30, f'[3/5] 원인 추적 검색어 생성 중 - {len(anomalies)}개 패턴')
        anomalies = await self._generate_search_queries(anomalies, company_info, industry_info)

        # 4단계: 이상 패턴별 웹 리서치 병렬 실행 (Pro+Search)
        update(45, f'[4/5] 웹 리서치 진행 중 - {len(anomalies)}개 병렬 분석')
        search_results = await self._execute_parallel_research(anomalies, company_info, industry_info)
        print(f"  → 완료된 리서치: {len(search_results)}개")

        # 5단계: 종합 보고서 생성 (Pro)
        update(80, '[5/5] 종합 보고서 생성 중')
        report = await self._generate_report(
            financial_data, company_info, industry_info,
            anomalies, search_results
        )

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
            "report": report
        }

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
{financial_summary}

## 분석 관점

M&A 실사 전문가로서 다양한 시각에서 이상 징후를 찾아주세요.
아래는 예시일 뿐이며, 이 외에도 발견되는 모든 이상 패턴을 보고해주세요.

### A. 손익계산서(IS) 분석 예시
- 매출/영업이익/당기순이익 급변동, 흑자↔적자 전환
- 매출원가율/판관비율 이상 변동
- 영업외수익/비용 급증 (일회성 항목)
- 특정 비용 항목 이상 (인건비, 대손상각비 등)

### B. 재무상태표(BS) 분석 예시
- 자산/부채 구조 급변, 부채비율 이상
- 자본잠식, 누적결손금 심화
- 매출채권/재고자산 급증 (부실 징후)
- 충당부채/우발부채 급증 (숨겨진 리스크)

### C. 현금흐름표(CF) 분석 예시
- 영업현금흐름 적자 지속
- 투자/재무 현금흐름 이상 패턴
- 현금 급감

### D. Cross-Check 분석 예시 (재무제표 간 비교)
- [IS↔BS] 매출↑ but 매출채권 더 빠르게↑ → 매출 품질 의심
- [IS↔CF] 당기순이익 흑자 but 영업현금흐름 적자 → 이익의 질 의심
- [BS↔CF] 차입금↑ but 재무CF 불일치 → 숨겨진 거래
- [전체] 다년간 지속 패턴 (3년 연속 적자, 자본잠식 심화 등)

위 예시 외에도 PE 투자자 관점에서 우려되는 모든 이상 징후를 빠짐없이 찾아주세요.

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
- finding: 수치와 변화 사실만 기재
- context: 관련 항목 수치
- 이상 징후가 없으면 빈 배열 [] 반환
"""

        try:
            response = self.client.models.generate_content(
                model=MODEL_PRO,
                contents=prompt
            )

            result_text = response.text
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
            print(f"  [오류] 이상 감지 실패: {e}")
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
        industry_info: Dict[str, Any]
    ) -> str:
        """
        이상 패턴별 웹 리서치 프롬프트 생성 (생성된 검색어 사용)
        """
        company_name = company_info.get('corp_name', '')
        industry = industry_info.get('industry', '')

        # 생성된 검색어 리스트 포맷팅
        search_queries = anomaly.search_queries or []
        search_queries_str = "\n".join([f"- {q}" for q in search_queries]) if search_queries else "- (검색어 없음)"

        research_prompt = f"""당신은 M&A 실사 전문가입니다. 아래 재무제표 이상 패턴의 **원인**을 웹 검색으로 조사해야 합니다.

## [절대 규칙] 사실 기반 응답만 허용
🚫 **절대 금지 사항:**
- 검색 결과 없이 추측하거나 가정하는 것
- "~일 수 있습니다", "~로 추정됩니다" 같은 추론
- 사전 학습된 일반 지식으로 답변하는 것
- 검색에서 찾지 못한 내용을 마치 찾은 것처럼 작성하는 것

✅ **반드시 준수:**
- 오직 웹 검색에서 찾은 **실제 뉴스/기사/공시 내용만** 인용
- 검색 결과가 없으면 솔직하게 "찾지 못했습니다"라고 명시
- 모든 내용에 출처(기사 제목, 날짜, 매체)를 명시

## [필수] 웹 검색 수행 지침
⚠️ **반드시 Google Search 도구로 아래 검색어들을 실제로 검색하세요.**
⚠️ **재무 수치 검색 금지!** 이미 재무제표 데이터를 가지고 있습니다.

## 회사 정보
- 회사명: {company_name}
- 업종: {industry}

## 분석 대상 이상 패턴
- 기간: {anomaly.period}
- 항목: {anomaly.item}
- 발견 사실: {anomaly.finding}
- 관련 항목: {anomaly.context}

## ⭐ 필수 검색어 (아래 검색어들로 검색하세요)
{search_queries_str}

## 출력 형식 (엄격히 준수)

### 검색 결과 요약
[웹 검색에서 찾은 **실제** 뉴스/기사/공시 내용만 요약]
- 반드시 검색에서 찾은 사실만 기재
- 찾지 못한 내용은 "관련 정보를 찾지 못함"으로 명시

### 출처 (필수)
- 출처1: [기사 제목] - [매체명] ([날짜])
- 출처2: [기사 제목] - [매체명] ([날짜])
※ 검색 결과가 없으면 "검색 결과에서 관련 출처를 찾지 못했습니다." 명시

### 분석 결론
[검색 결과에 기반한 사실만 기재. 추측 절대 금지]

⚠️ **검색 결과가 없는 경우**: 반드시 "해당 이상 패턴의 원인을 설명하는 뉴스나 공시를 웹 검색에서 찾지 못했습니다."라고 명시하세요."""

        return research_prompt

    async def _execute_parallel_research(
        self,
        anomalies: List[Anomaly],
        company_info: Dict[str, Any],
        industry_info: Dict[str, Any]
    ) -> List[SearchResult]:
        """
        이상 패턴별 웹 리서치 병렬 실행

        각 이상 패턴에 대해:
        1. 템플릿 기반 리서치 프롬프트 구성
        2. Pro + Search로 실제 웹 리서치 수행

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

            return f"""당신은 M&A 실사 전문가입니다. 아래 회사의 재무 이상 패턴 원인을 넓은 범위에서 웹 검색으로 조사해야 합니다.

## [절대 규칙] 반드시 웹 검색 수행
⚠️ **Google Search 도구를 반드시 사용하세요.**
⚠️ 검색 결과 없이 응답하면 안 됩니다.

## 회사 정보
- 회사명: {company_name}
- 업종: {industry}

## 분석 대상
- 기간: {anomaly.period}
- 항목: {anomaly.item}
- 발견 사실: {anomaly.finding}

## ⭐ 대체 검색어 (반드시 검색)
{queries_str}

## 출력 형식
### 검색 결과 요약
[웹 검색에서 찾은 회사 관련 뉴스/기사 내용]

### 출처
- 출처1: [기사 제목] - [매체명] ([날짜])

### 분석 결론
[검색 결과에 기반한 분석 - 추측 금지]

⚠️ 검색 결과가 전혀 없으면 "관련 정보를 웹 검색에서 찾지 못했습니다."라고 명시하세요."""

        def research_one_sync(anomaly: Anomaly) -> SearchResult:
            """동기 함수로 API 호출 (스레드에서 실행) - Fallback 로직 포함"""
            # 1. 리서치 프롬프트 구성
            print(f"    [리서치 시작] {anomaly.period} {anomaly.item}")
            prompt = self._build_research_prompt(anomaly, company_info, industry_info)

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

                # ★ Fallback 로직: 소스가 없으면 대체 검색어로 재시도
                if not sources:
                    print(f"    [Fallback 시작] {anomaly.period} {anomaly.item} - 소스 없음, 대체 검색어로 재시도")

                    fallback_prompt = build_fallback_prompt(anomaly)
                    fallback_response = self.client.models.generate_content(
                        model=MODEL_RESEARCH,
                        contents=fallback_prompt,
                        config=types.GenerateContentConfig(
                            tools=[types.Tool(google_search=types.GoogleSearch())]
                        )
                    )

                    fallback_sources = extract_sources(fallback_response)
                    fallback_text = fallback_response.text if fallback_response.text else ""

                    if fallback_sources:
                        print(f"    [Fallback 성공] {anomaly.period} {anomaly.item} - {len(fallback_sources)}개 소스 발견")
                        sources = fallback_sources
                        result_text = f"[대체 검색 결과]\n{fallback_text}"
                    else:
                        print(f"    [Fallback 실패] {anomaly.period} {anomaly.item} - 대체 검색도 소스 없음")
                        result_text = f"{result_text}\n\n[참고: 대체 검색도 수행했으나 관련 출처를 찾지 못했습니다.]"

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
        search_results: List[SearchResult]
    ) -> str:
        """
        종합 보고서 생성 (Pro 모델)
        """
        company_name = company_info.get('corp_name', '')

        # 검색 결과 요약 (전체 전달)
        search_summary = ""
        for sr in search_results:
            search_summary += f"\n### {sr.task.query_type}: {sr.task.query}\n"
            search_summary += f"{sr.result}\n"

        anomalies_text = "\n".join([
            f"- {a.period} {a.item}\n  발견: {a.finding}\n  관련항목: {a.context}"
            for a in anomalies
        ])

        prompt = f"""
당신은 PE(사모펀드)의 M&A 실사 보고서를 작성하는 전문가입니다.
아래 정보를 종합하여 투자 검토용 재무 분석 보고서를 작성해주세요.

## 회사 정보
- 회사명: {company_name}
- 업종: {industry_info.get('industry', '')}
- 사업: {industry_info.get('business_description', '')}

## 감지된 이상 패턴
{anomalies_text}

## 조사 결과
{search_summary}

## 보고서 작성 지침
1. 각 이상 패턴에 대해 원인을 명확히 설명
2. 핵심 인사이트 요약

## 출력 형식
반드시 아래 형식을 정확히 따라 마크다운으로 작성하세요. 다른 섹션을 추가하지 마세요.

# {company_name} 재무 분석 보고서

## 요약
(3줄 이내 핵심 요약)

## 주요 발견사항

### 1. [발견사항 제목]
- **현상**: (무엇이 발생했는지)
- **원인**: (왜 발생했는지)

### 2. [발견사항 제목]
- **현상**: ...
- **원인**: ...

(이하 동일한 형식으로 모든 발견사항 작성)

---
보고서는 여기서 끝납니다. "투자 시사점", "추가 확인 필요 사항" 등 다른 섹션을 절대 추가하지 마세요.
"""

        try:
            response = self.client.models.generate_content(
                model=MODEL_PRO,
                contents=prompt
            )
            return response.text

        except Exception as e:
            print(f"  [오류] 보고서 생성 실패: {e}")
            return f"보고서 생성 실패: {str(e)}"

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
        is_data = financial_data.get('is') or financial_data.get('cis')
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
