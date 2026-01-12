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
    hint: str           # 조사 방향 힌트 (가설/추정)


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
        update(10, f'[1/4] 업종 파악 중 - {company_name}')
        industry_info = await self._identify_industry(company_info)
        print(f"  → 업종: {industry_info.get('industry', '파악 실패')}")

        # 2단계: 이상 감지 (Pro)
        update(20, '[2/4] 재무제표 이상 패턴 감지 중')
        anomalies = await self._detect_anomalies(financial_data, company_info, industry_info)
        print(f"  → 감지된 이상 패턴: {len(anomalies)}개")

        if not anomalies:
            update(100, '분석 완료 - 이상 패턴 없음')
            return {
                "success": True,
                "company_name": company_name,
                "industry_info": industry_info,
                "anomalies": [],
                "insights": "특이한 이상 패턴이 감지되지 않았습니다.",
                "report": "재무제표가 안정적으로 보입니다."
            }

        # 3단계: 이상 패턴별 웹 리서치 병렬 실행 (Pro+Search)
        update(35, f'[3/4] 웹 리서치 진행 중 - {len(anomalies)}개 병렬 분석')
        search_results = await self._execute_parallel_research(anomalies, company_info, industry_info)
        print(f"  → 완료된 리서치: {len(search_results)}개")

        # 4단계: 종합 보고서 생성 (Pro)
        update(80, '[4/4] 종합 보고서 생성 중')
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
        "context": "영업이익 54억원, 영업외수익 248억원",
        "hint": "영업이익 대비 당기순이익 괴리, 일회성 영업외수익 가능성"
    }},
    {{
        "period": "FY2020-FY2024",
        "item": "자본총계",
        "finding": "5년 연속 자본잠식 (-200억 → -527억)",
        "context": "누적결손금 1,200억원, 상환전환우선주 800억원",
        "hint": "지속적 적자 누적, 재무구조 취약"
    }}
]

주의사항:
- period: 단일 연도("FY2024") 또는 기간("FY2020-FY2024")
- finding: 수치와 변화 사실만 기재
- context: 관련 항목 수치
- hint: 조사 방향 힌트 (가설/추정만, 지시 없이)
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
                    context=a.get('context', ''),
                    hint=a.get('hint', '')
                )
                for a in anomalies_data
            ]

        except Exception as e:
            print(f"  [오류] 이상 감지 실패: {e}")
            return []

    def _build_research_prompt(
        self,
        anomaly: Anomaly,
        company_info: Dict[str, Any],
        industry_info: Dict[str, Any]
    ) -> str:
        """
        이상 패턴별 웹 리서치 프롬프트 생성 (템플릿 기반)
        """
        company_name = company_info.get('corp_name', '')
        industry = industry_info.get('industry', '')
        competitors = industry_info.get('competitors', [])
        competitors_str = ', '.join(competitors[:3]) if competitors else '정보 없음'
        # period에서 연도 추출 (FY2024 또는 FY2020-FY2024 형태)
        period = anomaly.period if anomaly.period else ''
        year = period.replace('FY', '').split('-')[-1] if period else ''  # 마지막 연도 사용

        research_prompt = f"""당신은 M&A 실사 전문가입니다. 아래 재무제표 이상 패턴의 원인을 웹 검색을 통해 조사하고 분석해주세요.

## 회사 정보
- 회사명: {company_name}
- 업종: {industry}
- 경쟁사: {competitors_str}

## 분석 대상 이상 패턴
- 기간: {anomaly.period}
- 항목: {anomaly.item}
- 발견: {anomaly.finding}
- 관련 항목: {anomaly.context}
- 힌트: {anomaly.hint}

## 조사 항목
위 이상 패턴의 원인을 파악하기 위해 다음 관점에서 웹 검색하여 조사해주세요:

1. **기업 고유 원인**: {company_name}의 {year}년 관련 뉴스, 공시, 경영진 결정, 구조조정, M&A, 소송/과징금 등
2. **산업 동향**: {industry} 업계의 {year}년 시장 상황, 경쟁 환경, 원가 변동
3. **거시경제 영향**: {year}년 금리, 환율, 인플레이션, 경기 상황이 해당 업종에 미친 영향
4. **경쟁사 비교**: {competitors_str} 등 경쟁사의 동일 시기 실적과 비교

## 출력
조사 결과를 바탕으로 이상 패턴의 원인을 분석하여 보고해주세요.
가능하면 구체적인 사실과 출처를 포함해주세요."""

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
        def research_one_sync(anomaly: Anomaly) -> SearchResult:
            """동기 함수로 API 호출 (스레드에서 실행)"""
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
                # 2. Pro + Search로 실제 웹 리서치 수행
                print(f"    [웹 리서치 시작] {anomaly.period} {anomaly.item}")

                response = self.client.models.generate_content(
                    model=MODEL_RESEARCH,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())]
                    )
                )

                # 소스 URL 추출 시도
                sources = []
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'grounding_metadata'):
                        metadata = candidate.grounding_metadata
                        if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                            for chunk in metadata.grounding_chunks:
                                if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri'):
                                    sources.append(chunk.web.uri)

                result_text = response.text if response.text else "결과 없음"
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
            f"- {a.period} {a.item}\n  발견: {a.finding}\n  관련항목: {a.context}\n  힌트: {a.hint}"
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
2. 기업 고유 이슈 vs 산업/거시경제 이슈 구분
3. 핵심 인사이트 요약

## 출력 형식
반드시 아래 형식을 정확히 따라 마크다운으로 작성하세요. 다른 섹션을 추가하지 마세요.

# {company_name} 재무 분석 보고서

## 요약
(3줄 이내 핵심 요약)

## 주요 발견사항

### 1. [발견사항 제목]
- **현상**: (무엇이 발생했는지)
- **원인**: (왜 발생했는지)
- **평가**: (기업이슈/산업이슈/거시이슈 중 하나)

### 2. [발견사항 제목]
- **현상**: ...
- **원인**: ...
- **평가**: ...

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
