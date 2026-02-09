"""
슈퍼 리서치 파이프라인

기업에 대한 종합적인 리서치를 수행하는 파이프라인입니다.
랭체인 스타일로 각 단계별 결과를 확인하고 다음 단계로 전달합니다.

파이프라인 구조:
[Step 1] 기업 기본정보 입력
    ↓
[Step 2] 병렬 검색 (사업검색, 공시검색)
    ↓
[Step 3] 키워드 추출
    ↓
[Step 4] 병렬 검색 (경쟁사, 국내M&A, 해외M&A)
    ↓
[Step 5] 보고서 생성 (Pro) → JSON 출력

사용법:
    pipeline = SuperResearchPipeline()

    # 전체 실행
    result = await pipeline.run(company_info, progress_callback)

    # 단계별 실행 (디버깅/확인용)
    step1 = await pipeline.step1_prepare(company_info)
    step2 = await pipeline.step2_parallel_search(step1)
    step3 = await pipeline.step3_extract_keywords(step2)
    step4 = await pipeline.step4_deep_research(step3)
    step5 = await pipeline.step5_generate_report(step4)  # 최종 결과 (JSON)
"""

import os
import asyncio
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv

# Gemini API
from google import genai
from google.genai import types

# Firecrawl API
from firecrawl import FirecrawlApp

# 환경변수 로드
load_dotenv()

# Gemini 클라이언트 초기화
gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# Firecrawl 클라이언트 초기화
firecrawl_client = FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY'))

# 모델 설정
MODEL_PRO = "gemini-3-pro-preview"
MODEL_FLASH = "gemini-3-flash-preview"

# 재시도 설정
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 15


def generate_with_retry(client_obj, model: str, contents, config=None, max_retries=MAX_RETRIES):
    """429 Rate Limit 에러 시 자동 재시도"""
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

            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                retry_match = re.search(r'retry.?in.?(\d+(?:\.\d+)?)', error_str.lower())
                if retry_match:
                    wait_time = float(retry_match.group(1)) + 1
                else:
                    wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)

                print(f"[Rate Limit] {wait_time:.1f}초 후 재시도... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e

    raise last_error


# ============================================================
# 데이터 클래스 정의
# ============================================================

@dataclass
class CompanyInput:
    """Step 1: 기업 기본 입력 정보"""
    corp_code: str
    corp_name: str
    ticker: Optional[str] = None
    market: Optional[str] = None
    industry: Optional[str] = None
    ceo_nm: Optional[str] = None
    est_dt: Optional[str] = None

    @classmethod
    def from_company_info(cls, info: Dict[str, Any]) -> 'CompanyInput':
        return cls(
            corp_code=info.get('corp_code', ''),
            corp_name=info.get('corp_name', ''),
            ticker=info.get('stock_code'),
            market=info.get('market_name'),
            industry=info.get('induty_name'),
            ceo_nm=info.get('ceo_nm'),
            est_dt=info.get('est_dt')
        )


@dataclass
class BusinessSearchResult:
    """사업 검색 결과"""
    raw_business: str  # 원본 사업 정보
    keywords: List[str] = field(default_factory=list)  # 추출된 키워드
    source: str = ""  # 데이터 소스


@dataclass
class DisclosureResult:
    """공시 검색 결과 (최근 2년)"""
    disclosure_items: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""


@dataclass
class MajorNewsResult:
    """주요 뉴스 검색 결과 (최근 2년)"""
    news_items: List[str] = field(default_factory=list)  # 중복 제거된 뉴스 리스트
    raw_news: str = ""  # 정리된 뉴스 텍스트 (보고서용)


@dataclass
class CompetitorInfo:
    """경쟁사 정보"""
    name: str
    ticker: Optional[str] = None
    business: str = ""
    reason: str = ""  # 경쟁사인 이유
    detailed_business: str = ""  # 상세 사업 내용 (Step 2처럼 리서치한 결과)
    scale: str = ""  # 규모 (매출, 직원수 등)


@dataclass
class MnACase:
    """M&A 사례"""
    acquirer: str
    target: str
    date: Optional[str] = None
    price: Optional[str] = None
    stake: Optional[str] = None
    conditions: Optional[str] = None
    details: Optional[str] = None
    region: str = "domestic"  # domestic or international


@dataclass
class Step2Result:
    """Step 2 결과: 병렬 검색"""
    company: CompanyInput
    business: BusinessSearchResult
    disclosure: DisclosureResult
    major_news: MajorNewsResult = field(default_factory=MajorNewsResult)  # 주요 뉴스 (2년)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Step3Result:
    """Step 3 결과: 키워드 추출"""
    company: CompanyInput
    keywords: List[str]
    business_summary: str
    previous: Step2Result = None


@dataclass
class Step4Result:
    """Step 4 결과: 심층 리서치"""
    company: CompanyInput
    keywords: List[str]
    competitors_domestic: List[CompetitorInfo]
    competitors_international: List[CompetitorInfo]
    mna_domestic: List[MnACase]
    mna_international: List[MnACase]
    previous: Step3Result = None


@dataclass
class PipelineResult:
    """최종 파이프라인 결과"""
    success: bool
    company: CompanyInput
    business: BusinessSearchResult
    disclosure: DisclosureResult
    major_news: MajorNewsResult  # 주요 뉴스 (2년)
    keywords: List[str]
    competitors_domestic: List[CompetitorInfo]
    competitors_international: List[CompetitorInfo]
    mna_domestic: List[MnACase]
    mna_international: List[MnACase]
    report: Optional[str] = None
    error: Optional[str] = None


# ============================================================
# 슈퍼 리서치 파이프라인
# ============================================================

class SuperResearchPipeline:
    """
    슈퍼 리서치 파이프라인

    각 단계별로 실행 가능하며, 중간 결과를 확인할 수 있습니다.
    Firecrawl을 통한 웹 검색 + Gemini를 통한 분석을 수행합니다.
    """

    def __init__(self):
        self.gemini = gemini_client
        self.firecrawl = firecrawl_client
        self._progress_callback: Optional[Callable] = None

    def _update(self, progress: int, message: str, step_result: Any = None):
        """진행 상태 업데이트"""
        print(f"[{progress}%] {message}")
        if self._progress_callback:
            self._progress_callback(progress, message, step_result)

    # ========================================================
    # 메인 실행
    # ========================================================

    async def run(
        self,
        company_info: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> PipelineResult:
        """
        전체 파이프라인 실행

        Args:
            company_info: 기업개황정보 딕셔너리
            progress_callback: 진행 상태 콜백 (progress, message, step_result)

        Returns:
            PipelineResult: 최종 결과
        """
        self._progress_callback = progress_callback

        try:
            # Step 1: 기업 정보 준비
            self._update(5, '[Step 1/5] 기업 정보 준비 중')
            step1 = await self.step1_prepare(company_info)
            self._update(10, f'[Step 1/5] 완료: {step1.corp_name}', step1)

            # Step 2: 병렬 검색 (사업검색, 공시검색)
            self._update(12, '[Step 2/5] 사업정보 및 공시 검색 중')
            step2 = await self.step2_parallel_search(step1)
            self._update(30, '[Step 2/5] 완료', step2)

            # Step 3: 키워드 추출
            self._update(32, '[Step 3/5] 사업 키워드 추출 중')
            step3 = await self.step3_extract_keywords(step2)
            self._update(45, f'[Step 3/5] 완료: {len(step3.keywords)}개 키워드', step3)

            # Step 4: 심층 리서치 (경쟁사, M&A)
            self._update(48, '[Step 4/5] 경쟁사 및 M&A 리서치 중')
            step4 = await self.step4_deep_research(step3)
            self._update(70, '[Step 4/5] 완료', step4)

            # Step 5: 보고서 생성
            self._update(72, '[Step 5/5] 종합 보고서 생성 중')
            step5 = await self.step5_generate_report(step4)
            self._update(100, '[완료] 기업 리서치 완료', step5)

            return step5

        except Exception as e:
            error_msg = f"파이프라인 실행 실패: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return PipelineResult(
                success=False,
                company=CompanyInput.from_company_info(company_info),
                business=BusinessSearchResult(raw_business=""),
                disclosure=DisclosureResult(),
                major_news=MajorNewsResult(),
                keywords=[],
                competitors_domestic=[],
                competitors_international=[],
                mna_domestic=[],
                mna_international=[],
                error=error_msg
            )

    # ========================================================
    # Step 1: 기업 정보 준비
    # ========================================================

    async def step1_prepare(self, company_info: Dict[str, Any]) -> CompanyInput:
        """
        Step 1: 기업 기본 정보 준비

        입력된 company_info를 정규화하고 CompanyInput 객체로 변환합니다.
        """
        company = CompanyInput.from_company_info(company_info)

        # 업종 정보가 없으면 AI로 파악
        if not company.industry and company.corp_name:
            industry = await self._identify_industry(company)
            company.industry = industry

        print(f"  → 기업: {company.corp_name}")
        print(f"  → 종목코드: {company.ticker or 'N/A'}")
        print(f"  → 업종: {company.industry or 'N/A'}")

        return company

    async def _identify_industry(self, company: CompanyInput) -> str:
        """Firecrawl 검색 + Gemini로 업종 파악"""
        try:
            # Firecrawl로 기업 정보 검색
            search_results = self.firecrawl.search(
                query=f"{company.corp_name} 업종 사업 분야",
                limit=3
            )

            # 검색 결과 텍스트 추출 (Pydantic 모델 처리)
            context = ""
            if search_results and search_results.web:
                for item in search_results.web[:3]:
                    context += f"- {item.title or ''}: {item.description or ''}\n"

            prompt = f"""다음 검색 결과를 바탕으로 기업의 주요 업종/산업을 한 줄로 간단히 알려주세요.

기업명: {company.corp_name}
종목코드: {company.ticker or 'N/A'}

검색 결과:
{context}

예시: "반도체 제조", "물류/운송", "전자상거래"

업종만 답변하세요:"""

            response = generate_with_retry(self.gemini, MODEL_FLASH, prompt)
            return response.text.strip()
        except Exception as e:
            print(f"  → 업종 파악 실패: {e}")
            return ""

    # ========================================================
    # Step 2: 병렬 검색
    # ========================================================

    async def step2_parallel_search(self, company: CompanyInput) -> Step2Result:
        """
        Step 2: 사업정보, 공시, 주요뉴스 병렬 검색

        - 사업 검색: 기업이 영위하는 모든 사업 파악
        - 공시 검색: 최근 2년 이내 주요 공시
        - 주요 뉴스: 최근 2년간 주요 이슈 (합병, 매각, 개발 등)
        """
        # 병렬 실행 (3개)
        business_task = self._search_business(company)
        disclosure_task = self._search_disclosure(company)
        major_news_task = self._search_major_news(company)

        business, disclosure, major_news = await asyncio.gather(
            business_task, disclosure_task, major_news_task
        )

        return Step2Result(
            company=company,
            business=business,
            disclosure=disclosure,
            major_news=major_news
        )

    async def _search_business(self, company: CompanyInput) -> BusinessSearchResult:
        """Firecrawl 웹 검색 + Gemini 분석으로 사업 정보 파악"""
        try:
            # Firecrawl로 웹 검색 (마크다운 콘텐츠 포함)
            search_results = self.firecrawl.search(
                query=f"{company.corp_name} 사업 영역 주요 사업 제품 서비스",
                limit=5,
                scrape_options={"formats": ["markdown"]}
            )

            # 검색 결과 컨텍스트 구성 (Pydantic 모델 처리)
            context = ""
            if search_results and search_results.web:
                for item in search_results.web[:5]:
                    title = item.metadata.title if item.metadata else ''
                    markdown = item.markdown or (item.metadata.og_description if item.metadata else '')
                    # 마크다운이 너무 길면 자르기
                    if markdown and len(markdown) > 2000:
                        markdown = markdown[:2000] + "..."
                    context += f"\n### {title}\n{markdown or ''}\n"

            print(f"  → Firecrawl 사업 검색 완료: {len(context)}자")

            # Gemini로 분석
            prompt = f"""다음 검색 결과를 바탕으로 기업의 사업 영역을 분석해주세요.

기업명: {company.corp_name}
종목코드: {company.ticker or 'N/A'}

## 검색 결과
{context}

## 분석 요청
1. 주요 사업 영역
2. 영위 중인 모든 사업
3. 진행 중인 R&D 및 신사업
4. 자회사/계열사 사업

## 결과 형식
- 보도자료, 전략적 제휴, 투자 등은 제외
- 실제 영위 중인 사업만 나열
- 각 사업별 간단한 설명 포함

결과:"""

            response = generate_with_retry(self.gemini, MODEL_PRO, prompt)
            raw_business = response.text.strip()
            print(f"  → 사업 분석 완료: {len(raw_business)}자")

            return BusinessSearchResult(
                raw_business=raw_business,
                source="firecrawl"
            )
        except Exception as e:
            print(f"  → 사업 검색 실패: {e}")
            return BusinessSearchResult(raw_business="", source="error")

    async def _search_disclosure(self, company: CompanyInput) -> DisclosureResult:
        """Firecrawl 뉴스 검색으로 공시/뉴스 파악 (최근 2년)"""
        current_year = datetime.now().year

        try:
            # 2년 전 날짜 계산
            from datetime import timedelta
            two_years_ago = datetime.now() - timedelta(days=730)
            today = datetime.now()
            date_range = f"cdr:1,cd_min:{two_years_ago.strftime('%m/%d/%Y')},cd_max:{today.strftime('%m/%d/%Y')}"

            # Firecrawl 뉴스 검색 (최근 2년)
            search_results = self.firecrawl.search(
                query=f"{company.corp_name} 공시 인수 합병 투자 임원 지배구조",
                limit=15,
                tbs=date_range  # 지난 2년
            )

            # 검색 결과 컨텍스트 구성 (Pydantic 모델 처리)
            context = ""
            if search_results and search_results.web:
                for item in search_results.web[:10]:
                    title = item.title if hasattr(item, 'title') else (item.metadata.title if item.metadata else '')
                    desc = item.description if hasattr(item, 'description') else (item.metadata.og_description if item.metadata else '')
                    context += f"- {title or ''}: {desc or ''}\n"

            print(f"  → Firecrawl 공시/뉴스 검색 완료: {len(context)}자")

            # Gemini로 요약
            prompt = f"""다음 검색 결과에서 중요한 공시/뉴스만 추출해주세요.

기업명: {company.corp_name}
검색 기간: {current_year - 2}년 ~ {current_year}년 (최근 2년)

## 검색 결과
{context}

## 추출 대상
1. 이사진 및 임원 관련 소식
2. 지배구조 관련 변동
3. 주요 사업 변동 (인수, 매각 등)
4. 대규모 투자/계약

## 제외 대상
- 홍보성 기사
- 단순 주가 뉴스

결과만 나열하세요 (의견/코멘트 없이):"""

            response = generate_with_retry(self.gemini, MODEL_FLASH, prompt)
            summary = response.text.strip()
            print(f"  → 공시/뉴스 분석 완료: {len(summary)}자")

            return DisclosureResult(summary=summary)
        except Exception as e:
            print(f"  → 공시 검색 실패: {e}")
            return DisclosureResult()

    async def _search_major_news(self, company: CompanyInput) -> MajorNewsResult:
        """Firecrawl 뉴스 검색으로 주요 뉴스 파악 (최근 2년)"""
        try:
            all_news = []

            # 2년 전 날짜 계산
            from datetime import timedelta
            two_years_ago = datetime.now() - timedelta(days=730)
            today = datetime.now()
            date_range = f"cdr:1,cd_min:{two_years_ago.strftime('%m/%d/%Y')},cd_max:{today.strftime('%m/%d/%Y')}"

            # 회사명에서 (주), ㈜ 제거
            clean_name = company.corp_name.replace('(주)', '').replace('㈜', '').strip()

            # 회사명만으로 검색 (최대한 많이 수집)
            try:
                search_results = self.firecrawl.search(
                    query=clean_name,
                    limit=30,  # 많이 가져오기
                    tbs=date_range
                )

                if search_results and search_results.web:
                    for item in search_results.web:
                        title = getattr(item, 'title', None) or (item.metadata.title if hasattr(item, 'metadata') and item.metadata else '')
                        desc = getattr(item, 'description', None) or (item.metadata.og_description if hasattr(item, 'metadata') and item.metadata else '')
                        url = getattr(item, 'url', '') or ''
                        # 제목이나 설명에 회사명이 포함된 것만
                        if title and (clean_name in title or clean_name in (desc or '')):
                            all_news.append(f"- {title}: {desc}")
            except Exception as e:
                print(f"  → 뉴스 검색 실패: {e}")

            print(f"  → Firecrawl 주요뉴스 검색 완료: {len(all_news)}건 수집")

            if not all_news:
                return MajorNewsResult(news_items=[], raw_news="관련 뉴스 없음")

            # 중복 제거
            unique_news = list(set(all_news))
            context = "\n".join(unique_news)

            # Gemini로 정리 (중복 제거 + 주요 이슈만 추출)
            prompt = f"""다음은 {company.corp_name}에 대한 검색 결과입니다. 뉴스 기사만 추출해주세요.

## 검색 결과
{context}

## 제외 대상 (반드시 제외)
- 채용공고, 구인광고
- 회사 홈페이지, 회사소개
- 기업정보 사이트 (잡코리아, 사람인, 크레딧잡 등)
- 광고/홍보성 기사
- 단순 주가/시황 뉴스
- 중복된 내용

## 추출 대상 (뉴스 기사만)
- 인수/합병/매각
- 신규 사업/제품 개발/출시
- 대규모 계약/수주
- 경영진 변동
- 실적 발표, 매출/영업이익 변화
- 법적 이슈/소송/과징금
- 투자 유치/자금 조달
- 사업 확장/축소
- 기타 회사에 중요한 뉴스

## 결과 형식
- 날짜가 있으면 포함
- 각 뉴스를 간결하게 한 줄씩 정리
- 의견/코멘트 없이 사실만 나열
- 뉴스 기사가 없으면 '해당 없음'"""

            response = generate_with_retry(self.gemini, MODEL_FLASH, prompt)
            raw_news = response.text.strip() if response and response.text else "정리 실패"
            print(f"  → 주요뉴스 정리 완료: {len(raw_news)}자")

            return MajorNewsResult(
                news_items=unique_news,
                raw_news=raw_news
            )
        except Exception as e:
            print(f"  → 주요뉴스 검색 실패: {e}")
            return MajorNewsResult()

    # ========================================================
    # Step 3: 키워드 추출
    # ========================================================

    async def step3_extract_keywords(self, step2: Step2Result) -> Step3Result:
        """
        Step 3: 사업 키워드 추출

        Step 2에서 얻은 사업 정보를 분석하여 핵심 키워드를 추출합니다.
        """
        prompt = f"""다음 사업 정보에서 핵심 사업 키워드만 추출해주세요.

## 사업 정보
{step2.business.raw_business}

## 규칙
- 보도자료, 전략적 제휴, 기술개발, 투자, 연구개발 등은 사업이 아님
- 물류, 운송, 택배, 반도체, 전자상거래 등 실제 사업만 추출
- JSON 배열 형식으로 반환

## 결과 형식
["키워드1", "키워드2", "키워드3"]

키워드 배열만 반환하세요:"""

        try:
            response = generate_with_retry(
                self.gemini, MODEL_FLASH, prompt
            )

            text = response.text.strip()

            # JSON 파싱 시도
            try:
                # ```json ... ``` 형식 처리
                if '```' in text:
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
                    if json_match:
                        text = json_match.group(1)

                keywords = json.loads(text)
                if isinstance(keywords, list):
                    keywords = [k for k in keywords if isinstance(k, str)]
                else:
                    keywords = []
            except json.JSONDecodeError:
                # 쉼표로 분리 시도
                keywords = [k.strip().strip('"\'') for k in text.split(',')]
                keywords = [k for k in keywords if k and len(k) < 50]

            print(f"  → 키워드 추출: {keywords}")

            # Step 2 결과의 business에 키워드 저장
            step2.business.keywords = keywords

            return Step3Result(
                company=step2.company,
                keywords=keywords,
                business_summary=step2.business.raw_business[:500],
                previous=step2
            )
        except Exception as e:
            print(f"  → 키워드 추출 실패: {e}")
            return Step3Result(
                company=step2.company,
                keywords=[],
                business_summary="",
                previous=step2
            )

    # ========================================================
    # Step 4: 심층 리서치
    # ========================================================

    async def step4_deep_research(self, step3: Step3Result) -> Step4Result:
        """
        Step 4: 경쟁사 및 M&A 리서치 (병렬)

        키워드별로 국내/해외 경쟁사와 M&A 사례를 검색합니다.
        """
        if not step3.keywords:
            print("  → 키워드 없음, 리서치 스킵")
            return Step4Result(
                company=step3.company,
                keywords=[],
                competitors_domestic=[],
                competitors_international=[],
                mna_domestic=[],
                mna_international=[],
                previous=step3
            )

        # 병렬 실행 (국내 경쟁사, 해외 경쟁사, 국내 M&A, 해외 M&A)
        competitors_domestic_task = self._search_competitors_domestic(step3)
        competitors_international_task = self._search_competitors_international(step3)
        mna_domestic_task = self._search_mna_domestic(step3)
        mna_international_task = self._search_mna_international(step3)

        competitors_domestic, competitors_international, mna_domestic, mna_international = await asyncio.gather(
            competitors_domestic_task, competitors_international_task, mna_domestic_task, mna_international_task
        )

        return Step4Result(
            company=step3.company,
            keywords=step3.keywords,
            competitors_domestic=competitors_domestic,
            competitors_international=competitors_international,
            mna_domestic=mna_domestic,
            mna_international=mna_international,
            previous=step3
        )

    async def _search_competitors_domestic(self, step3: Step3Result) -> List[CompetitorInfo]:
        """Firecrawl 웹 검색 + Gemini 분석으로 국내 경쟁사 파악"""
        keywords_str = ", ".join(step3.keywords[:5])
        clean_name = step3.company.corp_name.replace('(주)', '').replace('㈜', '').strip()
        industry = step3.company.industry or step3.keywords[0] if step3.keywords else ""

        try:
            all_results = []

            # 국내 경쟁사 검색어
            search_queries = [
                f"{clean_name} 국내 경쟁사 경쟁업체",
                f"{industry} 국내 시장점유율 순위",
                f"{step3.keywords[0] if step3.keywords else ''} 국내 대표 기업",
                f"{industry} 국내 대형사 주요 기업",
            ]

            for query in search_queries:
                if not query.strip():
                    continue
                try:
                    search_results = self.firecrawl.search(
                        query=query,
                        limit=5,
                        scrape_options={"formats": ["markdown"]}
                    )
                    if search_results and search_results.web:
                        all_results.extend(search_results.web[:5])
                except Exception as e:
                    print(f"  → 국내 경쟁사 검색 실패 ({query[:30]}): {e}")
                    continue

            # 검색 결과 컨텍스트 구성 (중복 제거)
            seen_titles = set()
            context = ""
            for item in all_results:
                title = item.metadata.title if item.metadata else ''
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                markdown = item.markdown or (item.metadata.og_description if item.metadata else '')
                if markdown and len(markdown) > 1500:
                    markdown = markdown[:1500] + "..."
                context += f"\n### {title}\n{markdown or ''}\n"

            print(f"  → Firecrawl 국내 경쟁사 검색 완료: {len(all_results)}건, {len(context)}자")

            # Gemini로 분석
            prompt = f"""다음 검색 결과를 바탕으로 국내 경쟁사를 분석해주세요.

기업명: {step3.company.corp_name}
영위사업: {keywords_str}

## 검색 결과
{context}

## 요청
위 사업 영역에서 경쟁하는 **국내(한국) 기업**만 추출해주세요.
외국 기업은 제외합니다.

## 결과 형식 (JSON)
[
  {{"name": "경쟁사명", "ticker": "종목코드(상장사만)", "business": "경쟁사업", "reason": "경쟁 이유"}}
]

JSON 배열만 반환하세요:"""

            response = generate_with_retry(self.gemini, MODEL_FLASH, prompt)
            text = response.text.strip()

            # JSON 파싱
            try:
                if '```' in text:
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
                    if json_match:
                        text = json_match.group(1)

                data = json.loads(text)
                competitors = [
                    CompetitorInfo(
                        name=c.get('name', ''),
                        ticker=c.get('ticker'),
                        business=c.get('business', ''),
                        reason=c.get('reason', '')
                    )
                    for c in data if isinstance(c, dict) and c.get('name')
                ]
                print(f"  → 국내 경쟁사 분석 완료: {len(competitors)}개")

                # 각 경쟁사에 대해 상세 리서치 (상위 5개만)
                for i, comp in enumerate(competitors[:5]):
                    try:
                        detail_result = self.firecrawl.search(
                            query=f"{comp.name} 사업 영역 주요 사업 매출 규모",
                            limit=3,
                            scrape_options={"formats": ["markdown"]}
                        )

                        detail_context = ""
                        if detail_result and detail_result.web:
                            for item in detail_result.web[:3]:
                                title = item.metadata.title if item.metadata else ''
                                md = item.markdown or ''
                                if len(md) > 1000:
                                    md = md[:1000] + "..."
                                detail_context += f"### {title}\n{md}\n"

                        if detail_context:
                            detail_prompt = f"""다음은 {comp.name}에 대한 검색 결과입니다. 핵심 정보만 간결하게 정리해주세요.

{detail_context[:4000]}

## 정리 요청
1. 주요 사업 영역 (3-5줄)
2. 기업 규모 (매출, 직원수 등 있으면)

간결하게 정리해주세요:"""

                            detail_response = generate_with_retry(self.gemini, MODEL_FLASH, detail_prompt)
                            comp.detailed_business = detail_response.text.strip() if detail_response and detail_response.text else ""
                            print(f"  → 국내 경쟁사 상세: {comp.name} ({len(comp.detailed_business)}자)")
                    except Exception as e:
                        print(f"  → 국내 경쟁사 상세 검색 실패 ({comp.name}): {e}")

                return competitors
            except json.JSONDecodeError:
                print(f"  → 국내 경쟁사 JSON 파싱 실패")
                return []
        except Exception as e:
            print(f"  → 국내 경쟁사 검색 실패: {e}")
            return []

    async def _search_competitors_international(self, step3: Step3Result) -> List[CompetitorInfo]:
        """Firecrawl 웹 검색 + Gemini 분석으로 해외 경쟁사 파악"""
        keywords_str = ", ".join(step3.keywords[:5])
        clean_name = step3.company.corp_name.replace('(주)', '').replace('㈜', '').strip()
        industry = step3.company.industry or step3.keywords[0] if step3.keywords else ""

        # 업종 키워드 영문 변환용
        industry_en = step3.keywords[0] if step3.keywords else industry

        try:
            all_results = []

            # 해외 경쟁사 검색어 (영문 + 국문 혼합)
            search_queries = [
                f"{industry_en} global market leaders companies",
                f"{industry_en} top international competitors",
                f"{clean_name} global competitors international",
                f"{industry} 해외 글로벌 경쟁사",
            ]

            for query in search_queries:
                if not query.strip():
                    continue
                try:
                    search_results = self.firecrawl.search(
                        query=query,
                        limit=5,
                        scrape_options={"formats": ["markdown"]}
                    )
                    if search_results and search_results.web:
                        all_results.extend(search_results.web[:5])
                except Exception as e:
                    print(f"  → 해외 경쟁사 검색 실패 ({query[:30]}): {e}")
                    continue

            # 검색 결과 컨텍스트 구성 (중복 제거)
            seen_titles = set()
            context = ""
            for item in all_results:
                title = item.metadata.title if item.metadata else ''
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                markdown = item.markdown or (item.metadata.og_description if item.metadata else '')
                if markdown and len(markdown) > 1500:
                    markdown = markdown[:1500] + "..."
                context += f"\n### {title}\n{markdown or ''}\n"

            print(f"  → Firecrawl 해외 경쟁사 검색 완료: {len(all_results)}건, {len(context)}자")

            # Gemini로 분석
            prompt = f"""다음 검색 결과를 바탕으로 해외 경쟁사를 분석해주세요.

기업명: {step3.company.corp_name}
영위사업: {keywords_str}

## 검색 결과
{context}

## 요청
위 사업 영역에서 경쟁하는 **해외(외국) 기업**만 추출해주세요.
한국 기업은 제외합니다. 미국, 유럽, 일본, 중국 등 글로벌 경쟁사를 찾아주세요.

## 결과 형식 (JSON)
[
  {{"name": "경쟁사명", "ticker": "티커(상장사만, 예: AMZN, FDX)", "business": "경쟁사업", "reason": "경쟁 이유"}}
]

JSON 배열만 반환하세요:"""

            response = generate_with_retry(self.gemini, MODEL_FLASH, prompt)
            text = response.text.strip()

            # JSON 파싱
            try:
                if '```' in text:
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
                    if json_match:
                        text = json_match.group(1)

                data = json.loads(text)
                competitors = [
                    CompetitorInfo(
                        name=c.get('name', ''),
                        ticker=c.get('ticker'),
                        business=c.get('business', ''),
                        reason=c.get('reason', '')
                    )
                    for c in data if isinstance(c, dict) and c.get('name')
                ]
                print(f"  → 해외 경쟁사 분석 완료: {len(competitors)}개")

                # 각 경쟁사에 대해 상세 리서치 (상위 5개만)
                for i, comp in enumerate(competitors[:5]):
                    try:
                        detail_result = self.firecrawl.search(
                            query=f"{comp.name} company business overview revenue",
                            limit=3,
                            scrape_options={"formats": ["markdown"]}
                        )

                        detail_context = ""
                        if detail_result and detail_result.web:
                            for item in detail_result.web[:3]:
                                title = item.metadata.title if item.metadata else ''
                                md = item.markdown or ''
                                if len(md) > 1000:
                                    md = md[:1000] + "..."
                                detail_context += f"### {title}\n{md}\n"

                        if detail_context:
                            detail_prompt = f"""다음은 {comp.name}에 대한 검색 결과입니다. 핵심 정보만 간결하게 정리해주세요.

{detail_context[:4000]}

## 정리 요청
1. 주요 사업 영역 (3-5줄)
2. 기업 규모 (매출, 직원수 등 있으면)

간결하게 정리해주세요:"""

                            detail_response = generate_with_retry(self.gemini, MODEL_FLASH, detail_prompt)
                            comp.detailed_business = detail_response.text.strip() if detail_response and detail_response.text else ""
                            print(f"  → 해외 경쟁사 상세: {comp.name} ({len(comp.detailed_business)}자)")
                    except Exception as e:
                        print(f"  → 해외 경쟁사 상세 검색 실패 ({comp.name}): {e}")

                return competitors
            except json.JSONDecodeError:
                print(f"  → 해외 경쟁사 JSON 파싱 실패")
                return []
        except Exception as e:
            print(f"  → 해외 경쟁사 검색 실패: {e}")
            return []

    async def _search_mna_domestic(self, step3: Step3Result) -> List[MnACase]:
        """Firecrawl 뉴스 검색 + Gemini 분석으로 국내 M&A 사례 파악 (다중 검색)"""
        industry = step3.company.industry or ""
        keywords_str = ", ".join(step3.keywords[:5])

        try:
            # 최근 5년 날짜 범위
            from datetime import timedelta
            five_years_ago = datetime.now() - timedelta(days=1825)
            today = datetime.now()
            date_range = f"cdr:1,cd_min:{five_years_ago.strftime('%m/%d/%Y')},cd_max:{today.strftime('%m/%d/%Y')}"

            all_results = []

            # 다양한 검색어로 여러 번 검색 (핵심 키워드 개별 검색)
            search_queries = [
                f"{industry} 인수 합병",
                f"{industry} M&A 매각",
                f"{step3.keywords[0] if step3.keywords else ''} 기업 인수",
                f"{step3.keywords[1] if len(step3.keywords) > 1 else ''} 합병 인수합병",
            ]

            for query in search_queries:
                if not query.strip():
                    continue
                try:
                    search_results = self.firecrawl.search(
                        query=query,
                        limit=8,
                        tbs=date_range
                    )
                    if search_results and search_results.web:
                        all_results.extend(search_results.web[:8])
                except Exception as e:
                    print(f"  → 국내 M&A 검색 실패 ({query[:30]}): {e}")
                    continue

            # 검색 결과 컨텍스트 구성 (중복 제거)
            seen_titles = set()
            context = ""
            for item in all_results:
                title = getattr(item, 'title', None) or (item.metadata.title if hasattr(item, 'metadata') and item.metadata else '')
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                desc = getattr(item, 'description', None) or (item.metadata.og_description if hasattr(item, 'metadata') and item.metadata else '')
                context += f"- {title}: {desc or ''}\n"

            print(f"  → Firecrawl 국내 M&A 검색 완료: {len(all_results)}건, {len(context)}자")

            # Gemini로 분석
            prompt = f"""다음 검색 결과에서 국내 M&A(기업인수합병) 사례를 추출해주세요.

키워드: {keywords_str}

## 검색 결과
{context}

## 요청
위 키워드 관련 사업을 영위하는 국내 기업의 M&A 사례를 추출해주세요.

## 결과 형식 (JSON)
[
  {{
    "acquirer": "인수자",
    "target": "피인수자",
    "date": "YYYY-MM (또는 연도)",
    "price": "인수가격",
    "stake": "인수지분",
    "conditions": "인수조건",
    "details": "기타 내용"
  }}
]

JSON 배열만 반환하세요 (의견/코멘트 없이):"""

            response = generate_with_retry(self.gemini, MODEL_PRO, prompt)
            text = response.text.strip()

            try:
                if '```' in text:
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
                    if json_match:
                        text = json_match.group(1)

                data = json.loads(text)
                cases = [
                    MnACase(
                        acquirer=c.get('acquirer', ''),
                        target=c.get('target', ''),
                        date=c.get('date'),
                        price=c.get('price'),
                        stake=c.get('stake'),
                        conditions=c.get('conditions'),
                        details=c.get('details'),
                        region='domestic'
                    )
                    for c in data if isinstance(c, dict) and c.get('acquirer')
                ]
                print(f"  → 국내 M&A 분석 완료: {len(cases)}건")
                return cases
            except json.JSONDecodeError:
                print(f"  → 국내 M&A JSON 파싱 실패")
                return []
        except Exception as e:
            print(f"  → 국내 M&A 검색 실패: {e}")
            return []

    async def _search_mna_international(self, step3: Step3Result) -> List[MnACase]:
        """Firecrawl 뉴스 검색 + Gemini 분석으로 해외 M&A 사례 파악 (다중 검색)"""
        try:
            # 최근 5년 날짜 범위
            from datetime import timedelta
            five_years_ago = datetime.now() - timedelta(days=1825)
            today = datetime.now()
            date_range = f"cdr:1,cd_min:{five_years_ago.strftime('%m/%d/%Y')},cd_max:{today.strftime('%m/%d/%Y')}"

            # 핵심 키워드만 영문으로 변환 (간결하게)
            core_keywords = step3.keywords[:3] if step3.keywords else []
            translate_prompt = f"Translate to English (words only, comma-separated): {', '.join(core_keywords)}"
            translate_response = generate_with_retry(self.gemini, MODEL_FLASH, translate_prompt)
            english_keywords = translate_response.text.strip()

            all_results = []

            # 영문 키워드를 활용한 동적 검색어 생성
            keywords = english_keywords.split(',') if english_keywords else []
            keyword1 = keywords[0].strip() if len(keywords) > 0 else "industry"
            keyword2 = keywords[1].strip() if len(keywords) > 1 else keyword1
            keyword3 = keywords[2].strip() if len(keywords) > 2 else keyword1

            # 다양한 검색어로 여러 번 검색
            search_queries = [
                f"{keyword1} M&A acquisition deal",
                f"{keyword2} merger acquisition transaction",
                f"{keyword3} company acquisition 2024",
                f"{keyword1} {keyword2} M&A deal",
            ]

            for query in search_queries:
                try:
                    search_results = self.firecrawl.search(
                        query=query,
                        limit=8,
                        tbs=date_range
                    )
                    if search_results and search_results.web:
                        all_results.extend(search_results.web[:8])
                except Exception as e:
                    print(f"  → 해외 M&A 검색 실패 ({query[:30]}): {e}")
                    continue

            # 검색 결과 컨텍스트 구성 (중복 제거)
            seen_titles = set()
            context = ""
            for item in all_results:
                title = getattr(item, 'title', None) or (item.metadata.title if hasattr(item, 'metadata') and item.metadata else '')
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                desc = getattr(item, 'description', None) or (item.metadata.og_description if hasattr(item, 'metadata') and item.metadata else '')
                context += f"- {title}: {desc or ''}\n"

            print(f"  → Firecrawl 해외 M&A 검색 완료: {len(all_results)}건, {len(context)}자")

            # Gemini로 분석
            prompt = f"""Search results below contain international M&A cases. Extract them.

Keywords: {english_keywords}

## Search Results
{context}

## Request
Extract M&A cases of international companies in the above business areas.

## Result Format (JSON)
[
  {{
    "acquirer": "Acquirer company",
    "target": "Target company",
    "date": "YYYY-MM or year",
    "price": "Deal value",
    "stake": "Stake acquired",
    "conditions": "Deal conditions",
    "details": "Additional details"
  }}
]

Return JSON array only (no comments):"""

            response = generate_with_retry(self.gemini, MODEL_PRO, prompt)
            text = response.text.strip()

            try:
                if '```' in text:
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
                    if json_match:
                        text = json_match.group(1)

                data = json.loads(text)
                cases = [
                    MnACase(
                        acquirer=c.get('acquirer', ''),
                        target=c.get('target', ''),
                        date=c.get('date'),
                        price=c.get('price'),
                        stake=c.get('stake'),
                        conditions=c.get('conditions'),
                        details=c.get('details'),
                        region='international'
                    )
                    for c in data if isinstance(c, dict) and c.get('acquirer')
                ]
                print(f"  → 해외 M&A 분석 완료: {len(cases)}건")
                return cases
            except json.JSONDecodeError:
                print(f"  → 해외 M&A JSON 파싱 실패")
                return []
        except Exception as e:
            print(f"  → 해외 M&A 검색 실패: {e}")
            return []

    # ========================================================
    # Step 5: 보고서 생성
    # ========================================================

    async def step5_generate_report(self, step4: Step4Result) -> PipelineResult:
        """
        Step 5: 종합 보고서 생성
        """
        # 국내 경쟁사 섹션 (상세 사업내용 포함)
        competitors_domestic_text = ""
        if step4.competitors_domestic:
            comp_lines = []
            for c in step4.competitors_domestic:
                comp_line = f"### {c.name} ({c.ticker or 'N/A'})\n"
                comp_line += f"- 경쟁 분야: {c.business}\n"
                comp_line += f"- 경쟁 이유: {c.reason}\n"
                if c.detailed_business:
                    comp_line += f"- 사업 상세:\n{c.detailed_business}\n"
                comp_lines.append(comp_line)
            competitors_domestic_text = "\n".join(comp_lines)

        # 해외 경쟁사 섹션 (상세 사업내용 포함)
        competitors_intl_text = ""
        if step4.competitors_international:
            comp_lines = []
            for c in step4.competitors_international:
                comp_line = f"### {c.name} ({c.ticker or 'N/A'})\n"
                comp_line += f"- 경쟁 분야: {c.business}\n"
                comp_line += f"- 경쟁 이유: {c.reason}\n"
                if c.detailed_business:
                    comp_line += f"- 사업 상세:\n{c.detailed_business}\n"
                comp_lines.append(comp_line)
            competitors_intl_text = "\n".join(comp_lines)

        # 국내 M&A 섹션
        mna_domestic_text = ""
        if step4.mna_domestic:
            mna_domestic_text = "\n".join([
                f"- {m.acquirer} → {m.target} ({m.date or 'N/A'}): {m.price or 'N/A'}"
                for m in step4.mna_domestic
            ])
        # M&A 없으면 빈 문자열 (LLM이 섹션 생략하도록)

        # 해외 M&A 섹션
        mna_intl_text = ""
        if step4.mna_international:
            mna_intl_text = "\n".join([
                f"- {m.acquirer} → {m.target} ({m.date or 'N/A'}): {m.price or 'N/A'}"
                for m in step4.mna_international
            ])
        # M&A 없으면 빈 문자열 (LLM이 섹션 생략하도록)

        # 공시 정보
        disclosure_text = step4.previous.previous.disclosure.summary if step4.previous and step4.previous.previous else ""

        # 주요 뉴스 (2년)
        major_news_text = step4.previous.previous.major_news.raw_news if step4.previous and step4.previous.previous and step4.previous.previous.major_news else ""

        # 사업 영역 분석 (raw_business)
        business_text = step4.previous.previous.business.raw_business if step4.previous and step4.previous.previous else ""

        prompt = f"""다음 정보를 JSON 형식의 기업 리서치 보고서로 정리하세요.

## 원본 데이터

### 기업 정보
- 기업명: {step4.company.corp_name}
- 종목코드: {step4.company.ticker or 'N/A'}
- 업종: {step4.company.industry or 'N/A'}

### 사업 영역 분석
{business_text if business_text else ''}

### 영위 사업 키워드
{', '.join(step4.keywords) if step4.keywords else ''}

### 국내 경쟁사 분석
{competitors_domestic_text if competitors_domestic_text else ''}

### 해외 경쟁사 분석
{competitors_intl_text if competitors_intl_text else ''}

### 국내 M&A 사례
{mna_domestic_text if mna_domestic_text else ''}

### 해외 M&A 사례
{mna_intl_text if mna_intl_text else ''}

### 최근 공시/뉴스 (2년)
{disclosure_text if disclosure_text else ''}

### 주요 뉴스 (2년)
{major_news_text if major_news_text else ''}

---

## JSON 출력 형식 (★반드시 이 구조로★)

```json
{{
  "title": "기업명 기업 리서치 보고서",
  "overview": {{
    "기업명": "...",
    "업종": "...",
    "핵심사업": "...(키워드 기반으로 요약)"
  }},
  "business": {{
    "summary": "사업 개요 1-2문장",
    "areas": [
      {{
        "name": "사업영역명",
        "description": "설명 텍스트"
      }}
    ]
  }},
  "competitors": {{
    "domestic": [
      {{
        "name": "회사명",
        "ticker": "종목코드 또는 null",
        "field": "경쟁 분야",
        "reason": "경쟁 이유",
        "details": "사업 상세 (있으면)"
      }}
    ],
    "international": [
      {{
        "name": "회사명",
        "ticker": "종목코드 또는 null",
        "field": "경쟁 분야",
        "reason": "경쟁 이유",
        "details": "사업 상세 (있으면)"
      }}
    ]
  }},
  "mna": {{
    "domestic": [
      {{
        "acquirer": "인수기업",
        "target": "피인수기업",
        "date": "시기",
        "price": "금액"
      }}
    ],
    "international": [
      {{
        "acquirer": "인수기업",
        "target": "피인수기업",
        "date": "시기",
        "price": "금액"
      }}
    ]
  }},
  "news": "주요 뉴스/공시 내용 (텍스트)"
}}
```

## 필수 규칙
1. **반드시 위 JSON 구조 그대로 출력** - 다른 형식 금지
2. **요약/압축/생략 절대 금지** - 원본 데이터를 그대로 사용
3. 정보가 없는 항목은 빈 배열 `[]` 또는 빈 문자열 `""`
4. **JSON만 출력** - 서두 인사말, 설명, ```json 블록 없이 순수 JSON만
5. 모든 문자열은 이스케이프 처리 (특히 따옴표, 줄바꿈)

JSON:"""

        try:
            response = generate_with_retry(
                self.gemini, MODEL_PRO, prompt
            )

            raw_response = response.text.strip()

            # JSON 추출 (```json 블록 처리)
            json_text = raw_response

            # ```json ... ``` 블록 추출
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_response)
            if json_match:
                json_text = json_match.group(1).strip()
            else:
                # 서두 문구 제거 후 JSON 시작점 찾기
                unwanted_prefixes = [
                    "물론입니다", "네,", "네.", "알겠습니다", "다음과 같이",
                    "아래와 같이", "작성했습니다", "작성해 드리겠습니다",
                    "제공해주신", "요청하신"
                ]
                lines = json_text.split('\n')
                while lines and any(lines[0].strip().startswith(p) for p in unwanted_prefixes):
                    lines.pop(0)
                while lines and (lines[0].strip() == '' or lines[0].strip() == '---'):
                    lines.pop(0)
                json_text = '\n'.join(lines).strip()

            # JSON 파싱 시도
            report = None
            try:
                report = json.loads(json_text)
                print(f"  → 보고서 JSON 파싱 완료")
            except json.JSONDecodeError as je:
                print(f"  → JSON 파싱 실패: {je}, 텍스트로 반환")
                # JSON 파싱 실패 시 원본 텍스트 반환 (fallback)
                report = {"error": "JSON 파싱 실패", "raw_text": json_text}

            print(f"  → 보고서 생성 완료: {len(json_text)}자")

            # 이전 단계 결과에서 데이터 수집
            business = step4.previous.previous.business if step4.previous and step4.previous.previous else BusinessSearchResult(raw_business="")
            disclosure = step4.previous.previous.disclosure if step4.previous and step4.previous.previous else DisclosureResult()
            major_news = step4.previous.previous.major_news if step4.previous and step4.previous.previous else MajorNewsResult()

            return PipelineResult(
                success=True,
                company=step4.company,
                business=business,
                disclosure=disclosure,
                major_news=major_news,
                keywords=step4.keywords,
                competitors_domestic=step4.competitors_domestic,
                competitors_international=step4.competitors_international,
                mna_domestic=step4.mna_domestic,
                mna_international=step4.mna_international,
                report=report
            )
        except Exception as e:
            print(f"  → 보고서 생성 실패: {e}")
            return PipelineResult(
                success=False,
                company=step4.company,
                business=step4.previous.previous.business if step4.previous and step4.previous.previous else BusinessSearchResult(raw_business=""),
                disclosure=step4.previous.previous.disclosure if step4.previous and step4.previous.previous else DisclosureResult(),
                major_news=step4.previous.previous.major_news if step4.previous and step4.previous.previous else MajorNewsResult(),
                keywords=step4.keywords,
                competitors_domestic=step4.competitors_domestic,
                competitors_international=step4.competitors_international,
                mna_domestic=step4.mna_domestic,
                mna_international=step4.mna_international,
                error=str(e)
            )

# ============================================================
# 테스트 함수
# ============================================================

async def test_pipeline():
    """파이프라인 테스트"""
    pipeline = SuperResearchPipeline()

    # 테스트 기업 정보
    company_info = {
        'corp_code': '00126380',
        'corp_name': '삼성전자',
        'stock_code': '005930',
        'market_name': 'KOSPI',
    }

    def progress_callback(progress, message, step_result=None):
        print(f"\n>>> Progress: {progress}% - {message}")
        if step_result:
            print(f">>> Step Result Type: {type(step_result).__name__}")

    result = await pipeline.run(company_info, progress_callback)

    print("\n" + "="*60)
    print("최종 결과")
    print("="*60)
    print(f"성공: {result.success}")
    print(f"기업: {result.company.corp_name}")
    print(f"키워드: {result.keywords}")
    print(f"경쟁사: {len(result.competitors)}개")
    print(f"국내 M&A: {len(result.mna_domestic)}건")
    print(f"해외 M&A: {len(result.mna_international)}건")

    if result.report:
        print(f"\n보고서 미리보기 (처음 500자):")
        print(result.report[:500])

    return result


if __name__ == "__main__":
    asyncio.run(test_pipeline())
