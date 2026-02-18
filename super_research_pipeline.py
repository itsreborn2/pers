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
    """경쟁사/협력사 정보"""
    name: str
    ticker: Optional[str] = None
    business: str = ""
    reason: str = ""  # 경쟁/협력 관계 이유
    relationship_type: str = "competitor"  # competitor, partner, supplier, customer
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
    source_url: str = ""  # 원본 기사 URL (출처 추적용)


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
    partners_domestic: List[CompetitorInfo] = field(default_factory=list)  # 협력사 (국내)
    partners_international: List[CompetitorInfo] = field(default_factory=list)  # 협력사 (해외)
    mna_domestic: List[MnACase] = field(default_factory=list)
    mna_international: List[MnACase] = field(default_factory=list)
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
    partners_domestic: List[CompetitorInfo] = field(default_factory=list)
    partners_international: List[CompetitorInfo] = field(default_factory=list)
    mna_domestic: List[MnACase] = field(default_factory=list)
    mna_international: List[MnACase] = field(default_factory=list)
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

    # 불필요한 URL 필터링용 도메인
    _JUNK_DOMAINS = {
        # 채용 사이트
        'jobkorea.co.kr', 'saramin.co.kr', 'indeed.com', 'wanted.co.kr',
        'creditjob.co.kr', 'catch.co.kr', 'albamon.com', 'alba.co.kr',
        'linkedin.com', 'glassdoor.com', 'incruit.com', 'jobplanet.co.kr',
        'work.go.kr', 'jasoseol.com',
        # 위키/블로그/커뮤니티
        'namu.wiki', 'wikipedia.org',
        'tistory.com', 'blog.naver.com', 'cafe.naver.com', 'brunch.co.kr',
        # SNS
        'youtube.com', 'instagram.com', 'facebook.com', 'twitter.com',
        'pinterest.com', 'tiktok.com',
        # 쇼핑
        'shopping.naver.com', 'coupang.com', 'gmarket.co.kr',
        '11st.co.kr', 'auction.co.kr',
        # 단순 주가/금융 데이터 페이지
        'stock.naver.com', 'finance.naver.com', 'finance.daum.net',
        'fnguide.com', 'comp.fnguide.com', 'investing.com', 'stockplus.com',
        # 금융/증권 데이터 페이지
        'kind.krx.co.kr', 'dart.fss.or.kr', 'seibro.or.kr', 'thevc.kr',
        # 애널리스트/리포트 데이터 페이지
        'whynotsellreport.com', 'consensus.hankyung.com', 'srim.co.kr',
        # 법률/전문서비스 페이지
        'leeko.com', 'kimchang.com', 'yulchon.com', 'bkl.co.kr', 'shinkim.com',
        # 정적/CDN + 스팸
        'pstatic.net',
    }

    def _filter_search_results(self, items: list, company_name: str = '') -> List[dict]:
        """검색 결과에서 중복 제거 + 불필요한 URL 필터링

        Args:
            items: Firecrawl 검색 결과 아이템 리스트 (Pydantic 모델)
            company_name: 회사명 (관련성 필터링용, 빈 문자열이면 스킵)

        Returns:
            [{'url': str, 'title': str, 'desc': str}, ...] 필터링된 결과
        """
        from urllib.parse import urlparse

        seen_urls = set()
        seen_titles = set()
        filtered = []

        for item in items:
            url = getattr(item, 'url', '') or ''
            title = getattr(item, 'title', None) or (
                getattr(item.metadata, 'title', '') if hasattr(item, 'metadata') and item.metadata else ''
            )
            desc = getattr(item, 'description', None) or getattr(item, 'snippet', None) or (
                getattr(item.metadata, 'og_description', '') if hasattr(item, 'metadata') and item.metadata else ''
            )

            if not url or not title:
                continue

            # URL 중복 제거
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # 제목 중복 제거 (동일 기사 다른 URL)
            title_key = title.strip()[:60]
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            # 불필요 도메인 필터링 (정확한 도메인 suffix 매칭)
            try:
                domain = urlparse(url).netloc.lower()
                if any(domain == junk or domain.endswith('.' + junk) for junk in self._JUNK_DOMAINS):
                    continue
            except Exception:
                pass

            # 회사명 관련성 필터 (company_name이 주어진 경우)
            if company_name:
                clean = company_name.replace('(주)', '').replace('㈜', '').strip()
                combined = (title or '') + ' ' + (desc or '')
                # 정확한 이름 매칭 또는 접미사 제거 후 매칭
                matched = clean in combined
                if not matched and len(clean) >= 4:
                    # 흔한 접미사 제거 후 재매칭 (삼성전자→삼성, 포스코홀딩스→포스코)
                    for suffix in ['홀딩스', '그룹', '전자', '화학', '건설', '산업', '물산', '상사',
                                   '바이오', '제약', '에너지', '테크', '소프트']:
                        if clean.endswith(suffix) and len(clean) > len(suffix) + 1:
                            short_name = clean[:-len(suffix)]
                            if short_name in combined:
                                matched = True
                                break
                if not matched:
                    continue

            date = getattr(item, 'date', None) or ''
            filtered.append({'url': url, 'title': title or '', 'desc': desc or '', 'date': date})

        return filtered

    async def _scrape_urls(self, urls: List[str], max_chars: int = 3000, min_text_chars: int = 0) -> List[dict]:
        """URL 목록을 병렬로 크롤링하여 본문 텍스트 추출

        Args:
            urls: 크롤링할 URL 목록 (중복 자동 제거)
            max_chars: URL당 본문 최대 글자 수

        Returns:
            [{'url': str, 'title': str, 'content': str}, ...]
        """
        def _scrape_one(url):
            try:
                doc = self.firecrawl.scrape(
                    url,
                    formats=["markdown"],
                    only_main_content=True,
                    timeout=15000
                )
                title = getattr(doc.metadata, 'title', '') if doc.metadata else ''
                content = (doc.markdown or '')[:max_chars]
                if not content.strip():
                    return None
                # 품질 검증: 마크다운 문법 제거 후 실제 텍스트 길이 체크
                if min_text_chars > 0:
                    text_only = re.sub(r'\[.*?\]\(.*?\)', '', content)
                    text_only = re.sub(r'[#*\-_|>\[\](){}!]', '', text_only).strip()
                    if len(text_only) < min_text_chars:
                        print(f"  → [SCRAPE] 품질 미달: {url[:60]}... (텍스트 {len(text_only)}자 < {min_text_chars}자)")
                        return None
                print(f"  → [SCRAPE] 성공: {url[:60]}... ({len(content)}자)")
                return {'url': url, 'title': title or '', 'content': content}
            except Exception as e:
                print(f"  → [SCRAPE] 실패: {url[:60]}... ({e})")
                return None

        # 중복 URL 제거
        unique_urls = list(dict.fromkeys(urls))

        if not unique_urls:
            return []

        # 병렬 실행 (asyncio executor)
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, _scrape_one, url) for url in unique_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scraped = [r for r in results if r and not isinstance(r, Exception)]
        print(f"  → [SCRAPE] 완료: {len(scraped)}/{len(unique_urls)}개 성공")
        return scraped

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
                partners_domestic=[],
                partners_international=[],
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
        Step 2: 사업정보, 주요뉴스 병렬 검색

        - 사업 검색: 기업이 영위하는 모든 사업 파악
        - 주요 뉴스: 최근 2년간 주요 이슈 (합병, 매각, 개발 등)
        (공시는 DART 주석 데이터로 대체 — 재무상세 탭에서 표시)
        """
        # 병렬 실행 (2개)
        business_task = self._search_business(company)
        major_news_task = self._search_major_news(company)

        business, major_news = await asyncio.gather(
            business_task, major_news_task
        )

        return Step2Result(
            company=company,
            business=business,
            disclosure=DisclosureResult(),
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
                    title = getattr(item, 'title', None) or (item.metadata.title if hasattr(item, 'metadata') and item.metadata else '')
                    markdown = getattr(item, 'markdown', None) or getattr(item, 'description', None) or (item.metadata.og_description if hasattr(item, 'metadata') and item.metadata else '')
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
        """Firecrawl 검색 → 본문 크롤링 → Gemini 분석으로 공시/뉴스 파악 (최근 2년)"""
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
                tbs=date_range
            )

            # 검색 결과 수집 → 중복 제거 + 불필요 URL 필터링
            raw_items = search_results.web[:15] if search_results and search_results.web else []
            filtered = self._filter_search_results(raw_items, company_name=company.corp_name)

            print(f"  → Firecrawl 공시/뉴스 검색 완료: {len(raw_items)}건 → 필터 후 {len(filtered)}건")

            if not filtered:
                return DisclosureResult()

            # 필터된 URL만 본문 크롤링
            urls_to_scrape = [item['url'] for item in filtered]
            scraped = await self._scrape_urls(urls_to_scrape, max_chars=3000)

            if scraped:
                # 크롤링된 본문으로 컨텍스트 구성
                context = ""
                for i, article in enumerate(scraped, 1):
                    context += f"\n### 기사 {i}: {article['title']}\n"
                    context += f"URL: {article['url']}\n"
                    context += f"{article['content']}\n"
            else:
                # 크롤링 전체 실패 시 snippet fallback
                context = ""
                for item in filtered:
                    context += f"- {item['title']}: {item['desc']}"
                    context += f" [URL: {item['url']}]\n"

            # Gemini로 요약 (본문 기반)
            prompt = f"""다음은 {company.corp_name}의 공시/뉴스 관련 기사입니다. 기사 본문에 실제로 언급된 내용만 기반으로 중요한 공시/뉴스를 추출해주세요.

기업명: {company.corp_name}
검색 기간: {current_year - 2}년 ~ {current_year}년 (최근 2년)

## 기사 내용
{context}

## ★필수 규칙★
1. 기사 본문에 명시적으로 언급된 정보만 추출하세요
2. 기사에 없는 수치, 인명, 금액 등을 추측하거나 추가하지 마세요

## 추출 대상
1. 이사진 및 임원 관련 소식 (누가, 어떤 직책에, 선임/해임)
2. 지배구조 관련 변동 (최대주주 변경, 지분율 변동 등)
3. 주요 사업 변동 (인수, 매각, 합병 등)
4. 대규모 투자/계약
5. 실적 공시 (매출/이익 변동)
6. 법적 이슈 (소송, 과징금 등)

## 제외 대상
- 홍보성 기사
- 단순 주가 뉴스

## 출력 형식 (★반드시 준수★)
각 항목을 아래 형식으로 출력하세요:
- (YYYY.MM) 제목 요약 | 상세: 구체적 내용 1-2문장 [URL: 원본URL]

예시:
- (2024.03) 사외이사 신규 선임 | 상세: 구예원 사외이사 선임, 감사위원회 위원 겸임. 제38기 정기주주총회 결의 [URL: https://...]
- (2024.06) 대표이사 변경 | 상세: 김진국 대표이사 사임, 후임 박민수 선임. 경영전략 변화 예상 [URL: https://...]

규칙:
- 제목은 간결하게 (10자 이내), "| 상세:" 뒤에 구체적 내용 (누가/무엇을/왜/얼마나)
- URL은 기사 URL을 그대로 유지
- 의견/코멘트 없이 사실만 나열
- 최대 10건"""

            response = generate_with_retry(self.gemini, MODEL_FLASH, prompt)
            summary = response.text.strip()
            print(f"  → 공시/뉴스 분석 완료: {len(summary)}자")

            return DisclosureResult(summary=summary)
        except Exception as e:
            print(f"  → 공시 검색 실패: {e}")
            return DisclosureResult()

    async def _search_major_news(self, company: CompanyInput) -> MajorNewsResult:
        """Firecrawl 뉴스 검색 → 중복/불필요 필터링 → 본문 크롤링 → Gemini 분석 (최근 2년)"""
        try:
            all_items = []  # Firecrawl 검색 결과 원본

            # 2년 전 날짜 계산
            from datetime import timedelta
            two_years_ago = datetime.now() - timedelta(days=730)
            today = datetime.now()
            date_range = f"cdr:1,cd_min:{two_years_ago.strftime('%m/%d/%Y')},cd_max:{today.strftime('%m/%d/%Y')}"

            # 회사명에서 (주), ㈜ 제거
            clean_name = company.corp_name.replace('(주)', '').replace('㈜', '').strip()

            def _search_one(query, limit, use_news_source=False):
                """단일 검색 실행 (동기)"""
                try:
                    kwargs = dict(query=query, limit=limit, tbs=date_range)
                    if use_news_source:
                        kwargs['sources'] = ["news"]
                    results = self.firecrawl.search(**kwargs)

                    items = []
                    if results and hasattr(results, 'news') and results.news:
                        items.extend(list(results.news))
                    if results and hasattr(results, 'web') and results.web:
                        items.extend(list(results.web))
                    return items
                except Exception as e:
                    print(f"  → 검색 실패 ({query[:40]}...): {e}")
                    return []

            # PE Due Diligence 기준 검색 쿼리 (20개 카테고리 + 기본 1개)
            # (query, limit, use_news_source) — 기본 검색은 web(넓은 커버리지), 카테고리별은 news 소스만
            search_tasks = [
                # 기본 검색 (web — 넓은 커버리지)
                (clean_name, 20, False),
                # A. Corporate Action & Governance
                (f'"{clean_name}" 인수 합병 매각 M&A 분할', 5, True),
                (f'"{clean_name}" 대표이사 경영진 이사회 사임 선임', 5, True),
                (f'"{clean_name}" 최대주주 지분 매각 자사주 배당 주주', 5, True),
                (f'"{clean_name}" 자회사 계열사 분할 합병 설립 편입', 5, True),
                # B. Financial Events
                (f'"{clean_name}" 실적 매출 영업이익 적자 흑자전환', 5, True),
                (f'"{clean_name}" 유상증자 회사채 자금조달 차입 IPO', 5, True),
                (f'"{clean_name}" 신용등급 워크아웃 기업회생 부도 자본잠식', 5, True),
                (f'"{clean_name}" 감사의견 한정 거절 회계오류 재무제표 정정', 5, True),
                # C. Legal & Regulatory
                (f'"{clean_name}" 소송 과징금 공정위 제재 벌금 판결', 5, True),
                (f'"{clean_name}" 세무조사 추징금 조세소송 국세청', 5, True),
                (f'"{clean_name}" 인허가 면허 규제 허가취소 행정처분', 5, True),
                (f'"{clean_name}" 관계사거래 특수관계자 내부거래 일감몰아주기', 5, True),
                # D. Operational
                (f'"{clean_name}" 대규모 계약 수주 납품 장기계약 해지', 5, True),
                (f'"{clean_name}" 공장 증설 설비투자 부동산 매입 매각', 5, True),
                (f'"{clean_name}" 구조조정 감원 파업 노조 핵심인력 이탈', 5, True),
                (f'"{clean_name}" 주요거래처 고객 납품업체 공급망 거래중단', 5, True),
                # E. External Risk
                (f'"{clean_name}" 환경오염 산업재해 중대재해 제품리콜 안전', 5, True),
                (f'"{clean_name}" 해외진출 수출 해외법인 글로벌 철수 관세', 5, True),
                (f'"{clean_name}" 특허 기술이전 R&D 핵심기술 라이선스 침해', 5, True),
                (f'"{clean_name}" 채무보증 우발채무 담보 연대보증 약정', 5, True),
                # F. Growth & Strategy
                (f'"{clean_name}" 최대실적 사상최대 매출성장 영업이익증가 실적개선', 5, True),
                (f'"{clean_name}" 신사업 신시장 진출 해외확장 동남아 미국 유럽', 5, True),
                (f'"{clean_name}" 업무협약 MOU 전략적제휴 합작 JV 파트너십', 5, True),
                (f'"{clean_name}" 시장점유율 업계1위 경쟁력 수주잔고 수요증가', 5, True),
            ]

            # 전체 병렬 실행 (asyncio.to_thread로 동기 → 비동기 변환)
            loop = asyncio.get_running_loop()
            tasks = [loop.run_in_executor(None, _search_one, q, lim, ns) for q, lim, ns in search_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"  → 병렬 검색 #{i} 예외: {result}")
                elif isinstance(result, list):
                    all_items.extend(result)

            # 1단계: 중복 제거 + 불필요 URL 필터링 (회사명 관련성 포함)
            filtered = self._filter_search_results(all_items, company_name=company.corp_name)

            print(f"  → Firecrawl 주요뉴스 병렬검색 완료: {len(search_tasks)}개 쿼리, {len(all_items)}건 → 필터 후 {len(filtered)}건")

            if not filtered:
                return MajorNewsResult(news_items=[], raw_news="관련 뉴스 없음")

            # 2단계: snippet 기반으로 Gemini가 중요 뉴스 선별
            def _fmt_snippet(i, item):
                date_prefix = "({}) ".format(item['date']) if item.get('date') else ''
                return f"[{i+1}] {date_prefix}{item['title']}: {item['desc'][:150]} [URL: {item['url']}]"

            snippet_context = "\n".join([
                _fmt_snippet(i, item)
                for i, item in enumerate(filtered[:60])
            ])

            select_prompt = f"""다음은 {company.corp_name}에 대한 뉴스 검색 결과 snippet입니다.
PE 실사 관점에서 가장 중요한 뉴스의 번호를 선택해주세요.

## 검색 결과 (snippet)
{snippet_context}

## 선별 기준
중요: 인수/합병/매각, 경영진 변동, 실적 발표, 소송/과징금, 대규모 계약, 구조조정
제외: 채용공고, 홍보성 기사, 단순 주가뉴스, 프로모션/이벤트, 마케팅, CSR
★ 반드시 제외: 법무법인/회계법인 프로필, 증권사 리포트 목록, 투자데이터/주식블로그,
  다른 기업 뉴스에서 해당 기업이 단순 나열된 경우, 기업 홈페이지

## 출력
중요한 뉴스의 번호만 쉼표로 나열하세요 (예: 1,3,5,7,12).
최대 20개:"""

            select_response = generate_with_retry(self.gemini, MODEL_FLASH, select_prompt)
            selected_text = select_response.text.strip() if select_response and select_response.text else ""

            # 선택된 번호 파싱
            selected_indices = []
            for num in re.findall(r'\d+', selected_text):
                idx = int(num) - 1
                if 0 <= idx < len(filtered):
                    selected_indices.append(idx)
            selected_indices = selected_indices[:20]

            # 선택된 URL 목록
            selected_items = [filtered[i] for i in selected_indices] if selected_indices else filtered[:20]

            print(f"  → 주요뉴스 선별 완료: {len(selected_items)}개")

            # 3단계: 선별된 URL 본문 크롤링
            urls_to_scrape = [item['url'] for item in selected_items]
            scraped = await self._scrape_urls(urls_to_scrape, max_chars=3000, min_text_chars=100)

            # 4단계: 크롤링된 본문 + snippet fallback으로 Gemini 정리
            if scraped:
                body_context = ""
                for i, article in enumerate(scraped, 1):
                    body_context += f"\n### 기사 {i}: {article['title']}\n"
                    body_context += f"URL: {article['url']}\n"
                    body_context += f"{article['content']}\n"

                # 크롤링 실패한 URL은 snippet으로 보충
                scraped_urls = {a['url'] for a in scraped}
                snippet_fallback = ""
                for item in selected_items:
                    if item['url'] not in scraped_urls:
                        snippet_fallback += f"- {item['title']}: {item['desc']} [URL: {item['url']}]\n"

                context = f"## 기사 본문 (크롤링 성공)\n{body_context}"
                if snippet_fallback:
                    context += f"\n## 검색 snippet (크롤링 실패, 참고만)\n{snippet_fallback}"
            else:
                # 크롤링 전체 실패 시 snippet fallback
                context = "## 검색 결과 (snippet)\n"
                for item in selected_items:
                    context += f"- {item['title']}: {item['desc']} [URL: {item['url']}]\n"

            # Gemini로 정리 (본문 기반)
            prompt = f"""다음은 {company.corp_name}에 대한 뉴스 기사입니다. 기사 본문에 실제로 언급된 내용만 기반으로 기업 분석에 중요한 뉴스를 정리해주세요.

{context}

## ★필수 규칙★
1. 기사 본문에 명시적으로 언급된 정보만 사용하세요
2. 기사에 없는 수치, 금액, 인명 등을 추측하거나 추가하지 마세요

## 제외 대상 (★반드시 제외★)
- 채용공고, 구인광고, 회사소개 사이트
- 광고/홍보성 기사, 단순 주가/시황 뉴스
- 단순 상품/서비스 출시, 프로모션/이벤트, 마케팅, CSR 활동

## 추출 대상 (PE 실사 관점)
[Corporate Action] 인수/합병/매각/분할, 경영진 변동, 지분 변동
[Financial Events] 실적 발표, 자금 조달, 신용등급, 감사의견
[Legal & Regulatory] 소송/과징금, 세무조사, 인허가
[Operational] 대규모 계약/수주, 설비투자, 구조조정, 공급망 변동
[External Risk] 환경/안전, 해외사업, 특허/기술, 우발채무
[Growth & Strategy] 최대실적, 해외확장, 전략적 제휴, 시장점유율

## 결과 형식
- (YYYY.MM) 뉴스 제목 요약 | 상세: 구체적 내용 1-2문장 [URL: 원본URL]

규칙:
- 제목은 간결하게 (15자 이내), "| 상세:" 뒤에 기사 본문에서 확인된 구체적 수치·인명·금액 포함
- 날짜는 기사에서 추출하여 (YYYY.MM) 형식. 모르면 생략
- URL 유지
- 최대 20건 (카테고리별 균형있게)
- ★ 반드시 최신 날짜순으로 정렬 (가장 최근 → 과거 순서)
- 뉴스가 없으면 '해당 없음'"""

            response = generate_with_retry(self.gemini, MODEL_FLASH, prompt)
            raw_news = response.text.strip() if response and response.text else "정리 실패"
            print(f"  → 주요뉴스 정리 완료: {len(raw_news)}자")

            return MajorNewsResult(
                news_items=[f"- {item['title']}: {item['desc']} [URL: {item['url']}]" for item in filtered],
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

        # 병렬 실행 (국내 경쟁사/협력사, 해외 경쟁사/협력사, 국내 M&A, 해외 M&A)
        domestic_task = self._search_competitors_domestic(step3)
        international_task = self._search_competitors_international(step3)
        mna_domestic_task = self._search_mna_domestic(step3)
        mna_international_task = self._search_mna_international(step3)

        domestic_result, international_result, mna_domestic, mna_international = await asyncio.gather(
            domestic_task, international_task, mna_domestic_task, mna_international_task
        )

        # 각 함수가 (competitors, partners) 튜플 반환
        competitors_domestic, partners_domestic = domestic_result
        competitors_international, partners_international = international_result

        return Step4Result(
            company=step3.company,
            keywords=step3.keywords,
            competitors_domestic=competitors_domestic,
            competitors_international=competitors_international,
            partners_domestic=partners_domestic,
            partners_international=partners_international,
            mna_domestic=mna_domestic,
            mna_international=mna_international,
            previous=step3
        )

    async def _search_competitors_domestic(self, step3: Step3Result) -> tuple:
        """국내 경쟁사 + 협력사 검색 (본문 크롤링 기반)

        Returns:
            (competitors: List[CompetitorInfo], partners: List[CompetitorInfo])
        """
        keywords_str = ", ".join(step3.keywords[:5])
        clean_name = step3.company.corp_name.replace('(주)', '').replace('㈜', '').strip()
        industry = (step3.company.industry or step3.keywords[0]) if step3.keywords else ""

        try:
            all_items = []

            # 경쟁사 + 협력사/파트너 검색어 (확장)
            search_queries = [
                # 경쟁사 검색
                f"{clean_name} 국내 경쟁사 경쟁업체",
                f"{industry} 국내 시장점유율 순위",
                f"{step3.keywords[0] if step3.keywords else ''} 국내 대표 기업",
                f"{industry} 국내 대형사 주요 기업",
                # 협력사/파트너 검색
                f"{clean_name} 협력사 파트너 제휴",
                f"{clean_name} 공급업체 납품업체 벤더",
                f"{clean_name} 주요 고객사 거래처",
            ]

            # 병렬 검색
            loop = asyncio.get_running_loop()

            async def _do_search(query):
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda: self.firecrawl.search(query=query, limit=5)
                    )
                    return result
                except Exception as e:
                    print(f"  → 국내 경쟁사/협력사 검색 실패 ({query[:30]}): {e}")
                    return None

            search_tasks = [_do_search(q) for q in search_queries if q.strip()]
            search_results = await asyncio.gather(*search_tasks)

            for result in search_results:
                if result and hasattr(result, 'web') and result.web:
                    all_items.extend(result.web)

            # 필터링 (중복 제거 + 불필요 도메인 제거)
            filtered = self._filter_search_results(all_items)
            urls = [item['url'] for item in filtered]

            print(f"  → 국내 경쟁사/협력사 검색: {len(all_items)}건 → 필터링 후 {len(filtered)}건")

            # 본문 크롤링
            scraped = await self._scrape_urls(urls, max_chars=3000)

            # 크롤링 기반 context 구성
            if scraped:
                context = ""
                for s in scraped:
                    context += f"\n### {s['title']}\n[URL: {s['url']}]\n{s['content']}\n"
            else:
                # fallback: snippet 기반
                context = ""
                for item in filtered:
                    context += f"\n### {item['title']}\n{item['desc'] or ''}\n"

            print(f"  → 국내 경쟁사/협력사 context: {len(context)}자 ({'본문' if scraped else 'snippet'})")

            # Gemini로 경쟁사 + 협력사 동시 분석
            prompt = f"""다음 기사/검색 결과를 바탕으로 {step3.company.corp_name}의 국내 경쟁사와 협력사를 분석해주세요.

기업명: {step3.company.corp_name}
영위사업: {keywords_str}

## 기사 본문
{context[:15000]}

## 요청
위 기사 본문에서 **국내(한국) 기업**만 추출해주세요. 외국 기업은 제외합니다.

각 기업의 관계를 정확히 분류해주세요:
- **competitor**: 동일 시장에서 경쟁하는 기업
- **partner**: 전략적 제휴, 합작, 공동 개발 등 협력 관계
- **supplier**: 원재료, 부품, 서비스 등을 공급하는 업체
- **customer**: 주요 고객사, 납품처

★ 반드시 기사 본문에 명시된 정보만 사용하세요. 기사에 없는 정보는 추가하지 마세요.

## 결과 형식 (JSON)
[
  {{"name": "기업명", "ticker": "종목코드(상장사만)", "business": "해당 기업의 사업 분야", "reason": "관계 설명 (구체적)", "relationship_type": "competitor|partner|supplier|customer"}}
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
                all_entities = [
                    CompetitorInfo(
                        name=c.get('name', ''),
                        ticker=c.get('ticker'),
                        business=c.get('business', ''),
                        reason=c.get('reason', ''),
                        relationship_type=c.get('relationship_type', 'competitor')
                    )
                    for c in data if isinstance(c, dict) and c.get('name')
                ]

                # 경쟁사/협력사 분리
                competitors = [e for e in all_entities if e.relationship_type == 'competitor']
                partners = [e for e in all_entities if e.relationship_type in ('partner', 'supplier', 'customer')]

                print(f"  → 국내 분석 완료: 경쟁사 {len(competitors)}개, 협력사 {len(partners)}개")

                # 상세 리서치 (경쟁사 + 협력사 합쳐서 상위 10개, 병렬)
                all_for_detail = (competitors[:5] + partners[:5])

                async def _detail_research(comp):
                    try:
                        detail_result = await loop.run_in_executor(
                            None,
                            lambda c=comp: self.firecrawl.search(
                                query=f"{c.name} 사업 영역 주요 사업 매출 규모",
                                limit=3
                            )
                        )
                        if detail_result and hasattr(detail_result, 'web') and detail_result.web:
                            detail_items = self._filter_search_results(detail_result.web)
                            detail_urls = [d['url'] for d in detail_items[:3]]
                            detail_scraped = await self._scrape_urls(detail_urls, max_chars=2000)

                            if detail_scraped:
                                detail_context = "\n".join([f"### {s['title']}\n{s['content']}" for s in detail_scraped])
                            else:
                                detail_context = "\n".join([f"### {d['title']}\n{d['desc']}" for d in detail_items[:3]])

                            if detail_context:
                                detail_prompt = f"""다음은 {comp.name}에 대한 기사입니다. 기사 본문에 있는 정보만 사용하여 핵심 정보를 정리해주세요.

{detail_context[:4000]}

## 정리 요청
1. 주요 사업 영역 (3-5줄)
2. 기업 규모 (매출, 직원수 등 있으면)

간결하게 정리해주세요:"""
                                detail_response = generate_with_retry(self.gemini, MODEL_FLASH, detail_prompt)
                                comp.detailed_business = detail_response.text.strip() if detail_response and detail_response.text else ""
                                print(f"  → 국내 상세: {comp.name} ({len(comp.detailed_business)}자)")
                    except Exception as e:
                        print(f"  → 국내 상세 검색 실패 ({comp.name}): {e}")

                # 병렬 상세 리서치
                if all_for_detail:
                    await asyncio.gather(*[_detail_research(c) for c in all_for_detail])

                return (competitors, partners)
            except json.JSONDecodeError:
                print(f"  → 국내 경쟁사/협력사 JSON 파싱 실패")
                return ([], [])
        except Exception as e:
            print(f"  → 국내 경쟁사/협력사 검색 실패: {e}")
            return ([], [])

    async def _search_competitors_international(self, step3: Step3Result) -> tuple:
        """해외 경쟁사 + 협력사 검색 (본문 크롤링 기반)

        Returns:
            (competitors: List[CompetitorInfo], partners: List[CompetitorInfo])
        """
        keywords_str = ", ".join(step3.keywords[:5])
        clean_name = step3.company.corp_name.replace('(주)', '').replace('㈜', '').strip()
        industry = (step3.company.industry or step3.keywords[0]) if step3.keywords else ""
        industry_en = step3.keywords[0] if step3.keywords else industry

        try:
            all_items = []

            # 경쟁사 + 협력사 검색어 (영문 + 국문 혼합)
            search_queries = [
                # 경쟁사 검색
                f"{industry_en} global market leaders companies",
                f"{industry_en} top international competitors",
                f"{clean_name} global competitors international",
                f"{industry} 해외 글로벌 경쟁사",
                # 협력사/파트너 검색
                f"{clean_name} global partners strategic alliance",
                f"{clean_name} 해외 협력사 파트너 제휴",
                f"{clean_name} supply chain partners vendors",
            ]

            # 병렬 검색
            loop = asyncio.get_running_loop()

            async def _do_search(query):
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda: self.firecrawl.search(query=query, limit=5)
                    )
                    return result
                except Exception as e:
                    print(f"  → 해외 경쟁사/협력사 검색 실패 ({query[:30]}): {e}")
                    return None

            search_tasks = [_do_search(q) for q in search_queries if q.strip()]
            search_results = await asyncio.gather(*search_tasks)

            for result in search_results:
                if result and hasattr(result, 'web') and result.web:
                    all_items.extend(result.web)

            # 필터링 (중복 제거 + 불필요 도메인 제거)
            filtered = self._filter_search_results(all_items)
            urls = [item['url'] for item in filtered]

            print(f"  → 해외 경쟁사/협력사 검색: {len(all_items)}건 → 필터링 후 {len(filtered)}건")

            # 본문 크롤링
            scraped = await self._scrape_urls(urls, max_chars=3000)

            # 크롤링 기반 context 구성
            if scraped:
                context = ""
                for s in scraped:
                    context += f"\n### {s['title']}\n[URL: {s['url']}]\n{s['content']}\n"
            else:
                context = ""
                for item in filtered:
                    context += f"\n### {item['title']}\n{item['desc'] or ''}\n"

            print(f"  → 해외 경쟁사/협력사 context: {len(context)}자 ({'본문' if scraped else 'snippet'})")

            # Gemini로 경쟁사 + 협력사 동시 분석
            prompt = f"""다음 기사/검색 결과를 바탕으로 {step3.company.corp_name}의 해외 경쟁사와 협력사를 분석해주세요.

기업명: {step3.company.corp_name}
영위사업: {keywords_str}

## 기사 본문
{context[:15000]}

## 요청
위 기사 본문에서 **해외(외국) 기업**만 추출해주세요. 한국 기업은 제외합니다.
미국, 유럽, 일본, 중국 등 글로벌 기업을 찾아주세요.

각 기업의 관계를 정확히 분류해주세요:
- **competitor**: 동일 시장에서 경쟁하는 기업
- **partner**: 전략적 제휴, 합작, 공동 개발 등 협력 관계
- **supplier**: 원재료, 부품, 기술 등을 공급하는 업체
- **customer**: 주요 고객사, 바이어

★ 반드시 기사 본문에 명시된 정보만 사용하세요. 기사에 없는 정보는 추가하지 마세요.

## 결과 형식 (JSON)
[
  {{"name": "기업명", "ticker": "티커(상장사만, 예: AMZN, FDX)", "business": "해당 기업의 사업 분야", "reason": "관계 설명 (구체적)", "relationship_type": "competitor|partner|supplier|customer"}}
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
                all_entities = [
                    CompetitorInfo(
                        name=c.get('name', ''),
                        ticker=c.get('ticker'),
                        business=c.get('business', ''),
                        reason=c.get('reason', ''),
                        relationship_type=c.get('relationship_type', 'competitor')
                    )
                    for c in data if isinstance(c, dict) and c.get('name')
                ]

                # 경쟁사/협력사 분리
                competitors = [e for e in all_entities if e.relationship_type == 'competitor']
                partners = [e for e in all_entities if e.relationship_type in ('partner', 'supplier', 'customer')]

                print(f"  → 해외 분석 완료: 경쟁사 {len(competitors)}개, 협력사 {len(partners)}개")

                # 상세 리서치 (경쟁사 + 협력사 합쳐서 상위 10개, 병렬)
                all_for_detail = (competitors[:5] + partners[:5])

                async def _detail_research(comp):
                    try:
                        detail_result = await loop.run_in_executor(
                            None,
                            lambda c=comp: self.firecrawl.search(
                                query=f"{c.name} company business overview revenue",
                                limit=3
                            )
                        )
                        if detail_result and hasattr(detail_result, 'web') and detail_result.web:
                            detail_items = self._filter_search_results(detail_result.web)
                            detail_urls = [d['url'] for d in detail_items[:3]]
                            detail_scraped = await self._scrape_urls(detail_urls, max_chars=2000)

                            if detail_scraped:
                                detail_context = "\n".join([f"### {s['title']}\n{s['content']}" for s in detail_scraped])
                            else:
                                detail_context = "\n".join([f"### {d['title']}\n{d['desc']}" for d in detail_items[:3]])

                            if detail_context:
                                detail_prompt = f"""다음은 {comp.name}에 대한 기사입니다. 기사 본문에 있는 정보만 사용하여 핵심 정보를 정리해주세요.

{detail_context[:4000]}

## 정리 요청
1. 주요 사업 영역 (3-5줄)
2. 기업 규모 (매출, 직원수 등 있으면)

간결하게 정리해주세요:"""
                                detail_response = generate_with_retry(self.gemini, MODEL_FLASH, detail_prompt)
                                comp.detailed_business = detail_response.text.strip() if detail_response and detail_response.text else ""
                                print(f"  → 해외 상세: {comp.name} ({len(comp.detailed_business)}자)")
                    except Exception as e:
                        print(f"  → 해외 상세 검색 실패 ({comp.name}): {e}")

                # 병렬 상세 리서치
                if all_for_detail:
                    await asyncio.gather(*[_detail_research(c) for c in all_for_detail])

                return (competitors, partners)
            except json.JSONDecodeError:
                print(f"  → 해외 경쟁사/협력사 JSON 파싱 실패")
                return ([], [])
        except Exception as e:
            print(f"  → 해외 경쟁사/협력사 검색 실패: {e}")
            return ([], [])

    async def _search_mna_domestic(self, step3: Step3Result) -> List[MnACase]:
        """Firecrawl 검색 → 중복/불필요 필터링 → 본문 크롤링 → Gemini 분석으로 국내 M&A 사례 파악"""
        industry = step3.company.industry or ""
        keywords_str = ", ".join(step3.keywords[:5])

        try:
            # 최근 5년 날짜 범위
            from datetime import timedelta
            five_years_ago = datetime.now() - timedelta(days=1825)
            today = datetime.now()
            date_range = f"cdr:1,cd_min:{five_years_ago.strftime('%m/%d/%Y')},cd_max:{today.strftime('%m/%d/%Y')}"

            # 다양한 검색어로 여러 번 검색 (핵심 키워드 개별 검색) — 병렬
            search_queries = [
                f"{industry} 인수 합병",
                f"{industry} M&A 매각",
                f"{step3.keywords[0] if step3.keywords else ''} 기업 인수",
                f"{step3.keywords[1] if len(step3.keywords) > 1 else ''} 합병 인수합병",
            ]
            search_queries = [q for q in search_queries if q.strip()]

            def _mna_search(query):
                try:
                    results = self.firecrawl.search(query=query, limit=8, tbs=date_range)
                    return list(results.web[:8]) if results and results.web else []
                except Exception as e:
                    print(f"  → 국내 M&A 검색 실패 ({query[:30]}): {e}")
                    return []

            loop = asyncio.get_running_loop()
            search_tasks = [loop.run_in_executor(None, _mna_search, q) for q in search_queries]
            search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

            all_items = []
            for result in search_results_list:
                if isinstance(result, list):
                    all_items.extend(result)

            # 중복 제거 + 불필요 URL 필터링 (M&A는 업종 검색이라 company_name 필터 안 함)
            filtered = self._filter_search_results(all_items)

            print(f"  → Firecrawl 국내 M&A 검색 완료: {len(all_items)}건 → 필터 후 {len(filtered)}건")

            if not filtered:
                return []

            # 필터된 URL만 본문 크롤링
            urls_to_scrape = [item['url'] for item in filtered]
            scraped = await self._scrape_urls(urls_to_scrape, max_chars=3000)

            if scraped:
                # 크롤링된 본문으로 컨텍스트 구성
                context = ""
                for i, article in enumerate(scraped, 1):
                    context += f"\n### 기사 {i}: {article['title']}\n"
                    context += f"URL: {article['url']}\n"
                    context += f"{article['content']}\n"
            else:
                # 크롤링 전체 실패 시 snippet fallback
                print(f"  → 국내 M&A 본문 크롤링 실패, snippet fallback")
                context = ""
                for item in filtered:
                    context += f"- {item['title']}: {item['desc']} [URL: {item['url']}]\n"

            # Gemini로 분석 (본문 기반)
            prompt = f"""다음은 M&A 관련 기사입니다. 기사에 실제로 언급된 국내 M&A(기업인수합병) 사례만 추출해주세요.

키워드: {keywords_str}

## 기사 본문
{context}

## ★필수 규칙★
1. 기사 본문에 명시적으로 언급된 정보만 추출하세요
2. 기사에 없는 정보를 추측하거나 추가하지 마세요
3. 인수가격, 지분율 등은 기사에 나온 경우에만 포함하세요
4. 위 키워드 관련 사업을 영위하는 국내 기업의 M&A만 추출하세요

## 결과 형식 (JSON)
[
  {{
    "acquirer": "인수자",
    "target": "피인수자",
    "date": "YYYY-MM (또는 연도, 기사에 있는 경우만)",
    "price": "인수가격 (기사에 있는 경우만)",
    "stake": "인수지분 (기사에 있는 경우만)",
    "conditions": "인수조건 (기사에 있는 경우만)",
    "details": "기타 핵심 내용 (기사 본문 기반)",
    "source_url": "기사 URL"
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
                        region='domestic',
                        source_url=c.get('source_url', '')
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
        """Firecrawl 검색 → 본문 크롤링 → Gemini 분석으로 해외 M&A 사례 파악"""
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

            # 영문 키워드를 활용한 동적 검색어 생성
            keywords = english_keywords.split(',') if english_keywords else []
            keyword1 = keywords[0].strip() if len(keywords) > 0 else "industry"
            keyword2 = keywords[1].strip() if len(keywords) > 1 else keyword1
            keyword3 = keywords[2].strip() if len(keywords) > 2 else keyword1

            # 다양한 검색어로 여러 번 검색 — 병렬
            search_queries = [
                f"{keyword1} M&A acquisition deal",
                f"{keyword2} merger acquisition transaction",
                f"{keyword3} company acquisition 2024",
                f"{keyword1} {keyword2} M&A deal",
            ]

            def _intl_mna_search(query):
                try:
                    results = self.firecrawl.search(query=query, limit=8, tbs=date_range)
                    return list(results.web[:8]) if results and results.web else []
                except Exception as e:
                    print(f"  → 해외 M&A 검색 실패 ({query[:30]}): {e}")
                    return []

            loop = asyncio.get_running_loop()
            search_tasks = [loop.run_in_executor(None, _intl_mna_search, q) for q in search_queries]
            search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

            all_items = []
            for result in search_results_list:
                if isinstance(result, list):
                    all_items.extend(result)

            # 중복 제거 + 불필요 URL 필터링 (해외 M&A는 company_name 필터 안 함)
            filtered = self._filter_search_results(all_items)

            print(f"  → Firecrawl 해외 M&A 검색 완료: {len(all_items)}건 → 필터 후 {len(filtered)}건")

            if not filtered:
                return []

            # 필터된 URL만 본문 크롤링
            urls_to_scrape = [item['url'] for item in filtered]
            scraped = await self._scrape_urls(urls_to_scrape, max_chars=3000)

            if scraped:
                # 크롤링된 본문으로 컨텍스트 구성
                context = ""
                for i, article in enumerate(scraped, 1):
                    context += f"\n### Article {i}: {article['title']}\n"
                    context += f"URL: {article['url']}\n"
                    context += f"{article['content']}\n"
            else:
                # 크롤링 전체 실패 시 snippet fallback
                print(f"  → 해외 M&A 본문 크롤링 실패, snippet fallback")
                context = ""
                for item in filtered:
                    context += f"- {item['title']}: {item['desc']} [URL: {item['url']}]\n"

            # Gemini로 분석 (본문 기반)
            prompt = f"""Below are full article texts about M&A deals. Extract ONLY M&A cases that are explicitly mentioned in the article text.

Keywords: {english_keywords}

## Article Texts
{context}

## STRICT RULES
1. Extract ONLY information explicitly stated in the article text above
2. Do NOT guess, infer, or add information not present in the articles
3. Include deal value, stake % etc. ONLY if stated in the article
4. Extract international (non-Korean) company M&A cases only

## Result Format (JSON)
[
  {{
    "acquirer": "Acquirer company",
    "target": "Target company",
    "date": "YYYY-MM or year (only if in article)",
    "price": "Deal value (only if in article)",
    "stake": "Stake acquired (only if in article)",
    "conditions": "Deal conditions (only if in article)",
    "details": "Key details from article text",
    "source_url": "Article URL"
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
                        region='international',
                        source_url=c.get('source_url', '')
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

        # 국내 협력사 섹션
        def _format_partner(p):
            rel_label = {'partner': '제휴', 'supplier': '공급업체', 'customer': '고객사'}.get(p.relationship_type, '협력사')
            line = f"### {p.name} ({p.ticker or 'N/A'}) [{rel_label}]\n"
            line += f"- 사업 분야: {p.business}\n"
            line += f"- 관계: {p.reason}\n"
            if p.detailed_business:
                line += f"- 사업 상세:\n{p.detailed_business}\n"
            return line

        partners_domestic_text = ""
        if step4.partners_domestic:
            partners_domestic_text = "\n".join([_format_partner(p) for p in step4.partners_domestic])

        # 해외 협력사 섹션
        partners_intl_text = ""
        if step4.partners_international:
            partners_intl_text = "\n".join([_format_partner(p) for p in step4.partners_international])

        # 국내 M&A 섹션 (출처 URL 포함)
        mna_domestic_text = ""
        if step4.mna_domestic:
            mna_domestic_text = "\n".join([
                f"- {m.acquirer} → {m.target} ({m.date or 'N/A'}): {m.price or 'N/A'}" + (f" [URL: {m.source_url}]" if m.source_url else "")
                for m in step4.mna_domestic
            ])
        # M&A 없으면 빈 문자열 (LLM이 섹션 생략하도록)

        # 해외 M&A 섹션 (출처 URL 포함)
        mna_intl_text = ""
        if step4.mna_international:
            mna_intl_text = "\n".join([
                f"- {m.acquirer} → {m.target} ({m.date or 'N/A'}): {m.price or 'N/A'}" + (f" [URL: {m.source_url}]" if m.source_url else "")
                for m in step4.mna_international
            ])
        # M&A 없으면 빈 문자열 (LLM이 섹션 생략하도록)

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

### 국내 협력사 (파트너/공급업체/고객사)
{partners_domestic_text if partners_domestic_text else ''}

### 해외 협력사 (파트너/공급업체/고객사)
{partners_intl_text if partners_intl_text else ''}

### 국내 M&A 사례
{mna_domestic_text if mna_domestic_text else ''}

### 해외 M&A 사례
{mna_intl_text if mna_intl_text else ''}

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
  "partners": {{
    "domestic": [
      {{
        "name": "기업명",
        "ticker": "종목코드 또는 null",
        "relationship_type": "partner|supplier|customer",
        "field": "사업 분야",
        "reason": "관계 설명",
        "details": "사업 상세 (있으면)"
      }}
    ],
    "international": [
      {{
        "name": "기업명",
        "ticker": "티커 또는 null",
        "relationship_type": "partner|supplier|customer",
        "field": "사업 분야",
        "reason": "관계 설명",
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
        "price": "금액",
        "source_url": "원본 기사 URL (있으면)"
      }}
    ],
    "international": [
      {{
        "acquirer": "인수기업",
        "target": "피인수기업",
        "date": "시기",
        "price": "금액",
        "source_url": "원본 기사 URL (있으면)"
      }}
    ]
  }},
  "news": [
      {{
        "date": "YYYY.MM 또는 빈문자열",
        "title": "간결한 제목 (15자 이내)",
        "description": "핵심 내용 1-2문장 (구체적 수치·인명·금액 포함)",
        "url": "원본 URL 또는 빈문자열"
      }}
    ]
  // ★ news 배열은 반드시 최신 날짜 → 과거 순서 (역순 정렬)
}}
```

## 필수 규칙
1. **반드시 위 JSON 구조 그대로 출력** - 다른 형식 금지
2. **요약/압축/생략 절대 금지** - 원본 데이터를 그대로 사용
3. 정보가 없는 항목은 빈 배열 `[]` 또는 빈 문자열 `""`
4. **JSON만 출력** - 서두 인사말, 설명, ```json 블록 없이 순수 JSON만
5. 모든 문자열은 이스케이프 처리 (특히 따옴표, 줄바꿈)
6. **출처 URL 보존** - M&A의 source_url, 뉴스의 url 필드에 원본 데이터의 [URL: ...] 값을 반드시 포함하라. URL이 없으면 빈 문자열

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
                partners_domestic=step4.partners_domestic,
                partners_international=step4.partners_international,
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
                partners_domestic=step4.partners_domestic,
                partners_international=step4.partners_international,
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
    print(f"국내 경쟁사: {len(result.competitors_domestic)}개")
    print(f"해외 경쟁사: {len(result.competitors_international)}개")
    print(f"국내 협력사: {len(result.partners_domestic)}개")
    print(f"해외 협력사: {len(result.partners_international)}개")
    print(f"국내 M&A: {len(result.mna_domestic)}건")
    print(f"해외 M&A: {len(result.mna_international)}건")

    if result.report:
        print(f"\n보고서 미리보기 (처음 500자):")
        print(result.report[:500])

    return result


if __name__ == "__main__":
    asyncio.run(test_pipeline())
