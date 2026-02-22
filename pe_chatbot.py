"""
PE 전용 AI 챗봇 모듈

DART 재무제표 데이터 기반의 PE(Private Equity) 전문 AI 어시스턴트.
기업개황 + 재무데이터를 컨텍스트로 활용하여 정확한 답변을 제공하고,
Firecrawl을 통한 웹 검색도 지원한다.
"""

import os
import re
import asyncio
import time
import json
from typing import AsyncGenerator, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

# Firecrawl 클라이언트
from firecrawl import FirecrawlApp

# ============================================================
# 클라이언트 초기화
# ============================================================
API_TIMEOUT_MS = 300_000

gemini_client = genai.Client(
    api_key=os.getenv('GEMINI_API_KEY'),
    http_options=types.HttpOptions(timeout=API_TIMEOUT_MS)
)

firecrawl_client = FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY'))

MODEL_PRO = "gemini-3-pro-preview"
MODEL_FLASH = "gemini-3-flash-preview"

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 10

MAX_CONVERSATION_TURNS = 50

# 검색 제외 도메인
_JUNK_DOMAINS = {
    'jobkorea.co.kr', 'saramin.co.kr', 'indeed.com', 'wanted.co.kr',
    'creditjob.co.kr', 'catch.co.kr', 'albamon.com', 'alba.co.kr',
    'linkedin.com', 'glassdoor.com', 'incruit.com', 'jobplanet.co.kr',
    'work.go.kr', 'jasoseol.com',
    'namu.wiki', 'wikipedia.org',
    'tistory.com', 'blog.naver.com', 'cafe.naver.com', 'brunch.co.kr',
    'youtube.com', 'instagram.com', 'facebook.com', 'twitter.com',
    'pinterest.com', 'tiktok.com',
    'shopping.naver.com', 'coupang.com', 'gmarket.co.kr',
    '11st.co.kr', 'auction.co.kr',
    'stock.naver.com', 'finance.naver.com', 'finance.daum.net',
    'fnguide.com', 'comp.fnguide.com', 'investing.com', 'stockplus.com',
    'kind.krx.co.kr', 'dart.fss.or.kr', 'seibro.or.kr', 'thevc.kr',
    'pstatic.net',
}


def sanitize_for_prompt(text: str, max_length: int = 500) -> str:
    """사용자 입력 새니타이징 (프롬프트 인젝션 방지)"""
    if not text:
        return text
    text = str(text)[:max_length]
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    injection_patterns = [
        r'(?i)\b(ignore|disregard|forget)\s+(previous|above|all)\s+(instructions?|prompts?|rules?)',
        r'(?i)\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be|new\s+instructions?)',
        r'(?i)\b(system\s*:?\s*prompt|<<\s*sys|<\|im_start\|>)',
        r'```\s*(system|instruction)',
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, '[FILTERED]', text)
    return text.strip()


def generate_with_retry(model: str, contents, config=None, max_retries=MAX_RETRIES, step_name: str = ""):
    """429/타임아웃 에러 시 자동 재시도"""
    last_error = None
    label = f"[PE챗봇-{step_name}] " if step_name else "[PE챗봇] "

    for attempt in range(max_retries):
        try:
            if config:
                return gemini_client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            else:
                return gemini_client.models.generate_content(
                    model=model, contents=contents
                )
        except Exception as e:
            error_str = str(e)
            last_error = e

            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                retry_match = re.search(r'retry.?in.?(\d+(?:\.\d+)?)', error_str.lower())
                wait_time = float(retry_match.group(1)) + 1 if retry_match else INITIAL_RETRY_DELAY * (2 ** attempt)
                print(f"{label}[Rate Limit] {wait_time:.1f}초 후 재시도 ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            elif '503' in error_str or 'UNAVAILABLE' in error_str:
                wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                print(f"{label}[503 Unavailable] {wait_time:.1f}초 후 재시도 ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            elif 'timeout' in error_str.lower() or 'ReadTimeout' in error_str:
                wait_time = 10 * (attempt + 1)
                print(f"{label}[Timeout] {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e

    raise last_error


# ============================================================
# PEChatbot 클래스
# ============================================================
class PEChatbot:
    """PE 전문 AI 챗봇"""

    def __init__(self, company_info: dict, preview_data: dict,
                 analysis_result=None, research_result=None):
        self.company_info = company_info or {}
        self.preview_data = preview_data or {}
        self.analysis_result = analysis_result
        self.research_result = research_result
        self.conversation_history = []
        self.company_name = self.company_info.get('corp_name', '알 수 없음')
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """PE 전문 시스템 프롬프트 + 재무데이터 컨텍스트"""
        financial_context = self._build_financial_context()
        company_context = self._build_company_context()
        analysis_context = self._build_analysis_context()

        return f"""당신은 대한민국 PE(Private Equity) 딜 소싱 및 실사 전문 AI 어시스턴트입니다.

## 역할 및 전문성
- PE 펀드의 인수 후보 기업 평가를 위한 재무 분석 전문가
- EV/EBITDA, Net Debt, NWC, 레버리지 비율 등 PE 핵심 지표 중심 응답
- 인수 후보 평가, 위험 요인, 밸류에이션 관점을 자동 반영
- 모든 답변은 한국어로 작성

## 분석 대상 기업
{company_context}

## 참조 재무 데이터 (DART 공시 기준)
{financial_context}

{analysis_context}

## 반할루시네이션 규칙 (★절대 준수★)
1. 위 재무 데이터에 없는 수치는 절대 생성하지 마라
2. 수치를 인용할 때 반드시 연도와 출처를 명시하라 (예: "2023년 DART 기준 매출 1,234백만원")
3. 데이터에서 직접 확인할 수 없는 사항은 "DART 데이터에서 확인 불가"라고 명시하라
4. 웹 검색 결과 기반 정보는 [웹검색] 태그를, DART 데이터 기반 정보는 [DART] 태그를 붙여라
5. 추정이나 가정을 할 경우 "추정" 또는 "가정"이라고 명시하라

## PE 관점 응답 프레임워크
답변 시 다음 관점을 자동 반영:
- **딜 소싱**: 인수 타당성, 산업 매력도
- **밸류에이션**: EV/EBITDA, 순차입금, NWC 조정
- **리스크**: 규제, 소송, 지배구조, 키맨, 고객 집중도
- **Exit**: 잠재 바이어, IPO 가능성, 멀티플 확장 여지

## 대화 규칙 (★최우선★)
- 사용자가 인사(안녕, 안녕하세요, 하이, hi 등)를 하면 **짧고 자연스럽게 인사로 응답**하라
  - 예: "안녕하세요! 노랑풍선에 대해 궁금한 점이 있으시면 편하게 물어보세요."
  - 인사에 재무분석을 시작하지 마라
- 사용자가 간단한 질문(예: "매출이 얼마야?")을 하면 **짧게 핵심만 답변**하라
- 사용자가 심화 분석을 요청할 때만 상세한 PE 관점 분석을 제공하라
- **사용자 질문의 범위와 깊이에 맞춰 응답 길이를 조절하라** — 질문이 짧으면 답변도 짧게

## 응답 형식
- 마크다운 형식 사용
- 숫자는 천 단위 콤마 표기 (예: 1,234)
- **재무 수치 비교는 반드시 마크다운 테이블로 작성** (연도별 추이, 지표 비교, 계산 과정 등)
- 핵심 판단은 불릿 포인트로, 정량 데이터는 테이블로 분리
- 번호 목록(1. 2. 3.)은 각 항목을 별도 줄에 작성
- 간결하되 핵심을 놓치지 말 것
- 시각적으로 읽기 쉽게 섹션을 구분하고 구조화하라
- 분석 응답 시 불필요한 서론을 넣지 말고 바로 본문을 시작하라"""

    def _build_company_context(self) -> str:
        """기업개황 정보 텍스트 생성"""
        info = self.company_info
        if not info:
            return "기업 정보 없음"

        lines = [f"- 회사명: {info.get('corp_name', 'N/A')}"]
        if info.get('corp_name_eng'):
            lines.append(f"- 영문명: {info['corp_name_eng']}")
        if info.get('ceo_nm'):
            lines.append(f"- 대표자: {info['ceo_nm']}")
        if info.get('market_name'):
            lines.append(f"- 시장구분: {info['market_name']}")
        if info.get('stock_code'):
            lines.append(f"- 종목코드: {info['stock_code']}")
        if info.get('induty_code'):
            lines.append(f"- 업종코드: {info['induty_code']}")
        if info.get('induty_name'):
            lines.append(f"- 업종명: {info['induty_name']}")
        if info.get('est_dt_formatted'):
            lines.append(f"- 설립일: {info['est_dt_formatted']}")
        if info.get('acc_mt_formatted'):
            lines.append(f"- 결산월: {info['acc_mt_formatted']}")
        if info.get('adres'):
            lines.append(f"- 주소: {info['adres']}")
        if info.get('hm_url'):
            lines.append(f"- 홈페이지: {info['hm_url']}")

        return '\n'.join(lines)

    def _build_financial_context(self) -> str:
        """preview_data → 구조화된 재무 참조 테이블 생성 (엑셀에 포함되는 모든 데이터)"""
        sections = []

        # VCM 데이터 (핵심 요약)
        vcm_display = self.preview_data.get('vcm_display')
        if vcm_display and isinstance(vcm_display, list) and len(vcm_display) > 0:
            sections.append(self._format_table_data("[VCM] Financials (단위: 백만원)", vcm_display))

        # VCM 메타데이터 (세부 하위항목 — 툴팁 데이터)
        vcm_meta = self.preview_data.get('vcm')
        if vcm_meta and isinstance(vcm_meta, list) and len(vcm_meta) > 0:
            sections.append(self._format_table_data("[VCM-Frontdata] 세부항목 메타데이터 (원 단위)", vcm_meta))

        # IS 원본
        is_data = self.preview_data.get('is')
        if is_data and isinstance(is_data, list) and len(is_data) > 0:
            sections.append(self._format_table_data("[IS원본] 손익계산서", is_data))

        # BS 원본
        bs_data = self.preview_data.get('bs')
        if bs_data and isinstance(bs_data, list) and len(bs_data) > 0:
            sections.append(self._format_table_data("[BS원본] 재무상태표", bs_data))

        # CIS 포괄손익계산서
        cis_data = self.preview_data.get('cis')
        if cis_data and isinstance(cis_data, list) and len(cis_data) > 0:
            sections.append(self._format_table_data("[CIS원본] 포괄손익계산서", cis_data))

        # CF 원본
        cf_data = self.preview_data.get('cf')
        if cf_data and isinstance(cf_data, list) and len(cf_data) > 0:
            sections.append(self._format_table_data("[CF원본] 현금흐름표", cf_data))

        # 주석 데이터 (DART 공시 주석 테이블들)
        notes = self.preview_data.get('notes')
        if notes and isinstance(notes, list):
            for note in notes:
                title = note.get('title', '주석')
                note_type = note.get('type', '')
                source = note.get('source', '')
                note_data = note.get('data', [])
                if note_data and isinstance(note_data, list) and len(note_data) > 0:
                    label = f"[주석-{note_type}] {title}"
                    if source:
                        label += f" ({source})"
                    sections.append(self._format_table_data(label, note_data))

        if not sections:
            return "재무 데이터 없음"

        return '\n\n'.join(sections)

    def _format_table_data(self, title: str, data: list, max_rows: int = 0) -> str:
        """리스트 데이터를 텍스트 테이블로 변환"""
        if not data:
            return ""

        # 헤더 추출
        headers = list(data[0].keys())
        if not headers:
            return ""

        lines = [f"### {title}"]

        # 헤더 행
        lines.append('| ' + ' | '.join(str(h) for h in headers) + ' |')
        lines.append('| ' + ' | '.join('---' for _ in headers) + ' |')

        # 데이터 행 (max_rows=0이면 전체 출력)
        rows_to_render = data if max_rows <= 0 else data[:max_rows]
        for row in rows_to_render:
            vals = []
            for h in headers:
                v = row.get(h, '')
                if v is None:
                    v = ''
                vals.append(str(v))
            lines.append('| ' + ' | '.join(vals) + ' |')

        return '\n'.join(lines)

    def _build_analysis_context(self) -> str:
        """AI 분석 보고서 + 리서치 결과 컨텍스트"""
        parts = []

        if self.analysis_result:
            report = self.analysis_result if isinstance(self.analysis_result, str) else self.analysis_result.get('report', '')
            if report:
                # 보고서가 너무 길면 잘라내기
                if len(report) > 8000:
                    report = report[:8000] + "\n\n(... 보고서 일부 생략)"
                parts.append(f"## AI 재무 분석 보고서\n{report}")

        if self.research_result:
            research = self.research_result
            if isinstance(research, dict):
                report_text = research.get('report', '')
                if report_text:
                    if len(report_text) > 6000:
                        report_text = report_text[:6000] + "\n\n(... 리서치 일부 생략)"
                    parts.append(f"## 기업 리서치 보고서\n{report_text}")

                # 경쟁사 정보
                competitors = research.get('competitors_domestic', [])
                if competitors:
                    comp_lines = ["## 국내 경쟁사"]
                    for c in competitors[:5]:
                        comp_lines.append(f"- **{c.get('name', '')}**: {c.get('business', '')}")
                    parts.append('\n'.join(comp_lines))

        if not parts:
            return ""

        return '\n\n'.join(parts)

    def classify_question(self, user_message: str) -> str:
        """질문 유형 분류: default / search (키워드 기반, LLM 호출 없음)

        - search: 웹 검색이 명시적으로 필요한 경우만
        - default: 그 외 모든 질문 (재무 분석, 지표, PE 평가 등) → Flash 단일 에이전트
        """
        msg = user_message.lower().strip()
        search_keywords = [
            '검색', '찾아', '찾아줘', '서치', 'search',
            '최근 뉴스', '뉴스', '기사',
            '시장 동향', '시장동향', '업계 동향', '업계동향',
            '경쟁사 현황', '경쟁사현황', '경쟁업체',
            'M&A 동향', 'M&A동향', 'm&a 동향', 'm&a동향',
            '최신 정보', '최신정보', '최근 이슈', '최근이슈',
            '인터넷', '웹에서', '온라인',
        ]
        for kw in search_keywords:
            if kw.lower() in msg:
                return 'search'
        return 'default'

    async def _generate_search_queries(self, user_message: str) -> list:
        """PE 맞춤 검색어 생성"""
        company = sanitize_for_prompt(self.company_name, max_length=50)
        prompt = f"""사용자가 "{sanitize_for_prompt(user_message)}" 라고 질문했습니다.
대상 기업: {company}

PE(Private Equity) 관점에서 이 질문에 답하기 위한 웹 검색어를 5-7개 생성하세요.
다각도 검색이 필요합니다: M&A, 산업동향, 실적, 규제, 지배구조, 소송 등.

형식: 각 줄에 검색어 하나씩. 다른 텍스트 없이 검색어만 출력.
"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: generate_with_retry(MODEL_FLASH, prompt, step_name="검색어생성")
            )
            queries = [line.strip().strip('-').strip('•').strip() for line in result.text.strip().split('\n') if line.strip()]
            return queries[:7]
        except Exception:
            return [f"{company} 최근 뉴스", f"{company} M&A", f"{company} 실적"]

    async def _execute_search(self, queries: list) -> tuple:
        """Firecrawl 병렬 검색 실행. (검색텍스트, 참조URL목록) 반환"""
        from urllib.parse import urlparse

        async def search_one(query):
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda q=query: firecrawl_client.search(q, limit=5)
                )
                # Firecrawl v2: SearchData has .web attribute (list of SearchResultWeb)
                items = None
                if hasattr(result, 'web') and result.web:
                    items = result.web
                elif hasattr(result, 'data') and result.data:
                    items = result.data
                elif isinstance(result, list):
                    items = result

                if not items:
                    return []
                return items
            except Exception as e:
                print(f"[PE챗봇] 검색 실패 ({query[:30]}): {e}")
                return []

        # 병렬 검색
        tasks = [search_one(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 필터링 및 통합
        all_items = []
        for r in results:
            if isinstance(r, list):
                all_items.extend(r)
            elif isinstance(r, Exception):
                print(f"[PE챗봇] 검색 예외: {type(r).__name__}: {r}")

        filtered = self._filter_search_results(all_items)
        print(f"[PE챗봇] 검색: {len(queries)}쿼리 → {len(all_items)}건 → 필터링 {len(filtered)}건")

        if not filtered:
            return "검색 결과가 없습니다.", []

        # 상위 결과 크롤링
        urls = [item['url'] for item in filtered[:8]]
        scraped = await self._scrape_urls(urls)

        # 참조 URL 목록 생성
        references = []
        for s in scraped:
            references.append({'url': s['url'], 'title': s['title']})

        scraped_urls = {s['url'] for s in scraped}
        for item in filtered[:12]:
            if item['url'] not in scraped_urls:
                references.append({'url': item['url'], 'title': item['title']})

        # 텍스트 생성
        search_text_parts = []
        for s in scraped:
            search_text_parts.append(f"### {s['title']}\nURL: {s['url']}\n{s['content']}\n")

        for item in filtered[:12]:
            if item['url'] not in scraped_urls:
                search_text_parts.append(f"### {item['title']}\nURL: {item['url']}\n{item.get('desc', '')}\n")

        return '\n\n'.join(search_text_parts[:10]), references

    def _filter_search_results(self, items: list) -> list:
        """검색 결과 필터링 (중복 제거 + 도메인 필터링)"""
        from urllib.parse import urlparse

        seen_urls = set()
        seen_titles = set()
        filtered = []

        for item in items:
            url = getattr(item, 'url', '') or (item.get('url', '') if isinstance(item, dict) else '')
            title = getattr(item, 'title', None) or (item.get('title', '') if isinstance(item, dict) else '')
            if not title and hasattr(item, 'metadata') and item.metadata:
                title = getattr(item.metadata, 'title', '')
            desc = getattr(item, 'description', None) or getattr(item, 'snippet', None) or ''
            if not desc and isinstance(item, dict):
                desc = item.get('description', '') or item.get('snippet', '')

            if not url or not title:
                continue

            if url in seen_urls:
                continue
            seen_urls.add(url)

            title_key = title.strip()[:60]
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            try:
                domain = urlparse(url).netloc.lower()
                if any(domain == junk or domain.endswith('.' + junk) for junk in _JUNK_DOMAINS):
                    continue
            except Exception:
                pass

            filtered.append({'url': url, 'title': title or '', 'desc': desc or ''})

        return filtered

    async def _scrape_urls(self, urls: list, max_chars: int = 3000) -> list:
        """URL 병렬 크롤링"""
        def _scrape_one(url):
            try:
                doc = firecrawl_client.scrape(
                    url, formats=["markdown"], only_main_content=True, timeout=15000
                )
                title = getattr(doc.metadata, 'title', '') if doc.metadata else ''
                content = (doc.markdown or '')[:max_chars]
                if not content.strip():
                    return None
                return {'url': url, 'title': title or '', 'content': content}
            except Exception as e:
                print(f"[PE챗봇] 크롤링 실패: {url[:50]}... ({e})")
                return None

        unique_urls = list(dict.fromkeys(urls))
        if not unique_urls:
            return []

        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, _scrape_one, url) for url in unique_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if r and not isinstance(r, Exception)]

    async def _run_default_agent(self, user_message: str) -> AsyncGenerator[str, None]:
        """기본 응답: Flash 단일 에이전트 (재무 분석, PE 평가, 지표 계산 모두 처리)"""
        messages = self._build_messages(user_message)

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: generate_with_retry(
                    MODEL_FLASH, messages,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=8192,
                    ),
                    step_name="응답"
                )
            )
            text = response.text or ""
            for chunk in self._simulate_streaming(text):
                yield chunk
        except Exception as e:
            yield f"\n\n오류가 발생했습니다: {str(e)}"

    async def _run_complex_agent(self, user_message: str) -> AsyncGenerator[str, None]:
        """복잡한 질문: 3-에이전트 병렬 호출"""

        # Phase 1: Financial Analyst + Market Researcher 병렬
        yield {"type": "status", "content": "재무 분석 및 시장 조사 중..."}

        async def financial_analysis():
            prompt = f"""{self.system_prompt}

다음은 사용자의 대화 기록과 질문입니다.
{self._format_conversation_for_prompt()}
사용자: {sanitize_for_prompt(user_message)}

당신은 Financial Analyst 에이전트입니다.
위 재무 데이터를 기반으로 정량적 분석을 수행하세요.
- 핵심 재무지표 계산 (EV/EBITDA, 순차입금, NWC 등)
- 연도별 추이 분석
- 이상 항목 식별
- 수치 근거 반드시 명시

분석 결과만 출력하세요."""

            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: generate_with_retry(MODEL_PRO, prompt, step_name="재무분석")
                )
                return result.text or ""
            except Exception as e:
                return f"재무 분석 실패: {e}"

        async def market_research():
            queries = await self._generate_search_queries(user_message)
            search_text, refs = await self._execute_search(queries)

            prompt = f"""당신은 Market Researcher 에이전트입니다.
대상 기업: {sanitize_for_prompt(self.company_name, 50)}

사용자 질문: {sanitize_for_prompt(user_message)}

아래 웹 검색 결과를 기반으로 시장 및 산업 정보를 정리하세요.
- 산업 동향, 경쟁 현황
- 기업 관련 뉴스 및 이슈
- M&A 시장 동향
- 규제 환경

검색 결과:
{search_text[:6000]}

분석 결과만 출력하세요. 출처 URL을 포함하세요."""

            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: generate_with_retry(MODEL_PRO, prompt, step_name="시장조사")
                )
                return result.text or "", refs
            except Exception as e:
                return f"시장 조사 실패: {e}", refs

        # 병렬 실행
        financial_task = asyncio.create_task(financial_analysis())
        market_task = asyncio.create_task(market_research())

        financial_result = await financial_task
        market_result_tuple = await market_task
        market_result = market_result_tuple[0] if isinstance(market_result_tuple, tuple) else market_result_tuple
        search_refs = market_result_tuple[1] if isinstance(market_result_tuple, tuple) and len(market_result_tuple) > 1 else []

        # Phase 2: PE Advisor (종합)
        yield {"type": "status", "content": "PE 관점 종합 분석 중..."}

        advisor_prompt = f"""{self.system_prompt}

사용자의 대화 기록:
{self._format_conversation_for_prompt()}
사용자: {sanitize_for_prompt(user_message)}

## Financial Analyst 분석 결과
{financial_result[:4000]}

## Market Researcher 조사 결과
{market_result[:4000]}

당신은 PE Advisor입니다. 위 두 에이전트의 결과를 종합하여 PE 관점의 최종 답변을 작성하세요.

답변 구성:
1. 핵심 요약 (2-3문장)
2. 재무 분석 (DART 데이터 기반, [DART] 태그) — 수치는 반드시 마크다운 테이블로 정리
3. 시장/산업 분석 (웹 검색 기반, [웹검색] 태그)
4. PE 관점 평가 (투자 매력도, 리스크, 밸류에이션)
5. 추가 확인 필요 사항 — 각 항목은 별도 줄에 번호 매겨서 작성

마크다운 형식으로 작성하세요. 재무 수치 비교/추이는 반드시 마크다운 테이블(|항목|FY2022|FY2023|FY2024|)을 사용하세요."""

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: generate_with_retry(
                    MODEL_PRO, advisor_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=8192,
                    ),
                    step_name="PE어드바이저"
                )
            )
            text = result.text or ""
            for chunk in self._simulate_streaming(text):
                yield chunk
            # 참조 URL 전달
            if search_refs:
                yield {"type": "references", "content": search_refs}
        except Exception as e:
            print(f"[PE챗봇] complex agent 오류: {e}")
            import traceback
            traceback.print_exc()
            yield f"\n\n종합 분석 중 오류 발생: {str(e)}"

    async def _run_search_agent(self, user_message: str) -> AsyncGenerator[str, None]:
        """검색 중심 질문: 검색 후 Flash 응답"""

        yield {"type": "search_start"}

        queries = await self._generate_search_queries(user_message)
        search_text, search_refs = await self._execute_search(queries)

        yield {"type": "search_done"}

        # 검색 결과 기반 답변 생성
        messages = self._build_messages(user_message, extra_context=f"""
## 웹 검색 결과
아래 검색 결과를 참고하여 답변하되, DART 데이터와 구분하여 출처를 명시하세요.
{search_text[:6000]}
""")

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: generate_with_retry(
                    MODEL_PRO, messages,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=6144,
                    ),
                    step_name="검색응답"
                )
            )
            text = response.text or ""
            for chunk in self._simulate_streaming(text):
                yield chunk
            # 참조 URL 전달
            if search_refs:
                yield {"type": "references", "content": search_refs}
        except Exception as e:
            yield f"\n\n검색 기반 답변 생성 중 오류: {str(e)}"

    def _build_messages(self, user_message: str, extra_context: str = "") -> str:
        """대화 히스토리 포함 메시지 빌드"""
        parts = [self.system_prompt]

        if extra_context:
            parts.append(extra_context)

        # 대화 히스토리
        if self.conversation_history:
            parts.append("\n## 이전 대화")
            for msg in self.conversation_history[-10:]:  # 최근 10턴
                role = "사용자" if msg['role'] == 'user' else "어시스턴트"
                content = msg['content'][:1000]  # 각 메시지 1000자 제한
                parts.append(f"{role}: {content}")

        parts.append(f"\n사용자: {sanitize_for_prompt(user_message)}")
        parts.append("\n사용자의 질문 의도에 맞게 답변하세요. 재무 데이터가 필요한 질문에만 데이터를 활용하고, 일상적 대화에는 자연스럽게 응답하세요.")

        return '\n\n'.join(parts)

    def _format_conversation_for_prompt(self) -> str:
        """대화 히스토리를 프롬프트용 텍스트로 변환"""
        if not self.conversation_history:
            return "(이전 대화 없음)"

        lines = []
        for msg in self.conversation_history[-6:]:
            role = "사용자" if msg['role'] == 'user' else "어시스턴트"
            content = msg['content'][:500]
            lines.append(f"{role}: {content}")
        return '\n'.join(lines)

    def _simulate_streaming(self, text: str, chunk_size: int = 15):
        """텍스트를 청크로 분할 (스트리밍 시뮬레이션)"""
        # 단어 단위로 분할하여 자연스러운 스트리밍
        words = text.split(' ')
        buffer = []
        for word in words:
            buffer.append(word)
            if len(buffer) >= chunk_size:
                yield ' '.join(buffer) + ' '
                buffer = []
        if buffer:
            yield ' '.join(buffer)

    async def chat(self, user_message: str) -> AsyncGenerator[dict, None]:
        """메인 챗 함수 — SSE 이벤트 생성"""

        # 대화 내역 관리
        if len(self.conversation_history) >= MAX_CONVERSATION_TURNS * 2:
            # 오래된 대화 제거 (최근 20턴만 유지)
            self.conversation_history = self.conversation_history[-40:]

        # 사용자 메시지 저장
        self.conversation_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })

        # 질문 분류 (키워드 기반, LLM 호출 없음 — 즉시)
        question_type = self.classify_question(user_message)
        print(f"[PE챗봇] 질문 유형: {question_type} | 질문: {user_message[:50]}...")

        yield {"type": "classify", "content": question_type}

        # 에이전트 선택 및 실행
        full_response = []

        if question_type == 'search':
            async for chunk in self._run_search_agent(user_message):
                if isinstance(chunk, dict):
                    yield chunk
                else:
                    full_response.append(chunk)
                    yield {"type": "token", "content": chunk}
        else:  # default: Flash 단일 에이전트 (재무 분석, PE 평가 모두 처리)
            async for chunk in self._run_default_agent(user_message):
                if isinstance(chunk, dict):
                    yield chunk
                else:
                    full_response.append(chunk)
                    yield {"type": "token", "content": chunk}

        # 어시스턴트 응답 저장
        assistant_response = ''.join(full_response)
        self.conversation_history.append({
            'role': 'assistant',
            'content': assistant_response,
            'timestamp': datetime.now().isoformat()
        })

        yield {"type": "done"}

    def get_suggestions(self) -> list:
        """PE 관점 추천 질문을 랜덤으로 선별"""
        import random

        # 기본 PE 분석 질문 풀 (매번 랜덤 4개 선택)
        base_pool = [
            # 밸류에이션
            "적정 EV/EBITDA 멀티플은 얼마로 보는 게 합리적일까?",
            "현재 밸류에이션 수준에서 인수 가격 협상 여지는?",
            "DCF 기준 적정 기업가치 레인지를 추정해줘",
            "PER, PBR 기준으로 동종업계 대비 할인/할증 수준은?",
            # 수익성·성장성
            "매출 성장률과 영업이익률 추이에서 보이는 패턴은?",
            "EBITDA 마진 개선 여지가 있을까?",
            "매출원가율과 판관비율 변동의 주요 원인은?",
            "최근 3개년 수익성 트렌드가 인수 후 유지 가능한 수준인가?",
            # 재무 안정성
            "순차입금 대비 EBITDA 배수는 적정한가?",
            "NWC 변동 추이에서 운전자본 관리 효율성은?",
            "유동비율과 부채비율로 본 재무 건전성은?",
            "차입금 만기 구조와 리파이낸싱 리스크는?",
            # 딜 구조·리스크
            "인수 시 주요 리스크 요인 Top 5를 정리해줘",
            "LBO 구조 적용 시 레버리지 적정 수준은?",
            "이 기업의 핵심 투자 매력 포인트는 뭐야?",
            "고객 집중도나 키맨 리스크가 있는지 분석해줘",
            "매출채권 회전율과 재고자산 회전율 추이는?",
            # Exit 전략
            "예상 가능한 Exit 시나리오와 멀티플 확장 가능성은?",
            "전략적 바이어 관점에서 이 기업의 시너지 포인트는?",
            "IPO Exit 가능성과 필요 조건은?",
            # 실사 관점
            "일회성 비용/수익 항목이 있다면 정상화 조정 포인트는?",
            "영업외손익에서 주의 깊게 봐야 할 항목은?",
            "자본적 지출(CAPEX) 추이와 유지보수 투자 수준은?",
            "잉여현금흐름(FCF) 창출력은 어떤 수준인가?",
        ]

        selected = random.sample(base_pool, min(4, len(base_pool)))

        # AI 분석 완료 시 추가 질문 풀에서 1개
        if self.analysis_result:
            analysis_pool = [
                "AI 분석에서 발견된 가장 큰 이상 징후는?",
                "재무 분석 보고서의 핵심 인사이트 요약해줘",
                "AI가 감지한 전년 대비 급변 항목들을 정리해줘",
                "분석 결과에서 딜브레이커가 될 만한 요소가 있어?",
            ]
            selected.append(random.choice(analysis_pool))

        # 리서치 완료 시 추가 질문 풀에서 1개
        if self.research_result:
            research_pool = [
                "동종업계 경쟁사 대비 이 기업의 포지셔닝은?",
                "최근 업계 M&A 동향과 이 기업의 위치는?",
                "리서치 결과에서 드러난 시장 리스크는?",
                "산업 내 규제 환경 변화가 미칠 영향은?",
            ]
            selected.append(random.choice(research_pool))

        return selected[:6]

    def get_history(self) -> list:
        """대화 내역 반환"""
        return self.conversation_history

    def update_context(self, analysis_result=None, research_result=None):
        """AI 분석 / 리서치 결과 업데이트"""
        if analysis_result is not None:
            self.analysis_result = analysis_result
        if research_result is not None:
            self.research_result = research_result
        # 시스템 프롬프트 재생성
        self.system_prompt = self._build_system_prompt()
