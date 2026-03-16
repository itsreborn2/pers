"""
LLM 기반 계정과목 분류 모듈

DART 재무제표의 계정과목명을 LLM(Gemini)으로 분류하여
표준 카테고리 매핑 테이블을 생성합니다.

핵심 원칙:
1. LLM은 매핑만 수행 — 숫자는 절대 전달하지 않음
2. BS + IS + CF 전체 계정을 LLM에 거침
3. 하드코딩 규칙을 프롬프트 지시사항으로 변환
4. 캐시 필수 — 동일 회사 재분석시 LLM 비용 0원
"""

import os
import re
import json
import time
import hashlib
import sqlite3
from typing import Optional
from contextlib import contextmanager
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

# ============================================================
# Gemini 클라이언트 초기화
# ============================================================
API_TIMEOUT_MS = 300_000  # 300초 (5분)
_client = genai.Client(
    api_key=os.getenv('GEMINI_API_KEY'),
    http_options=types.HttpOptions(timeout=API_TIMEOUT_MS)
)

MODEL = "gemini-3.1-pro-preview"

# 재시도 설정
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 10


def generate_with_retry(client_obj, model: str, contents, config=None,
                        max_retries=MAX_RETRIES, step_name: str = ""):
    """429/503/타임아웃 에러 시 자동 재시도 래퍼 (financial_insight_analyzer.py에서 복제)"""
    last_error = None
    label = f"[{step_name}] " if step_name else ""

    for attempt in range(max_retries):
        try:
            if config:
                return client_obj.models.generate_content(
                    model=model, contents=contents, config=config
                )
            else:
                return client_obj.models.generate_content(
                    model=model, contents=contents
                )
        except Exception as e:
            error_str = str(e)
            last_error = e

            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                retry_match = re.search(r'retry.?in.?(\d+(?:\.\d+)?)', error_str.lower())
                wait_time = float(retry_match.group(1)) + 1 if retry_match else INITIAL_RETRY_DELAY * (2 ** attempt)
                print(f"{label}[Rate Limit] 429 에러. {wait_time:.1f}초 후 재시도 ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            elif '503' in error_str or 'UNAVAILABLE' in error_str:
                wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                print(f"{label}[503] {wait_time:.1f}초 후 재시도 ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            elif 'timeout' in error_str.lower() or 'ReadTimeout' in error_str or 'ConnectTimeout' in error_str:
                wait_time = 10 * (attempt + 1)
                print(f"{label}[Timeout] {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e

    raise last_error


# ============================================================
# 계정명 전처리
# ============================================================
def _normalize_account_name(name: str) -> str:
    """계정과목명 정규화 (분류 비교용)"""
    if not name:
        return ''
    s = re.sub(r'\s', '', str(name))
    # 로마숫자 접두사 제거
    s = re.sub(r'^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.', '', s)
    s = re.sub(r'^[IVX]+\.', '', s)
    # 주석 참조 제거
    s = re.sub(r'\(주[석\d,\s]*\)', '', s)
    s = re.sub(r'\[주석[\d,\s]*\]', '', s)
    return s.strip()


# ============================================================
# 프롬프트 템플릿
# ============================================================

BS_CLASSIFICATION_PROMPT = """당신은 한국 재무제표(K-IFRS/K-GAAP) 전문 분류기입니다.
아래 재무상태표(BS) 계정과목명 리스트를 분류하세요.

## 분류 규칙 (BS)

### 유동자산 (current_asset)
- 현금및현금성자산, 단기금융상품, 당기손익-공정가치측정금융자산
- 매출채권, 미수금, 미수수익, 선급금, 선급비용, 계약자산
- 재고자산, 상품, 제품, 원재료, 재공품, 저장품
- 기타유동자산, 기타유동금융자산

### 비유동자산 (non_current_asset)
- 유형자산 (토지, 건물, 기계장치 등)
- 무형자산 (영업권, 산업재산권, 개발비, 소프트웨어 등)
- 사용권자산 (리스)
- 투자자산: 장기금융상품, 매도가능금융자산, 지분법투자, 관계기업투자, 종속기업투자
- 보증금, 임차보증금
- 이연법인세자산

### 유동부채 (current_liability)
- 매입채무, 미지급금, 미지급비용, 선수금, 선수수익, 예수금, 예수보증금
- 계약부채, 연차충당부채
- 단기차입금, 유동성장기부채, 유동성사채
- 전환사채(유동), 상환전환우선주부채, 전환우선주부채
- 리스부채, 파생상품부채, 기타금융부채
- 미지급법인세, 당기법인세부채, 충당부채

### 비유동부채 (non_current_liability)
- 장기매입채무, 장기미지급금, 장기선수금, 장기예수금
- 임대보증금, 예수보증금(비유동)
- 사채 (유동성 제외), 장기차입금
- 퇴직급여충당부채, 퇴직급여채무, 확정급여채무, 순확정급여부채
- 장기충당부채, 이연법인세부채
- 리스부채(비유동), 기타금융부채(비유동)

### 자본 (equity)
- 자본금, 이익잉여금, 미처분이익잉여금, 결손금
- 자본잉여금, 자본조정, 기타포괄손익누계액, 기타자본항목
- 비지배지분

### 특수 분류
- section_header: "자산", "부채", "자본" 같은 대분류 헤더
- subtotal: "유동자산 합계", "비유동자산 소계" 등
- total: "자산총계", "부채총계", "자본총계", "부채와자본총계"
- skip: 빈 행, 의미 없는 구분선

## 그룹핑 규칙 (BS)

### 유동자산 그룹
- "현금및현금성자산" → group: null (단독 표시)
- "단기금융상품" + "당기손익-공정가치측정금융자산" + "기타포괄손익-공정가치측정금융자산" → group: "단기투자자산"
- "매출채권" + "미수금" + "미수수익" + "선급금" + "선급비용" + "계약자산" + "기타금융자산(유동)" → group: "매출채권및기타채권"
- "재고자산" → group: null (단독 표시)

### 비유동자산 그룹
- "유형자산" → group: null (단독 표시, "무형자산", "사용권자산" 제외!)
- "무형자산" → group: null (단독 표시)
- "사용권자산" → group: null (단독 표시)
- "장기금융상품" + "매도가능금융자산" + "관계기업투자" + "종속기업투자" + "지분법적용투자" → group: "장기투자자산"
- "보증금", "임차보증금" → group: null

### 유동부채 그룹
- "매입채무" + "미지급금" + "미지급비용" + "선수금" + "선수수익" + "예수금" + "예수보증금" + "연차충당부채" + "계약부채" → group: "매입채무및기타채무"
- "단기차입금" + "유동성장기부채" + "유동성사채" + "전환사채" + "상환전환우선주부채" + "전환우선주부채" → group: "유동차입부채"
- "리스부채" + "파생상품부채" + "기타금융부채" → group: "기타금융부채"

### 비유동부채 그룹
- "장기매입채무" + "장기미지급금" + "장기미지급비용" + "장기선수금" + "장기선수수익" + "장기예수금" + "임대보증금" + "예수보증금" + "계약부채(비유동)" → group: "매입채무및기타채무[비유동]"
- "사채"(유동성 제외) + "장기차입금" → group: "비유동차입부채"
- "퇴직급여충당부채", "퇴직급여채무", "확정급여채무" → group: "퇴직급여채무"

### 자본 그룹
- "자본금" → group: null (단독)
- "이익잉여금", "미처분이익잉여금", "결손금" → group: null (단독, display_name: "이익잉여금")
- "자본잉여금" + "자본조정" + "기타포괄손익누계액" + "기타자본항목" → group: "기타자본구성요소"

## 이중계산 방지
- "유동자산" 집계항목이 있으면 하위 개별 항목들의 합이 이를 초과하지 않아야 합니다
- 소계/합계 행은 standard_category를 "subtotal" 또는 "total"로 분류하세요

## 부호 규칙
- 대부분의 자산 항목: sign "+"
- 대부분의 부채/자본 항목: sign "+"
- "결손금" → sign "-" (이익잉여금이 음수)
- 개별 비용/수익 항목 (other_expense, other_income 등): sign "+" (이미 비용/수익 카테고리에 분류되므로 부호 변환 불필요)
  - 예: "유형자산처분손실", "외화환산손실", "대손상각비" → sign "+" (비용 항목)
  - 예: "유형자산처분이익", "외화환산이익" → sign "+" (수익 항목)
- sign "-"는 집계/합계 행에서만 사용 (이름이 반대 의미를 나타내는 경우):
  - "영업손실(이익)" = 영업이익의 반대 → sign "-"
  - "당기순손실" = 당기순이익의 반대 → sign "-"
  - "법인세비용차감전순손실" → sign "-"
  - "매출총손실" → sign "-"

## 주의사항
- "(주1)", "[주석2,3]" 등 주석 참조는 무시하고 분류
- "Ⅴ.유형자산" → "유형자산"으로 처리 (로마숫자 접두사 무시)
- 확신이 낮으면 confidence를 낮게 설정 (0.8 미만이면 standard_category: null)
- display_name은 사용자에게 표시할 깔끔한 이름 (주석 참조 제거, 로마숫자 제거)

## 업종 정보
{industry_info}

## 분류할 계정과목 리스트
{account_names}

위 계정과목명을 JSON 배열로 분류하여 반환하세요. 각 항목:
- raw_name: 원본 계정과목명 (그대로)
- standard_category: 표준 분류 (current_asset, non_current_asset, current_liability, non_current_liability, equity, section_header, subtotal, total, skip 중 하나. 확신 없으면 null)
- display_name: 표시용 이름 (주석/로마숫자 제거, 깔끔하게)
- group: 그룹핑 대상이면 그룹명 (예: "매출채권및기타채권"), 단독이면 null
- sign: "+" 또는 "-"
- confidence: 0.0~1.0
- reason: 분류 근거 (한국어, 간결하게)
"""

IS_CLASSIFICATION_PROMPT = """당신은 한국 재무제표(K-IFRS/K-GAAP) 전문 분류기입니다.
아래 손익계산서(IS) 계정과목명 리스트를 분류하세요.

## 분류 규칙 (IS)

### 매출 (revenue)
- 매출액, 영업수익, 수익(revenue)

### 매출원가 (cogs)
- 매출원가, 영업비용(서비스업), 제조원가

### 매출총이익 (gross_profit)
- 매출총이익, 총이익

### 판매비와관리비 (sga)
- 판매비와관리비 합계행 및 하위항목 전부:
  - 인건비: 급여, 직원급여, 종업원급여, 퇴직급여, 복리후생비, 주식보상비용
  - 감가상각비, 무형자산상각비
  - 지급수수료, 판매수수료
  - 광고선전비, 대손상각비, 경상연구개발비
  - 기타 판관비 항목들

### 영업이익 (operating_income)
- 영업이익, 영업손익
- "영업손실(이익)" → category: operating_income, sign: "-"
- "영업이익(손실)" → category: operating_income, sign: "+"

### 금융수익/비용
- interest_income: 이자수익, 금융수익, 배당금수익
- interest_expense: 이자비용, 금융원가, 금융비용 ← "원가"도 반드시 포함!

### 영업외수익/비용
- other_income: 기타수익, 기타영업외수익, 외환차익, 외화환산이익, 유형자산처분이익, 지분법이익
- other_expense: 기타비용, 기타영업외비용, 외환차손, 외화환산손실, 유형자산처분손실, 유형자산손상차손, 지분법손실

### 법인세 전/후
- ebt: 법인세비용차감전이익(손실), 법인세비용차감전순이익(손실)
- tax: 법인세비용, 법인세등
- net_income: 당기순이익, 당기순손실, 연결당기순이익

### ★ 계속영업/중단영업 구분 (매우 중요!) ★
- "계속영업이익", "계속영업손실", "계속영업이익(손실)" → subtotal (net_income이 아님!)
- "중단영업손익", "중단영업이익", "중단영업손실" → other_income (sign: 손실이면 "-")
- "당기순이익" = 계속영업이익 + 중단영업손익 → 이것만 net_income
- 만약 "계속영업이익"과 "당기순이익"이 함께 있으면, 반드시 "당기순이익"만 net_income으로 분류!
- "지배주주에 귀속되는 순이익/손실", "비지배지분 순이익" → subtotal (당기순이익의 하위 배분)

### 특수 분류
- section_header: "포괄손익계산서", "손익계산서" 등 제목
- subtotal: 중간 소계, 계속영업이익(손실)
- total: 총계
- skip: 빈 행, 구분선

## 그룹핑 규칙 (IS)

### 판관비 그룹
- "급여" + "퇴직급여" + "복리후생비" + "주식보상비용" → group: "인건비"
- "감가상각비" → group: null (단독, VCM EBITDA 계산에 필요)
- "무형자산상각비" → group: null (단독)
- 나머지 판관비 항목 → group: null (개별 표시, 상위 6개까지만 표시)

### 영업외 그룹
- "이자수익" + "배당금수익" + 기타 금융수익 하위 → group: "금융수익" (금융수익 합계행이 있으면)
- "이자비용" + 기타 금융비용 하위 → group: "금융비용"

## 이중계산 방지
- "금융수익" 집계항목이 있으면 하위 "이자수익", "배당금수익" 등은 별도 계상하지 마세요
  → 하위항목은 standard_category 유지하되, group: "금융수익_하위" (코드에서 합산 방지)
- "판매비와관리비" 합계가 있으면 개별 SGA 항목은 합계를 초과하지 않아야 합니다
  → 합계행: standard_category: "sga", group: null
  → 개별항목: standard_category: "sga", group: "sga_detail"

## 부호 규칙 (★매우 중요★)
- 매출, 매출총이익, 영업이익, 수익: sign "+"
- 매출원가, 판관비, 비용: sign "+" (비용은 양수로 표기, 코드에서 차감)

### "(이익)" 또는 "(손실)" 괄호 패턴 부호 결정 핵심 규칙:
- "XXX이익(손실)" 또는 "XXX이익" → sign "+" (이익이 기본)
- "XXX손실(이익)" 또는 "XXX손실" → sign "-" (손실이 기본)
- 핵심: 괄호 안이 아닌, **괄호 바깥(앞쪽)의 단어**가 부호를 결정!

### 구체적 예시:
- "당기순이익(손실)" → sign "+" (이익이 앞)
- "당기순손실(이익)" → sign "-" (손실이 앞)
- "당기순손실" → sign "-"
- "당기순이익" → sign "+"
- "영업이익(손실)" → sign "+"
- "영업손실(이익)" → sign "-"
- "법인세비용차감전순이익(손실)" → sign "+" (이익이 앞)
- "법인세비용차감전순손실" → sign "-"
- "계속영업이익(손실)" → sign "+"
- "중단영업순이익(손실)" → sign "+"
- "처분손실", "평가손실", "감액손실" → sign "-" (단독 손실 항목)

### 주의: 아래처럼 하면 안 됩니다!
- "당기순이익(손실)"에 "손실"이 포함되어 있다고 sign="-"로 하면 ❌ 틀림!
- 괄호 안의 "(손실)"은 대안 표기일 뿐, 부호를 결정하지 않음

## 주의사항
- "(주1)", "[주석2,3]" 등 주석 참조는 무시
- "Ⅰ.매출액" → "매출액" (로마숫자 접두사 무시)
- 확신 낮으면 confidence < 0.8, standard_category: null
- display_name은 깔끔한 표시용 이름

## 업종 정보
{industry_info}

## 분류할 계정과목 리스트
{account_names}

위 계정과목명을 JSON 배열로 분류하여 반환하세요. 각 항목:
- raw_name: 원본 계정과목명
- standard_category: revenue, cogs, gross_profit, sga, operating_income, interest_income, interest_expense, other_income, other_expense, ebt, tax, net_income, section_header, subtotal, total, skip 중 하나 (확신 없으면 null)
- display_name: 표시용 이름
- group: 그룹핑 대상이면 그룹명, 아니면 null
- sign: "+" 또는 "-"
- confidence: 0.0~1.0
- reason: 분류 근거
"""

CF_CLASSIFICATION_PROMPT = """당신은 한국 재무제표(K-IFRS/K-GAAP) 전문 분류기입니다.
아래 현금흐름표(CF) 계정과목명 리스트를 분류하세요.

## 분류 규칙 (CF)

### 영업활동 (operating_cf)
- 영업활동으로 인한 현금흐름 및 하위항목
- 당기순이익, 감가상각비, 무형자산상각비 (조정항목)
- 매출채권 증감, 재고자산 증감, 매입채무 증감 (운전자본 변동)

### 투자활동 (investing_cf)
- 투자활동으로 인한 현금흐름 및 하위항목
- 유형자산 취득/처분, 무형자산 취득/처분
- 금융자산 취득/처분, 투자 관련

### 재무활동 (financing_cf)
- 재무활동으로 인한 현금흐름 및 하위항목
- 차입금 증가/상환, 사채 발행/상환
- 배당금 지급, 자기주식 취득/처분

### 특수
- section_header: 대분류 헤더
- subtotal: 소계
- total: 기초현금, 기말현금, 현금증감 등
- skip: 빈 행

## 부호 규칙
- 유입: sign "+"
- 유출: sign "-"
- 계정명에 "감소", "지급", "취득", "상환" → sign "-"
- 계정명에 "증가", "수취", "처분", "발행" → sign "+"
- 합계행: sign "+" (순액이므로)

## 업종 정보
{industry_info}

## 분류할 계정과목 리스트
{account_names}

위 계정과목명을 JSON 배열로 분류하여 반환하세요. 각 항목:
- raw_name: 원본 계정과목명
- standard_category: operating_cf, investing_cf, financing_cf, section_header, subtotal, total, skip 중 하나
- display_name: 표시용 이름
- group: 그룹핑 대상이면 그룹명, 아니면 null
- sign: "+" 또는 "-"
- confidence: 0.0~1.0
- reason: 분류 근거
"""


# ============================================================
# DB 캐시 관련
# ============================================================
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'financial_data.db')


@contextmanager
def _get_db():
    """데이터베이스 연결 컨텍스트 매니저"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_classification_cache_table():
    """account_classification_cache 테이블 생성"""
    with _get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS account_classification_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_code TEXT NOT NULL,
                report_type TEXT NOT NULL CHECK(report_type IN ('BS', 'IS', 'CF')),
                account_name_raw TEXT NOT NULL,
                standard_category TEXT,
                display_name TEXT,
                group_name TEXT,
                sign_convention TEXT CHECK(sign_convention IN ('+', '-')),
                confidence REAL,
                reason TEXT,
                source TEXT NOT NULL CHECK(source IN ('rule', 'llm', 'manual')),
                model_version TEXT,
                prompt_hash TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(company_code, report_type, account_name_raw)
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_acc_cache_company
            ON account_classification_cache(company_code, report_type)
        ''')
        print("[AccountClassifier] 캐시 테이블 초기화 완료")


def _get_cached_classifications(company_code: str, report_type: str,
                                  account_names: list) -> dict:
    """
    DB 캐시에서 분류 결과 조회.
    Returns: {raw_name: {standard_category, display_name, group, sign, confidence, reason}}
    """
    if not account_names:
        return {}

    results = {}
    with _get_db() as conn:
        cursor = conn.cursor()
        # SQLite IN 쿼리는 파라미터 수 제한이 있으므로 배치 처리
        batch_size = 500
        for i in range(0, len(account_names), batch_size):
            batch = account_names[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch))
            cursor.execute(f'''
                SELECT account_name_raw, standard_category, display_name,
                       group_name, sign_convention, confidence, reason
                FROM account_classification_cache
                WHERE company_code = ? AND report_type = ?
                AND account_name_raw IN ({placeholders})
            ''', [company_code, report_type] + batch)

            for row in cursor.fetchall():
                results[row['account_name_raw']] = {
                    'raw_name': row['account_name_raw'],
                    'standard_category': row['standard_category'],
                    'display_name': row['display_name'],
                    'group': row['group_name'],
                    'sign': row['sign_convention'] or '+',
                    'confidence': row['confidence'] or 1.0,
                    'reason': row['reason'] or 'cached',
                }

    return results


def _save_classifications_to_cache(company_code: str, report_type: str,
                                     classifications: list, model_version: str,
                                     prompt_hash: str):
    """분류 결과를 DB 캐시에 저장"""
    if not classifications:
        return

    with _get_db() as conn:
        cursor = conn.cursor()
        for item in classifications:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO account_classification_cache
                    (company_code, report_type, account_name_raw, standard_category,
                     display_name, group_name, sign_convention, confidence, reason,
                     source, model_version, prompt_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'llm', ?, ?)
                ''', (
                    company_code,
                    report_type,
                    item.get('raw_name', ''),
                    item.get('standard_category'),
                    item.get('display_name', item.get('raw_name', '')),
                    item.get('group'),
                    item.get('sign', '+'),
                    item.get('confidence', 0.0),
                    item.get('reason', ''),
                    model_version,
                    prompt_hash,
                ))
            except Exception as e:
                print(f"[AccountClassifier] 캐시 저장 실패: {item.get('raw_name')}: {e}")


def _compute_prompt_hash(prompt_template: str, account_names: list) -> str:
    """프롬프트 + 계정명 리스트의 해시 계산 (캐시 무효화용)"""
    content = prompt_template[:200] + '|' + '|'.join(sorted(account_names))
    return hashlib.md5(content.encode()).hexdigest()[:16]


# ============================================================
# 핵심 분류 함수
# ============================================================

async def classify_accounts(account_names: list, statement_type: str,
                            company_code: str = 'unknown',
                            industry: str = None) -> list:
    """
    계정명 리스트를 받아 LLM으로 분류 매핑 테이블 반환.
    숫자는 절대 전달하지 않음 — 계정명만 전달.

    Args:
        account_names: 계정과목명 리스트 (예: ['현금및현금성자산', '매출채권', ...])
        statement_type: 'BS', 'IS', 'CF'
        company_code: 회사 코드 (캐시 키)
        industry: 업종 정보 (선택)

    Returns:
        list of dict: [
            {
                'raw_name': '현금및현금성자산',
                'standard_category': 'current_asset',
                'display_name': '현금및현금성자산',
                'group': null,
                'sign': '+',
                'confidence': 0.95,
                'reason': '유동자산 - 현금성 자산'
            },
            ...
        ]
    """
    if not account_names:
        return []

    # 빈 문자열/None 필터링
    clean_names = [str(n).strip() for n in account_names if n and str(n).strip()]
    if not clean_names:
        return []

    print(f"[AccountClassifier] {statement_type} 분류 시작: {len(clean_names)}개 계정")

    # 1. 캐시 조회
    cached = _get_cached_classifications(company_code, statement_type, clean_names)
    uncached_names = [n for n in clean_names if n not in cached]

    if not uncached_names:
        print(f"[AccountClassifier] {statement_type} 전체 캐시 히트: {len(cached)}개")
        return [cached[n] for n in clean_names if n in cached]

    print(f"[AccountClassifier] {statement_type} 캐시 히트: {len(cached)}개, LLM 호출 필요: {len(uncached_names)}개")

    # 2. 프롬프트 선택
    prompt_templates = {
        'BS': BS_CLASSIFICATION_PROMPT,
        'IS': IS_CLASSIFICATION_PROMPT,
        'CF': CF_CLASSIFICATION_PROMPT,
    }
    prompt_template = prompt_templates.get(statement_type, BS_CLASSIFICATION_PROMPT)

    industry_info = f"업종: {industry}" if industry else "업종 정보 없음 (일반적인 분류 규칙 적용)"

    # 3. 계정명 리스트를 프롬프트에 삽입
    account_list_str = '\n'.join(f'- {name}' for name in uncached_names)

    prompt = prompt_template.format(
        industry_info=industry_info,
        account_names=account_list_str
    )

    prompt_hash = _compute_prompt_hash(prompt_template, uncached_names)

    # 4. LLM 호출 (JSON 응답 강제)
    llm_results = _call_llm_classification(prompt, statement_type, uncached_names)

    # 5. 결과 검증 및 보정
    validated_results = _validate_classification_results(llm_results, uncached_names, statement_type)

    # 6. 캐시 저장
    _save_classifications_to_cache(company_code, statement_type, validated_results, MODEL, prompt_hash)

    # 7. 캐시 + LLM 결과 병합 (원래 순서 유지)
    all_results = {}
    all_results.update(cached)
    for item in validated_results:
        all_results[item['raw_name']] = item

    # 원래 순서 유지
    ordered_results = []
    for name in clean_names:
        if name in all_results:
            ordered_results.append(all_results[name])
        else:
            # 분류 실패한 항목 → 기본값
            ordered_results.append({
                'raw_name': name,
                'standard_category': None,
                'display_name': _normalize_account_name(name) or name,
                'group': None,
                'sign': '+',
                'confidence': 0.0,
                'reason': '분류 실패',
            })

    print(f"[AccountClassifier] {statement_type} 분류 완료: {len(ordered_results)}개")
    return ordered_results


def _call_llm_classification(prompt: str, statement_type: str,
                               account_names: list) -> list:
    """LLM 호출하여 분류 결과 JSON 파싱"""
    try:
        config = types.GenerateContentConfig(
            temperature=0.1,  # 낮은 온도로 일관성 확보
            response_mime_type="application/json",
        )

        response = generate_with_retry(
            client_obj=_client,
            model=MODEL,
            contents=prompt,
            config=config,
            max_retries=MAX_RETRIES,
            step_name=f"AccountClassifier-{statement_type}"
        )

        # 응답 파싱
        response_text = response.text.strip()

        # JSON 파싱
        try:
            results = json.loads(response_text)
        except json.JSONDecodeError:
            # JSON 블록 추출 시도
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                results = json.loads(json_match.group())
            else:
                print(f"[AccountClassifier] JSON 파싱 실패: {response_text[:200]}")
                return []

        if isinstance(results, dict) and 'classifications' in results:
            results = results['classifications']

        if not isinstance(results, list):
            print(f"[AccountClassifier] 예상 외 응답 형태: {type(results)}")
            return []

        print(f"[AccountClassifier] {statement_type} LLM 응답: {len(results)}개 항목")
        return results

    except Exception as e:
        print(f"[AccountClassifier] LLM 호출 실패: {e}")
        return []


def _validate_classification_results(llm_results: list, expected_names: list,
                                       statement_type: str) -> list:
    """LLM 결과 검증 및 보정"""
    # 유효 카테고리 정의
    valid_categories = {
        'BS': {'current_asset', 'non_current_asset', 'current_liability',
               'non_current_liability', 'equity', 'section_header',
               'subtotal', 'total', 'skip', None},
        'IS': {'revenue', 'cogs', 'gross_profit', 'sga', 'operating_income',
               'interest_income', 'interest_expense', 'other_income',
               'other_expense', 'ebt', 'tax', 'net_income',
               'section_header', 'subtotal', 'total', 'skip', None},
        'CF': {'operating_cf', 'investing_cf', 'financing_cf',
               'section_header', 'subtotal', 'total', 'skip', None},
    }

    valid_cats = valid_categories.get(statement_type, set())

    # LLM 결과를 raw_name 기준 딕셔너리로 변환
    llm_by_name = {}
    for item in llm_results:
        raw = item.get('raw_name', '')
        if raw:
            llm_by_name[raw] = item

    validated = []
    for name in expected_names:
        if name in llm_by_name:
            item = llm_by_name[name]
        else:
            # LLM이 누락한 항목 → normalized 이름으로 재시도
            norm_name = _normalize_account_name(name)
            found = False
            for llm_item in llm_results:
                if _normalize_account_name(llm_item.get('raw_name', '')) == norm_name:
                    item = llm_item.copy()
                    item['raw_name'] = name  # 원본 이름 복원
                    found = True
                    break
            if not found:
                item = {
                    'raw_name': name,
                    'standard_category': None,
                    'display_name': norm_name or name,
                    'group': None,
                    'sign': '+',
                    'confidence': 0.0,
                    'reason': 'LLM 응답에서 누락',
                }

        # 카테고리 유효성 검증
        category = item.get('standard_category')
        if category and category not in valid_cats:
            item['standard_category'] = None
            item['confidence'] = min(item.get('confidence', 0), 0.3)
            item['reason'] = f"무효 카테고리 '{category}' → null"

        # sign 유효성
        if item.get('sign') not in ('+', '-'):
            item['sign'] = '+'

        # confidence 범위
        conf = item.get('confidence', 0)
        if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
            item['confidence'] = 0.5

        # 필수 필드 보정
        if not item.get('display_name'):
            item['display_name'] = _normalize_account_name(name) or name
        if not item.get('raw_name'):
            item['raw_name'] = name

        validated.append(item)

    return validated


# ============================================================
# 유틸리티: 분류 결과를 VCM 구조로 변환
# ============================================================

def build_vcm_from_mapping(mapping: list, df, years: list,
                            statement_type: str) -> dict:
    """
    LLM 매핑 테이블 + 원본 DataFrame으로 VCM 구조 조립.
    LLM은 매핑만, 숫자 조립은 코드가 수행.

    Args:
        mapping: classify_accounts()의 반환값
        df: 원본 재무제표 DataFrame (label_ko, FYxxxx 컬럼)
        years: ['FY2024', 'FY2023', ...] 연도 컬럼 리스트
        statement_type: 'BS', 'IS', 'CF'

    Returns:
        {
            'items': [{'name': ..., 'parent': ..., 'values': {year: val}, 'type': ...}],
            'groups': {'그룹명': {'items': [...], 'total': {year: val}}},
            'categories': {'category': [items]},
        }
    """
    result = {
        'items': [],
        'groups': {},
        'categories': {},
    }

    # 매핑을 raw_name 기준 딕셔너리로 변환
    mapping_dict = {item['raw_name']: item for item in mapping}

    # label_ko 컬럼 찾기
    label_col = None
    for col_name in ['label_ko', '계정과목', '항목', 'label']:
        if col_name in df.columns:
            label_col = col_name
            break

    if label_col is None:
        print(f"[AccountClassifier] DataFrame에 label 컬럼 없음: {list(df.columns)[:5]}")
        return result

    # 연도 컬럼 확인
    available_years = [y for y in years if y in df.columns]

    for _, row in df.iterrows():
        raw_name = str(row.get(label_col, '')).strip()
        if not raw_name:
            continue

        classification = mapping_dict.get(raw_name)
        if not classification:
            continue

        category = classification.get('standard_category')
        if category in ('section_header', 'subtotal', 'total', 'skip', None):
            continue

        # 연도별 값 추출
        values = {}
        for year in available_years:
            val = row.get(year)
            if val is not None:
                try:
                    val = float(str(val).replace(',', '').replace(' ', ''))
                    # 부호 적용
                    if classification.get('sign') == '-':
                        val = -abs(val)
                    values[year] = val
                except (ValueError, TypeError):
                    pass

        if not values:
            continue

        display_name = classification.get('display_name', raw_name)
        group = classification.get('group')

        item_entry = {
            'raw_name': raw_name,
            'display_name': display_name,
            'category': category,
            'group': group,
            'sign': classification.get('sign', '+'),
            'values': values,
        }

        result['items'].append(item_entry)

        # 카테고리별 분류
        if category not in result['categories']:
            result['categories'][category] = []
        result['categories'][category].append(item_entry)

        # 그룹별 합산
        if group:
            if group not in result['groups']:
                result['groups'][group] = {
                    'items': [],
                    'total': {y: 0 for y in available_years},
                }
            result['groups'][group]['items'].append(item_entry)
            for year, val in values.items():
                result['groups'][group]['total'][year] = \
                    result['groups'][group]['total'].get(year, 0) + val

    return result


# ============================================================
# 초기화 (모듈 로드 시 캐시 테이블 생성)
# ============================================================
try:
    init_classification_cache_table()
except Exception as e:
    print(f"[AccountClassifier] 초기화 경고: {e}")
