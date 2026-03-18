"""
VCM v2 (LLM 분류) 검증 테스트 스크립트

검증 체크리스트:
================

## 1. 기본 동작 검증
- [V-01] 서버 헬스체크 (GET / → 200)
- [V-02] DART 추출 → VCM v1 생성 정상
- [V-03] VCM v2 API 호출 성공 (POST /api/vcm-v2/{task_id})
- [V-04] VCM 비교 API 호출 성공 (POST /api/vcm-compare/{task_id})

## 2. LLM 분류 정확성 (BS)
- [V-05] BS 유동자산 분류: 현금, 단기금융상품, 매출채권 등 정확히 current_asset
- [V-06] BS 비유동자산 분류: 유형자산, 무형자산 등 정확히 non_current_asset
- [V-07] BS 유동부채 분류: 매입채무, 단기차입금 등 정확히 current_liability
- [V-08] BS 비유동부채 분류: 사채, 장기차입금 등 정확히 non_current_liability
- [V-09] BS 자본 분류: 자본금, 이익잉여금 등 정확히 equity
- [V-10] BS 총계/소계: section_header, subtotal, total 정확히 분류

## 3. LLM 분류 정확성 (IS)
- [V-11] IS 매출 분류: 매출액/영업수익 → revenue
- [V-12] IS 매출원가 분류: 매출원가/영업비용 → cogs
- [V-13] IS 판관비 분류: 판매비와관리비 + 하위항목 → sga
- [V-14] IS 영업이익 분류 + 부호: "영업손실(이익)" → sign: "-"
- [V-15] IS 금융비용 분류: "금융원가" → interest_expense (하드코딩에서 누락되던 케이스)
- [V-16] IS 당기순이익 분류 + 부호

## 4. 그룹핑 정확성
- [V-17] 매출채권및기타채권 그룹: 매출채권+미수금+선급금 등 합산
- [V-18] 유동차입부채 그룹: 단기차입금+유동성장기부채+유동성사채 합산
- [V-19] 기타자본구성요소 그룹: 자본잉여금+자본조정+기타포괄손익누계액
- [V-20] 인건비 그룹: 급여+퇴직급여+복리후생비

## 5. 수치 정확성 (v1 vs v2 비교)
- [V-21] 자산총계 일치 (v1 ≈ v2, 오차 ±1백만원)
- [V-22] 부채총계 일치
- [V-23] 자본총계 일치
- [V-24] 매출 일치
- [V-25] 영업이익 일치
- [V-26] 당기순이익 일치

## 6. 산술 검증 (크로스체크)
- [V-27] 자산총계 = 유동자산 + 비유동자산 (±2)
- [V-28] 부채와자본총계 = 부채총계 + 자본총계 (±2)
- [V-29] 매출총이익 = 매출 - 매출원가 (직접 계산 가능한 경우)
- [V-30] NWC = 유동자산 - 유동부채

## 7. 캐시 검증
- [V-31] 첫 호출: LLM 호출 발생 (캐시 미스)
- [V-32] 재호출: LLM 호출 없음 (캐시 히트)
- [V-33] DB에 캐시 레코드 저장 확인

## 8. 엣지케이스
- [V-34] 주석 참조: "(주1)", "[주석2,3]" 포함 계정명 정상 분류
- [V-35] 로마숫자: "Ⅴ.영업이익" → "영업이익"으로 정규화
- [V-36] "영업손실(이익)" 부호 반전 → sign: "-"
- [V-37] "금융원가" → interest_expense (기존 하드코딩 누락 케이스)

## 9. 프롬프트 일관성
- [V-38] 동일 입력 재실행 → 동일 분류 결과 (캐시 경유)

## 10. 에러 핸들링
- [V-39] 빈 계정명 리스트 → 빈 결과 반환 (에러 없음)
- [V-40] LLM 호출 실패 → 기본값 반환 (confidence=0)

Usage:
    python3 test_vcm_v2.py --url http://localhost:8002
    python3 test_vcm_v2.py --url http://localhost:8002 --company "삼성전자" --corp-code 00126380
    python3 test_vcm_v2.py --url http://localhost:8002 --unit-only  # account_classifier 단위 테스트만
"""

import os
import sys
import json
import time
import sqlite3
import asyncio
import argparse
import requests
from datetime import datetime

# ============================================================
# 테스트 기업 목록 (5개 - 제조/서비스/소형/엣지/비상장)
# ============================================================
TEST_COMPANIES = [
    {"name": "삼성전자", "corp_code": "00126380", "category": "대기업(제조)", "note": "표준 연결 BS/IS"},
    {"name": "E1", "corp_code": "00356361", "category": "서비스", "note": "매출 하위항목"},
    {"name": "아이티엠반도체", "corp_code": "00579980", "category": "제조(소형)", "note": "판관비 상세"},
    {"name": "경동인베스트", "corp_code": "00231567", "category": "엣지케이스", "note": "비유동자산 소계 중복"},
    {"name": "이노메트리", "corp_code": "01011888", "category": "엣지케이스", "note": "IS 구조 특성"},
]

# ============================================================
# 유틸리티
# ============================================================
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = []  # (test_id, status, message)

def log_result(test_id, passed, message=""):
    status = "PASS" if passed else "FAIL"
    marker = PASS if passed else FAIL
    results.append((test_id, status, message))
    print(f"  {marker} {test_id}: {message}")

def log_skip(test_id, message=""):
    results.append((test_id, "SKIP", message))
    print(f"  {SKIP} {test_id}: {message}")

def parse_number(s):
    """'1,234' → 1234, 1234.0 → 1234, '' → 0"""
    if s is None:
        return 0
    if isinstance(s, (int, float)):
        import math
        if math.isnan(s):
            return 0
        return int(round(s))
    s_str = str(s).strip()
    if not s_str or s_str == '':
        return 0
    try:
        return int(round(float(s_str.replace(',', '').replace(' ', ''))))
    except (ValueError, TypeError):
        return 0


# ============================================================
# Part A: account_classifier 단위 테스트
# ============================================================
async def test_unit_classifier():
    """account_classifier 모듈 단위 테스트"""
    print("\n" + "=" * 60)
    print("Part A: account_classifier 단위 테스트")
    print("=" * 60)

    sys.path.insert(0, os.path.dirname(__file__))
    from account_classifier import (
        classify_accounts, _normalize_account_name,
        _get_cached_classifications, _save_classifications_to_cache,
        init_classification_cache_table
    )

    # V-34: 주석 참조 정규화
    test_cases = [
        ('현금및현금성자산(주1)', '현금및현금성자산'),
        ('Ⅴ.영업이익', '영업이익'),
        ('유형자산[주석2,3]', '유형자산'),
        ('III.매출총이익', '매출총이익'),
    ]
    for raw, expected in test_cases:
        result = _normalize_account_name(raw)
        log_result("V-34/V-35", result == expected,
                   f"normalize('{raw}') → '{result}' (expected: '{expected}')")

    # V-39: 빈 리스트 처리
    empty_result = await classify_accounts([], 'BS', 'TEST')
    log_result("V-39", empty_result == [],
               f"classify_accounts([]) → {empty_result}")

    # V-05~V-10: BS 분류 테스트 (실제 LLM 호출)
    print(f"\n  {INFO} BS 분류 테스트 (LLM 호출)...")
    bs_test_accounts = [
        '현금및현금성자산', '단기금융상품', '매출채권', '미수금', '재고자산',
        '유형자산', '무형자산', '사용권자산', '장기금융상품',
        '매입채무', '미지급금', '단기차입금', '유동성장기부채',
        '사채', '장기차입금', '퇴직급여충당부채',
        '자본금', '이익잉여금', '자본잉여금', '기타포괄손익누계액',
        '자산총계', '부채총계', '자본총계',
        'Ⅰ.유동자산',  # section_header
    ]

    bs_mapping = await classify_accounts(bs_test_accounts, 'BS', 'TEST_UNIT', '제조업')
    bs_map = {item['raw_name']: item for item in bs_mapping}

    # V-05: 유동자산
    for name in ['현금및현금성자산', '단기금융상품', '매출채권', '재고자산']:
        cls = bs_map.get(name, {})
        log_result("V-05", cls.get('standard_category') == 'current_asset',
                   f"BS '{name}' → {cls.get('standard_category')} (expected: current_asset)")

    # V-06: 비유동자산
    for name in ['유형자산', '무형자산', '사용권자산']:
        cls = bs_map.get(name, {})
        log_result("V-06", cls.get('standard_category') == 'non_current_asset',
                   f"BS '{name}' → {cls.get('standard_category')} (expected: non_current_asset)")

    # V-07: 유동부채
    for name in ['매입채무', '미지급금', '단기차입금']:
        cls = bs_map.get(name, {})
        log_result("V-07", cls.get('standard_category') == 'current_liability',
                   f"BS '{name}' → {cls.get('standard_category')} (expected: current_liability)")

    # V-08: 비유동부채
    for name in ['사채', '장기차입금', '퇴직급여충당부채']:
        cls = bs_map.get(name, {})
        log_result("V-08", cls.get('standard_category') == 'non_current_liability',
                   f"BS '{name}' → {cls.get('standard_category')} (expected: non_current_liability)")

    # V-09: 자본
    for name in ['자본금', '이익잉여금', '자본잉여금']:
        cls = bs_map.get(name, {})
        log_result("V-09", cls.get('standard_category') == 'equity',
                   f"BS '{name}' → {cls.get('standard_category')} (expected: equity)")

    # V-10: 총계/헤더
    cls = bs_map.get('자산총계', {})
    log_result("V-10", cls.get('standard_category') == 'total',
               f"BS '자산총계' → {cls.get('standard_category')} (expected: total)")
    cls = bs_map.get('Ⅰ.유동자산', {})
    log_result("V-10", cls.get('standard_category') in ('section_header', 'subtotal'),
               f"BS 'Ⅰ.유동자산' → {cls.get('standard_category')} (expected: section_header/subtotal)")

    # V-17: 그룹핑 - 매출채권및기타채권
    for name in ['매출채권', '미수금']:
        cls = bs_map.get(name, {})
        log_result("V-17", cls.get('group') == '매출채권및기타채권',
                   f"BS '{name}' group → '{cls.get('group')}' (expected: '매출채권및기타채권')")

    # V-19: 그룹핑 - 기타자본구성요소
    for name in ['자본잉여금', '기타포괄손익누계액']:
        cls = bs_map.get(name, {})
        log_result("V-19", cls.get('group') == '기타자본구성요소',
                   f"BS '{name}' group → '{cls.get('group')}' (expected: '기타자본구성요소')")

    # V-11~V-16: IS 분류 테스트
    print(f"\n  {INFO} IS 분류 테스트 (LLM 호출)...")
    is_test_accounts = [
        '매출액', '매출원가', '매출총이익',
        '판매비와관리비', '급여', '퇴직급여', '복리후생비', '감가상각비',
        '영업이익', '영업손실(이익)',
        '이자수익', '이자비용', '금융원가', '금융수익', '금융비용',
        '기타수익', '기타비용',
        '법인세비용차감전이익', '법인세비용', '당기순이익', '당기순손실',
    ]

    is_mapping = await classify_accounts(is_test_accounts, 'IS', 'TEST_UNIT', '제조업')
    is_map = {item['raw_name']: item for item in is_mapping}

    # V-11: 매출
    cls = is_map.get('매출액', {})
    log_result("V-11", cls.get('standard_category') == 'revenue',
               f"IS '매출액' → {cls.get('standard_category')} (expected: revenue)")

    # V-12: 매출원가
    cls = is_map.get('매출원가', {})
    log_result("V-12", cls.get('standard_category') == 'cogs',
               f"IS '매출원가' → {cls.get('standard_category')} (expected: cogs)")

    # V-13: 판관비
    cls = is_map.get('판매비와관리비', {})
    log_result("V-13", cls.get('standard_category') == 'sga',
               f"IS '판매비와관리비' → {cls.get('standard_category')} (expected: sga)")
    cls = is_map.get('급여', {})
    log_result("V-13", cls.get('standard_category') == 'sga',
               f"IS '급여' → {cls.get('standard_category')} (expected: sga)")

    # V-14: 영업이익 부호
    cls = is_map.get('영업이익', {})
    log_result("V-14", cls.get('standard_category') == 'operating_income' and cls.get('sign') == '+',
               f"IS '영업이익' → cat={cls.get('standard_category')}, sign={cls.get('sign')}")

    # V-36: 영업손실(이익) 부호 반전
    cls = is_map.get('영업손실(이익)', {})
    log_result("V-36", cls.get('sign') == '-',
               f"IS '영업손실(이익)' → cat={cls.get('standard_category')}, sign={cls.get('sign')} (expected: sign='-')")

    # V-15/V-37: 금융원가 → interest_expense (핵심 엣지케이스!)
    cls = is_map.get('금융원가', {})
    log_result("V-15/V-37", cls.get('standard_category') == 'interest_expense',
               f"IS '금융원가' → {cls.get('standard_category')} (expected: interest_expense) ★핵심 엣지케이스")

    # V-16: 당기순이익/당기순손실
    cls = is_map.get('당기순이익', {})
    log_result("V-16", cls.get('standard_category') == 'net_income',
               f"IS '당기순이익' → {cls.get('standard_category')}")
    cls = is_map.get('당기순손실', {})
    log_result("V-16", cls.get('sign') == '-',
               f"IS '당기순손실' → sign={cls.get('sign')} (expected: '-')")

    # V-20: 인건비 그룹
    for name in ['급여', '퇴직급여', '복리후생비']:
        cls = is_map.get(name, {})
        log_result("V-20", cls.get('group') in ('인건비', 'sga_detail'),
                   f"IS '{name}' group → '{cls.get('group')}'")

    # V-31~V-33: 캐시 검증
    print(f"\n  {INFO} 캐시 검증...")
    cached = _get_cached_classifications('TEST_UNIT', 'BS', ['현금및현금성자산'])
    log_result("V-33", '현금및현금성자산' in cached,
               f"DB 캐시 조회: {'현금및현금성자산' in cached} ({len(cached)}개 히트)")

    # V-32: 재호출 캐시 히트 (LLM 호출 없이 즉시 반환)
    t0 = time.time()
    bs_cached = await classify_accounts(bs_test_accounts, 'BS', 'TEST_UNIT', '제조업')
    t1 = time.time()
    cache_time = t1 - t0
    log_result("V-32", cache_time < 1.0,
               f"캐시 히트 응답시간: {cache_time:.3f}초 (expected: <1s)")

    # V-38: 프롬프트 일관성 (캐시 경유이므로 동일 결과 보장)
    bs_cached_map = {item['raw_name']: item for item in bs_cached}
    same = all(bs_cached_map.get(k, {}).get('standard_category') == v.get('standard_category')
               for k, v in bs_map.items())
    log_result("V-38", same, f"재실행 일관성: {'동일' if same else '불일치'}")


# ============================================================
# Part B: 통합 테스트 (서버 + DART 추출 + v1 vs v2 비교)
# ============================================================
def test_integration(base_url, company_name, corp_code, category=""):
    """단일 기업 통합 테스트"""
    print(f"\n{'=' * 60}")
    print(f"기업: {company_name} ({corp_code}) [{category}]")
    print(f"{'=' * 60}")

    # V-01: 서버 헬스체크
    try:
        r = requests.get(f"{base_url}/", timeout=10)
        log_result("V-01", r.status_code == 200, f"서버 상태: {r.status_code}")
    except Exception as e:
        log_result("V-01", False, f"서버 연결 실패: {e}")
        return None

    # V-02: DART 추출 시작
    print(f"  {INFO} DART 추출 시작...")
    try:
        r = requests.post(f"{base_url}/api/extract", json={
            "corp_code": corp_code,
            "corp_name": company_name,
            "start_year": 2023,
            "end_year": 2024,
        }, timeout=30)
        data = r.json()
        task_id = data.get('task_id')
        log_result("V-02", task_id is not None, f"task_id: {task_id}")
    except Exception as e:
        log_result("V-02", False, f"추출 시작 실패: {e}")
        return None

    if not task_id:
        return None

    # 추출 완료 대기 (최대 360초)
    print(f"  {INFO} 추출 완료 대기...")
    status = None
    for _ in range(72):  # 72 * 5s = 360s
        time.sleep(5)
        try:
            r = requests.get(f"{base_url}/api/status/{task_id}", timeout=10)
            status_data = r.json()
            status = status_data.get('status')
            progress = status_data.get('progress', 0)
            step = status_data.get('step', '')
            if progress > 0 or step:
                print(f"    진행중: {progress}% - {step}")
            if status == 'completed':
                break
            elif status == 'error':
                log_result("V-02", False, f"추출 에러: {status_data.get('error', 'unknown')}")
                return task_id
        except:
            pass

    log_result("V-02", status == 'completed', f"추출 상태: {status}")

    if status != 'completed':
        return task_id

    # V-03: VCM v2 실행
    print(f"\n  {INFO} VCM v2 (LLM 분류) 실행...")
    try:
        t0 = time.time()
        r = requests.post(f"{base_url}/api/vcm-v2/{task_id}", timeout=300)
        t1 = time.time()
        v2_data = r.json()
        v2_success = v2_data.get('success', False)
        log_result("V-03", v2_success,
                   f"VCM v2: success={v2_success}, "
                   f"frontdata={v2_data.get('stats', {}).get('frontdata_rows', 0)}행, "
                   f"financials={v2_data.get('stats', {}).get('financials_rows', 0)}행, "
                   f"소요시간: {t1 - t0:.1f}초")

        if not v2_success:
            print(f"    에러: {v2_data.get('error', 'unknown')}")
            if 'traceback' in v2_data:
                print(f"    {v2_data['traceback'][:500]}")
            return task_id
    except Exception as e:
        log_result("V-03", False, f"VCM v2 실패: {e}")
        return task_id

    # V-04: v1 vs v2 비교
    print(f"\n  {INFO} v1 vs v2 비교...")
    try:
        r = requests.post(f"{base_url}/api/vcm-compare/{task_id}", timeout=300)
        cmp = r.json()
        cmp_success = cmp.get('success', False)
        log_result("V-04", cmp_success, f"비교 API: success={cmp_success}")

        if cmp_success:
            total = cmp.get('total_cells', 0)
            matches = cmp.get('matches', 0)
            match_rate = cmp.get('match_rate', '0%')
            diffs = cmp.get('diffs', [])
            v1_only = cmp.get('only_in_v1', [])
            v2_only = cmp.get('only_in_v2', [])

            print(f"    총 셀: {total}, 일치: {matches}, 일치율: {match_rate}")
            print(f"    v1에만 있는 항목 ({len(v1_only)}개): {v1_only[:10]}")
            print(f"    v2에만 있는 항목 ({len(v2_only)}개): {v2_only[:10]}")

            if diffs:
                print(f"    차이 항목 ({len(diffs)}개, 상위 10개):")
                for d in diffs[:10]:
                    print(f"      {d['item']} [{d['year']}]: v1={d['v1']} vs v2={d['v2']}")

            # V-21~V-26: 주요 항목 수치 비교
            v2_display = v2_data.get('display_v2', [])
            v2_by_name = {r.get('항목', ''): r for r in v2_display}

            # v1 데이터 가져오기
            try:
                status_r = requests.get(f"{base_url}/api/status/{task_id}", timeout=10)
                v1_display = status_r.json().get('preview_data', {}).get('vcm_display', [])
            except:
                v1_display = []
            v1_by_name = {r.get('항목', ''): r for r in v1_display}

            key_items = [
                ("V-21", "자산총계"),
                ("V-22", "부채총계"),
                ("V-23", "자본총계"),
                ("V-24", "매출"),
                ("V-25", "영업이익"),
                ("V-26", "당기순이익"),
            ]

            for test_id, item_name in key_items:
                v1_row = v1_by_name.get(item_name, {})
                v2_row = v2_by_name.get(item_name, {})

                # 최신 연도 비교
                for col in sorted([k for k in v1_row if k.startswith('FY')], reverse=True)[:1]:
                    v1_val = parse_number(v1_row.get(col, ''))
                    v2_val = parse_number(v2_row.get(col, ''))
                    diff = abs(v1_val - v2_val)
                    log_result(test_id, diff <= 1,
                               f"'{item_name}' [{col}]: v1={v1_val:,} vs v2={v2_val:,} (차이: {diff:,})")

            # V-27~V-30: 산술 검증 (v2 기준)
            print(f"\n  {INFO} 산술 검증 (v2 기준)...")
            for col in sorted([k for k in v2_by_name.get('자산총계', {}) if k.startswith('FY')], reverse=True)[:1]:
                자산 = parse_number(v2_by_name.get('자산총계', {}).get(col, ''))
                유동자산 = parse_number(v2_by_name.get('유동자산', {}).get(col, ''))
                비유동자산 = parse_number(v2_by_name.get('비유동자산', {}).get(col, ''))

                if 자산 and (유동자산 or 비유동자산):
                    diff = abs(자산 - (유동자산 + 비유동자산))
                    log_result("V-27", diff <= 2,
                               f"자산총계({자산:,}) = 유동({유동자산:,}) + 비유동({비유동자산:,}), 차이: {diff:,}")
                else:
                    log_skip("V-27", f"자산총계={자산}, 유동={유동자산}, 비유동={비유동자산}")

                부채 = parse_number(v2_by_name.get('부채총계', {}).get(col, ''))
                자본 = parse_number(v2_by_name.get('자본총계', {}).get(col, ''))
                부자총 = parse_number(v2_by_name.get('부채와자본총계', {}).get(col, ''))

                if 부자총 and (부채 or 자본):
                    diff = abs(부자총 - (부채 + 자본))
                    log_result("V-28", diff <= 2,
                               f"부채와자본({부자총:,}) = 부채({부채:,}) + 자본({자본:,}), 차이: {diff:,}")
                else:
                    log_skip("V-28", f"부채와자본총계={부자총}, 부채={부채}, 자본={자본}")

                매출 = parse_number(v2_by_name.get('매출', {}).get(col, ''))
                원가 = parse_number(v2_by_name.get('매출원가', {}).get(col, ''))
                총이익 = parse_number(v2_by_name.get('매출총이익', {}).get(col, ''))

                if 매출 and 원가:
                    expected = 매출 - 원가
                    diff = abs(총이익 - expected)
                    log_result("V-29", diff <= 2,
                               f"매출총이익({총이익:,}) = 매출({매출:,}) - 원가({원가:,}), expected: {expected:,}")
                else:
                    log_skip("V-29", f"매출={매출}, 원가={원가}")

                유동자산_n = parse_number(v2_by_name.get('유동자산', {}).get(col, ''))
                유동부채_n = parse_number(v2_by_name.get('유동부채', {}).get(col, ''))
                nwc = parse_number(v2_by_name.get('NWC', {}).get(col, ''))

                if 유동자산_n or 유동부채_n:
                    expected_nwc = 유동자산_n - 유동부채_n
                    diff = abs(nwc - expected_nwc)
                    log_result("V-30", diff <= 2,
                               f"NWC({nwc:,}) = 유동자산({유동자산_n:,}) - 유동부채({유동부채_n:,}), expected: {expected_nwc:,}")
                else:
                    log_skip("V-30", f"유동자산={유동자산_n}, 유동부채={유동부채_n}")

        else:
            print(f"    비교 실패: {cmp.get('error', 'unknown')}")

    except Exception as e:
        log_result("V-04", False, f"비교 실패: {e}")

    return task_id


# ============================================================
# 결과 요약
# ============================================================
def print_summary():
    print("\n" + "=" * 60)
    print("검증 결과 요약")
    print("=" * 60)

    pass_count = sum(1 for _, s, _ in results if s == "PASS")
    fail_count = sum(1 for _, s, _ in results if s == "FAIL")
    skip_count = sum(1 for _, s, _ in results if s == "SKIP")
    total = len(results)

    print(f"총 {total}개 테스트: {PASS} {pass_count}개 | {FAIL} {fail_count}개 | {SKIP} {skip_count}개")

    if fail_count > 0:
        print(f"\n실패 항목:")
        for test_id, status, msg in results:
            if status == "FAIL":
                print(f"  {FAIL} {test_id}: {msg}")

    print(f"\n합격률: {pass_count}/{pass_count + fail_count} ({pass_count / (pass_count + fail_count) * 100:.1f}%)" if pass_count + fail_count > 0 else "")


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="VCM v2 검증 테스트")
    parser.add_argument('--url', default='http://localhost:8002', help='서버 URL')
    parser.add_argument('--company', help='특정 기업만 테스트')
    parser.add_argument('--corp-code', help='특정 기업 corp_code')
    parser.add_argument('--unit-only', action='store_true', help='단위 테스트만 실행')
    parser.add_argument('--skip-unit', action='store_true', help='단위 테스트 건너뛰기')
    args = parser.parse_args()

    print(f"VCM v2 검증 테스트 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"서버: {args.url}")

    # Part A: 단위 테스트
    if not args.skip_unit:
        asyncio.run(test_unit_classifier())

    if args.unit_only:
        print_summary()
        return

    # Part B: 통합 테스트
    if args.company and args.corp_code:
        test_integration(args.url, args.company, args.corp_code, "수동지정")
    else:
        for co in TEST_COMPANIES:
            task_id = test_integration(args.url, co['name'], co['corp_code'], co['category'])

    print_summary()


if __name__ == '__main__':
    main()
