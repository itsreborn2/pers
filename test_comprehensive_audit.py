#!/usr/bin/env python3
"""
종합 데이터 품질 검증 스크립트
- 다양한 업종/규모/구조의 기업을 자동 추출 + 검증
- 모든 잠재적 숫자 오류를 자동 감지
- DART 서버 부하 방지: 순차 추출, 기업간 10초 대기

사용법:
  python3 test_comprehensive_audit.py [--url URL] [--quick]
  --url: 서버 URL (기본: http://localhost:8002)
  --quick: 빠른 검증 (5개 기업만)
"""
import requests
import json
import time
import sys
import os

BASE_URL = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else "http://localhost:8002"
QUICK = '--quick' in sys.argv
EMAIL = os.environ.get("AUDIT_EMAIL", "admin@example.com")
PASSWORD = os.environ.get("AUDIT_PASSWORD", "admin123")
TIMEOUT = 300  # 추출 타임아웃 5분
DELAY = 10  # 기업간 대기 (DART rate limit 방지)

# ============================================================
# 테스트 기업 — 다양한 edge case 커버
# ============================================================
TEST_COMPANIES = [
    # (이름, corp_code, start_year, end_year, 특성)
    # 대형 상장사
    ("삼성전자", "00126380", 2021, 2024, "대형주, XBRL IS 없을 수 있음"),
    ("CJ대한통운", "00113410", 2021, 2025, "동일계정명 유동/비유동, EBITDA Notes 간헐적"),
    ("한화에어로스페이스", "00126566", 2020, 2024, "방산, 별도재무제표"),
    ("E1", "00165583", 2021, 2024, "중견기업"),
    ("패스트파이브", "01290406", 2021, 2024, "비상장, HTML IS 폴백, 영업손실"),
    # 대형주 다양한 업종
    ("현대자동차", "00164742", 2021, 2024, "자동차, 사용권자산 많음"),
    ("SK하이닉스", "00164779", 2021, 2024, "반도체, 장기미지급금"),
    ("NAVER", "01133821", 2021, 2024, "IT, 무형자산 많음"),
    ("카카오", "01136920", 2021, 2024, "IT, 중단영업"),
    ("POSCO홀딩스", "00138040", 2021, 2024, "철강, 지분법"),
    # 특수 업종
    ("대한항공", "00113526", 2021, 2024, "항공, 리스부채/사용권자산 대규모"),
    ("현대건설", "00164478", 2021, 2024, "건설, COGS에 D&A 포함"),
    ("롯데케미칼", "00159055", 2021, 2024, "화학"),
    # 소형주/edge case
    ("아이티엠반도체", "00978290", 2021, 2024, "소형주"),
    ("노랑풍선", "00632845", 2020, 2024, "여행, 판관비 구조 특이"),
    # 자본잠식/특수
    ("영풍", "00126849", 2020, 2024, "자본잠식, D&A 미추출"),
    ("셀트리온", "00627498", 2021, 2024, "바이오"),
    ("LG에너지솔루션", "01571872", 2022, 2024, "배터리, 최근 상장"),
]

if QUICK:
    TEST_COMPANIES = TEST_COMPANIES[:5]

# ============================================================
# 검증 항목 정의
# ============================================================
def run_checks(name, vcm_d, vcm, years, traits):
    """모든 검증 항목 실행"""
    results = []
    latest_fy = years[-1]

    # --- 1. BS 균형 ---
    for fy in years:
        ta = next((r[fy] for r in vcm_d if r.get('항목') == '자산총계' and r.get(fy)), None)
        tlc = next((r[fy] for r in vcm_d if r.get('항목') == '부채와자본총계' and r.get(fy)), None)
        if ta and tlc:
            diff = abs(ta - tlc)
            if diff > 2:
                results.append(('FAIL', f'BS균형 {fy}: diff={diff:,.0f}'))
        elif ta or tlc:
            results.append(('WARN', f'BS균형 {fy}: 자산총계={ta}, 부채와자본총계={tlc}'))
    if not any(r[0] == 'FAIL' for r in results if 'BS균형' in r[1]):
        results.append(('PASS', 'BS균형'))

    # --- 2. 비유동부채 sub-items GAP ---
    for fy in [latest_fy]:
        ncl_total = None
        ncl_sub = 0
        in_ncl = False
        for row in vcm_d:
            item = row.get('항목', '')
            v = row.get(fy)
            if item == '비유동부채':
                ncl_total = v
                in_ncl = True
                continue
            if item == '부채총계':
                break
            if in_ncl and v and isinstance(v, (int, float)):
                ncl_sub += v
        if ncl_total and ncl_total != 0:
            gap_pct = abs(ncl_total - ncl_sub) / abs(ncl_total) * 100
            if gap_pct > 5:
                results.append(('FAIL', f'비유동부채GAP {fy}: {gap_pct:.1f}% (총계={ncl_total:,.0f}, sub합={ncl_sub:,.0f})'))
            else:
                results.append(('PASS', f'비유동부채GAP {fy}: {gap_pct:.1f}%'))
        elif ncl_total == 0 or ncl_total is None:
            results.append(('WARN', f'비유동부채 {fy}: 총계=None 또는 0'))

    # --- 3. 비유동차입부채 존재 ---
    has_ncl_borrow = any('비유동차입' in str(r.get('항목', '')) for r in vcm_d)
    if has_ncl_borrow:
        results.append(('PASS', '비유동차입부채 존재'))
    else:
        # 비유동부채가 0이면 OK
        ncl = next((r.get(latest_fy) for r in vcm_d if r.get('항목') == '비유동부채'), None)
        if ncl and ncl > 100000:  # 1억 이상
            results.append(('WARN', '비유동차입부채 행 없음 (비유동부채 > 0)'))
        else:
            results.append(('PASS', '비유동차입부채 불필요'))

    # --- 4. 접미사 충돌 ([비유동부채], [비유동자산] 등) ---
    ugly_suffixes = [r.get('항목', '') for r in vcm_d
                     if '[비유동부채]' in str(r.get('항목', ''))
                     or '[비유동자산]' in str(r.get('항목', ''))
                     or '[유동부채]' in str(r.get('항목', ''))
                     or '[유동자산]' in str(r.get('항목', ''))]
    if ugly_suffixes:
        results.append(('WARN', f'섹션 접미사 충돌: {ugly_suffixes[:3]}'))
    else:
        results.append(('PASS', '접미사 충돌 없음'))

    # --- 5. IS 핵심 항목 전 연도 존재 ---
    for item_name in ['매출', '영업이익', '당기순이익']:
        row = next((r for r in vcm_d if r.get('항목') == item_name), None)
        if row:
            missing = [fy for fy in years if not row.get(fy)]
            if missing:
                results.append(('FAIL', f'{item_name} None: {missing}'))
            else:
                results.append(('PASS', f'{item_name} 전연도 존재'))
        else:
            results.append(('FAIL', f'{item_name} 행 없음'))

    # --- 6. 판매비와관리비 ---
    sga = next((r for r in vcm_d if r.get('항목') == '판매비와관리비'), None)
    if sga:
        sga_missing = [fy for fy in years if not sga.get(fy)]
        if sga_missing:
            results.append(('WARN', f'판관비 None: {sga_missing}'))
        else:
            results.append(('PASS', '판관비 전연도'))
    else:
        results.append(('WARN', '판관비 행 없음'))

    # --- 7. 법인세비용 ---
    tax = next((r for r in vcm_d if r.get('항목') == '법인세비용'), None)
    if tax:
        tax_missing = [fy for fy in years if not tax.get(fy)]
        if tax_missing:
            results.append(('WARN', f'법인세 None: {tax_missing}'))
        else:
            results.append(('PASS', '법인세 전연도'))
    else:
        results.append(('WARN', '법인세 행 없음'))

    # --- 8. EBITDA D&A ---
    ebitda = next((r for r in vcm_d if r.get('항목') == 'EBITDA'), None)
    oi = next((r for r in vcm_d if r.get('항목') == '영업이익'), None)
    if ebitda and oi:
        da_zero_years = []
        for fy in years:
            e = ebitda.get(fy)
            o = oi.get(fy)
            if e and o and e == o:
                da_zero_years.append(fy)
        if da_zero_years:
            # 유형자산 확인 — 자산 있으면 D&A=0은 문제
            tangible = next((r.get(latest_fy) for r in vcm_d if r.get('항목') == '유형자산'), 0) or 0
            rou = next((r.get(latest_fy) for r in vcm_d if r.get('항목') == '사용권자산'), 0) or 0
            if tangible + rou > 100000:  # 1억 이상
                results.append(('FAIL', f'EBITDA D&A=0: {da_zero_years} (유형자산={tangible:,.0f}, 사용권={rou:,.0f})'))
            else:
                results.append(('PASS', f'EBITDA D&A=0 OK (자산 미보유)'))
        else:
            results.append(('PASS', 'EBITDA D&A>0 전연도'))
    else:
        results.append(('WARN', 'EBITDA 또는 영업이익 행 없음'))

    # --- 9. Net Debt ---
    nd = next((r for r in vcm_d if r.get('항목') == 'Net Debt'), None)
    if nd:
        nd_val = nd.get(latest_fy)
        if nd_val is not None:
            results.append(('PASS', f'Net Debt {latest_fy}={nd_val:,.0f}'))
        else:
            results.append(('WARN', f'Net Debt {latest_fy}=None'))
    else:
        results.append(('WARN', 'Net Debt 행 없음'))

    # --- 10. 유동자산 sub-items GAP ---
    for fy in [latest_fy]:
        ca_total = next((r.get(fy) for r in vcm_d if r.get('항목') == '유동자산'), None)
        if ca_total and ca_total != 0:
            ca_sub = 0
            in_ca = False
            for row in vcm_d:
                item = row.get('항목', '')
                v = row.get(fy)
                if item == '유동자산':
                    in_ca = True
                    continue
                if item in ('비유동자산', '자산총계'):
                    break
                if in_ca and v and isinstance(v, (int, float)):
                    ca_sub += v
            gap_pct = abs(ca_total - ca_sub) / abs(ca_total) * 100
            if gap_pct > 5:
                results.append(('WARN', f'유동자산GAP {fy}: {gap_pct:.1f}%'))
            else:
                results.append(('PASS', f'유동자산GAP {fy}: {gap_pct:.1f}%'))

    # --- 11. 영업외수익 구성 ---
    oe_income = next((r for r in vcm_d if r.get('항목') == '영업외수익'), None)
    fin_income = next((r for r in vcm_d if r.get('항목', '').strip() == '금융수익'), None)
    if oe_income and fin_income:
        for fy in [latest_fy]:
            oe_v = oe_income.get(fy)
            fi_v = fin_income.get(fy)
            if oe_v and fi_v and oe_v == fi_v:
                results.append(('WARN', f'영업외수익={fy}=금융수익 (기타수익/지분법 누락 가능)'))

    # --- 12. % of Sales 계산 ---
    pct_row = next((r for r in vcm_d if r.get('항목') == '% of Sales'), None)
    revenue = next((r for r in vcm_d if r.get('항목') == '매출'), None)
    if pct_row and revenue:
        for fy in years:
            rv = revenue.get(fy)
            pv = pct_row.get(fy)
            if rv and not pv:
                results.append(('WARN', f'% of Sales {fy}=None (매출={rv:,.0f})'))

    return results


def main():
    session = requests.Session()

    print(f"{'=' * 80}")
    print(f"  종합 데이터 품질 검증")
    print(f"  서버: {BASE_URL}")
    print(f"  기업 수: {len(TEST_COMPANIES)}")
    print(f"  모드: {'QUICK (5개)' if QUICK else 'FULL (18개)'}")
    print(f"{'=' * 80}\n")

    # 로그인
    r = session.post(f"{BASE_URL}/api/auth/login",
                     json={"email": EMAIL, "password": PASSWORD}, timeout=30)
    if r.status_code != 200 or not r.json().get("success"):
        print(f"로그인 실패: {r.status_code}")
        return
    print("로그인 OK\n")

    all_results = []

    for idx, (name, code, sy, ey, traits) in enumerate(TEST_COMPANIES):
        print(f"{'─' * 60}")
        print(f"[{idx + 1}/{len(TEST_COMPANIES)}] {name} ({code}) FY{sy}~FY{ey}")
        print(f"  특성: {traits}")
        print(f"{'─' * 60}")

        # 추출 시작
        try:
            r = session.post(f"{BASE_URL}/api/extract", json={
                "corp_code": code, "corp_name": name,
                "start_year": sy, "end_year": ey
            }, timeout=60)
            task_id = r.json().get("task_id")
        except Exception as e:
            print(f"  추출 시작 실패: {e}")
            all_results.append((name, 'ERROR', ['추출 시작 실패']))
            time.sleep(DELAY)
            continue

        # 완료 대기
        data = None
        for i in range(100):
            time.sleep(3)
            try:
                r = session.get(f"{BASE_URL}/api/status/{task_id}", timeout=120)
                d = r.json()
            except:
                continue
            status = d.get('status', '')
            if status == 'completed':
                data = d
                print(f"  완료 ({i * 3}s)")
                break
            elif status in ('failed', 'error'):
                print(f"  실패: {d.get('error', '')[:100]}")
                break
            if i % 20 == 0 and i > 0:
                print(f"  ... {i * 3}s ({status})")

        if not data:
            all_results.append((name, 'ERROR', ['추출 실패/타임아웃']))
            time.sleep(DELAY)
            continue

        vcm_d = data.get('preview_data', {}).get('vcm_display', [])
        vcm = data.get('preview_data', {}).get('vcm', [])
        years = sorted(set(k for r in vcm_d for k in r if k.startswith('FY')))

        if not vcm_d or not years:
            all_results.append((name, 'ERROR', ['VCM 데이터 없음']))
            time.sleep(DELAY)
            continue

        # 검증 실행
        checks = run_checks(name, vcm_d, vcm, years, traits)

        fails = [c for c in checks if c[0] == 'FAIL']
        warns = [c for c in checks if c[0] == 'WARN']
        passes = [c for c in checks if c[0] == 'PASS']

        overall = 'FAIL' if fails else ('WARN' if warns else 'PASS')
        all_results.append((name, overall, [c[1] for c in fails + warns]))

        # 결과 출력
        for status, msg in checks:
            icon = {'PASS': '✓', 'FAIL': '✗', 'WARN': '△'}.get(status, '?')
            if status != 'PASS':
                print(f"  {icon} {msg}")
        print(f"  → {len(passes)} PASS, {len(fails)} FAIL, {len(warns)} WARN")

        time.sleep(DELAY)

    # ============================================================
    # 종합 결과
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"  종합 결과")
    print(f"{'=' * 80}\n")

    total_pass = sum(1 for _, s, _ in all_results if s == 'PASS')
    total_warn = sum(1 for _, s, _ in all_results if s == 'WARN')
    total_fail = sum(1 for _, s, _ in all_results if s == 'FAIL')
    total_err = sum(1 for _, s, _ in all_results if s == 'ERROR')

    for name, status, issues in all_results:
        icon = {'PASS': '✓', 'WARN': '△', 'FAIL': '✗', 'ERROR': '!'}.get(status, '?')
        print(f"  {icon} {name:20s} [{status}]")
        for iss in issues[:3]:
            print(f"      {iss}")
        if len(issues) > 3:
            print(f"      ... +{len(issues) - 3}건")

    print(f"\n  PASS={total_pass}, WARN={total_warn}, FAIL={total_fail}, ERROR={total_err}")
    print(f"  총 {len(all_results)}개 기업 검증 완료")

    # 결과 JSON 저장
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'server': BASE_URL,
        'companies': len(all_results),
        'summary': {'pass': total_pass, 'warn': total_warn, 'fail': total_fail, 'error': total_err},
        'details': [{'name': n, 'status': s, 'issues': i} for n, s, i in all_results]
    }
    with open('audit_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: audit_results.json")


if __name__ == "__main__":
    main()
