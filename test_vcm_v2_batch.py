"""
VCM v2 배치 검증 테스트 — test_companies.json 전체 기업 병렬 실행

Usage:
    python3 test_vcm_v2_batch.py                           # 전체 (standard + edge + bottom50)
    python3 test_vcm_v2_batch.py --all                     # 소형주 배치까지 포함
    python3 test_vcm_v2_batch.py --category listed.standard
    python3 test_vcm_v2_batch.py --parallel 10             # 동시 10개
"""

import os
import sys
import json
import time
import math
import asyncio
import aiohttp
import argparse
from datetime import datetime
from collections import defaultdict

BASE_URL = "http://localhost:8002"
COMPANIES_FILE = "/home/servermanager/pers-dev/test_companies.json"

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def parse_number(s):
    if s is None:
        return 0
    if isinstance(s, (int, float)):
        if math.isnan(s):
            return 0
        return int(round(s))
    s_str = str(s).strip()
    if not s_str:
        return 0
    try:
        return int(round(float(s_str.replace(',', '').replace(' ', ''))))
    except (ValueError, TypeError):
        return 0


def load_companies(category_filter=None, include_all=False):
    """test_companies.json에서 테스트 가능한 기업 로드"""
    with open(COMPANIES_FILE) as f:
        data = json.load(f)

    companies = []
    skip_categories = {'known_failures', 'no_data'}

    def extract(obj, path=''):
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and item.get('corp_code'):
                    companies.append({
                        'name': item.get('name', ''),
                        'corp_code': item['corp_code'],
                        'category': path
                    })
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k in skip_categories:
                    continue
                new_path = f'{path}.{k}' if path else k
                extract(v, new_path)

    extract(data)

    if category_filter:
        # Support comma-separated categories and wildcards
        filters = [f.strip() for f in category_filter.split(',')]
        def matches_filter(category):
            for f in filters:
                target = f"listed.{f}" if not f.startswith(('listed.', 'unlisted.')) else f
                if target.endswith('*'):
                    if category.startswith(target[:-1]):
                        return True
                else:
                    if category == target:
                        return True
            return False
        companies[:] = [c for c in companies if matches_filter(c['category'])]
    elif not include_all:
        # Default: standard + edge_cases + bottom50 only
        core = {'listed.standard', 'listed.edge_cases', 'listed.bottom50_small_cap'}
        companies[:] = [c for c in companies if c['category'] in core]

    return companies


async def test_single_company(session, company, semaphore, results_list):
    """단일 기업 V2 검증 (비동기)"""
    name = company['name']
    corp_code = company['corp_code']
    category = company['category']
    result = {
        'name': name,
        'corp_code': corp_code,
        'category': category,
        'status': 'pending',
        'extract_time': 0,
        'v2_time': 0,
        'key_metrics': {},
        'arithmetic': {},
        'error': None,
    }

    async with semaphore:
        try:
            # Step 1: Extract
            t0 = time.time()
            async with session.post(f"{BASE_URL}/api/extract", json={
                "corp_code": corp_code,
                "corp_name": name,
                "start_year": 2023,
                "end_year": 2024,
            }, timeout=aiohttp.ClientTimeout(total=30)) as r:
                data = await r.json()
                task_id = data.get('task_id')

            if not task_id:
                result['status'] = 'extract_fail'
                result['error'] = data.get('error', 'no task_id')
                results_list.append(result)
                print(f"  {RED}✗{RESET} {name}: 추출 시작 실패 — {result['error']}")
                return

            # Step 2: Wait for extraction (max 360s)
            for _ in range(72):
                await asyncio.sleep(5)
                try:
                    async with session.get(f"{BASE_URL}/api/status/{task_id}",
                                          timeout=aiohttp.ClientTimeout(total=10)) as r:
                        status_data = await r.json()
                        status = status_data.get('status')
                        if status == 'completed':
                            break
                        elif status == 'error':
                            result['status'] = 'extract_error'
                            result['error'] = status_data.get('error') or status_data.get('message', 'unknown')
                            results_list.append(result)
                            # Shorten common errors
                            err_short = result['error']
                            if '재무제표를 찾을 수 없' in err_short:
                                err_short = 'DART 재무제표 없음'
                            elif 'GUEST_LIMIT' in err_short:
                                err_short = 'Guest limit exceeded'
                            print(f"  {RED}✗{RESET} {name}: 추출 에러 — {err_short[:80]}")
                            return
                except:
                    pass

            result['extract_time'] = time.time() - t0

            if status != 'completed':
                result['status'] = 'extract_timeout'
                results_list.append(result)
                print(f"  {RED}✗{RESET} {name}: 추출 타임아웃")
                return

            # Step 3: Run VCM v2
            t0 = time.time()
            async with session.post(f"{BASE_URL}/api/vcm-v2/{task_id}",
                                   timeout=aiohttp.ClientTimeout(total=300)) as r:
                v2_data = await r.json()
                v2_success = v2_data.get('success', False)

            result['v2_time'] = time.time() - t0

            if not v2_success:
                result['status'] = 'v2_fail'
                result['error'] = v2_data.get('error', 'unknown')
                results_list.append(result)
                print(f"  {RED}✗{RESET} {name}: V2 실패 — {result['error'][:80]}")
                return

            # Step 4: Compare v1 vs v2
            async with session.post(f"{BASE_URL}/api/vcm-compare/{task_id}",
                                   timeout=aiohttp.ClientTimeout(total=300)) as r:
                cmp = await r.json()

            if not cmp.get('success'):
                result['status'] = 'compare_fail'
                result['error'] = cmp.get('error', 'unknown')
                results_list.append(result)
                print(f"  {RED}✗{RESET} {name}: 비교 실패 — {result['error'][:80]}")
                return

            # Step 5: Key metric comparison
            v2_display = v2_data.get('display_v2', [])
            v2_by_name = {r.get('항목', ''): r for r in v2_display}

            # Get v1 data
            try:
                async with session.get(f"{BASE_URL}/api/status/{task_id}",
                                      timeout=aiohttp.ClientTimeout(total=10)) as r:
                    status_resp = await r.json()
                    v1_display = status_resp.get('preview_data', {}).get('vcm_display', [])
            except:
                v1_display = []
            v1_by_name = {r.get('항목', ''): r for r in v1_display}

            key_items = ['자산총계', '부채총계', '자본총계', '매출', '영업이익', '당기순이익', '매출총이익']
            metrics = {}
            all_match = True

            for item_name in key_items:
                v1_row = v1_by_name.get(item_name, {})
                v2_row = v2_by_name.get(item_name, {})

                # Find latest FY column
                fy_cols = sorted([k for k in list(v1_row.keys()) + list(v2_row.keys())
                                 if isinstance(k, str) and k.startswith('FY')], reverse=True)
                if not fy_cols:
                    continue

                col = fy_cols[0]
                v1_val = parse_number(v1_row.get(col, ''))
                v2_val = parse_number(v2_row.get(col, ''))
                diff = abs(v1_val - v2_val)
                match = diff <= 1

                metrics[item_name] = {
                    'v1': v1_val, 'v2': v2_val,
                    'diff': diff, 'match': match, 'fy': col
                }
                if not match and v1_val != 0:  # only count as mismatch if v1 has data
                    all_match = False

            result['key_metrics'] = metrics

            # Step 6: Arithmetic checks (v2 data)
            arith = {}
            fy_cols = sorted([k for k in v2_by_name.get('자산총계', {}).keys()
                             if isinstance(k, str) and k.startswith('FY')], reverse=True)
            if fy_cols:
                col = fy_cols[0]
                assets = parse_number(v2_by_name.get('자산총계', {}).get(col, ''))
                ca = parse_number(v2_by_name.get('유동자산', {}).get(col, ''))
                nca = parse_number(v2_by_name.get('비유동자산', {}).get(col, ''))
                if assets and (ca or nca):
                    arith['asset_eq'] = {'val': assets, 'calc': ca + nca,
                                         'diff': abs(assets - ca - nca),
                                         'pass': abs(assets - ca - nca) <= 2}

                liab = parse_number(v2_by_name.get('부채총계', {}).get(col, ''))
                eq = parse_number(v2_by_name.get('자본총계', {}).get(col, ''))
                total_le = parse_number(v2_by_name.get('부채와자본총계', {}).get(col, ''))
                if total_le and (liab or eq):
                    arith['balance'] = {'val': total_le, 'calc': liab + eq,
                                        'diff': abs(total_le - liab - eq),
                                        'pass': abs(total_le - liab - eq) <= 2}

            result['arithmetic'] = arith
            result['match_rate'] = cmp.get('match_rate', 'N/A')
            result['total_cells'] = cmp.get('total_cells', 0)
            result['matches'] = cmp.get('matches', 0)
            result['diffs_count'] = len(cmp.get('diffs', []))
            result['status'] = 'success'

            # Print result
            metric_strs = []
            for item in key_items:
                m = metrics.get(item)
                if not m:
                    continue
                mark = f"{GREEN}✓{RESET}" if m['match'] else f"{RED}✗{RESET}"
                if not m['match']:
                    metric_strs.append(f"{mark}{item}(v1={m['v1']:,},v2={m['v2']:,})")
                else:
                    metric_strs.append(f"{mark}{item}")

            arith_strs = []
            for k, a in arith.items():
                mark = f"{GREEN}✓{RESET}" if a['pass'] else f"{RED}✗{RESET}"
                arith_strs.append(f"{mark}{k}")

            status_mark = f"{GREEN}✓{RESET}" if all_match else f"{YELLOW}△{RESET}"
            print(f"  {status_mark} {name} [{category}] — "
                  f"추출{result['extract_time']:.0f}s, V2 {result['v2_time']:.1f}s | "
                  f"{' '.join(metric_strs)} | {' '.join(arith_strs)}")

        except Exception as e:
            result['status'] = 'exception'
            err_msg = str(e) or f"{type(e).__name__}"
            result['error'] = err_msg
            results_list.append(result)
            print(f"  {RED}✗{RESET} {name}: 예외 — {err_msg[:100]}")
            return

    results_list.append(result)


def print_batch_summary(results):
    """배치 테스트 결과 요약"""
    print(f"\n{'='*80}")
    print(f"배치 검증 결과 요약 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    total = len(results)
    success = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']

    print(f"\n총 {total}개 기업 | {GREEN}성공: {len(success)}{RESET} | {RED}실패: {len(failed)}{RESET}")

    # Failure breakdown
    if failed:
        fail_types = defaultdict(list)
        for r in failed:
            fail_types[r['status']].append(r['name'])
        print(f"\n실패 상세:")
        for status, names in fail_types.items():
            print(f"  {status} ({len(names)}): {', '.join(names[:10])}")

    # Key metric comparison
    if success:
        print(f"\n{'='*80}")
        print(f"주요 지표 비교 (v1 vs v2)")
        print(f"{'='*80}")

        key_items = ['자산총계', '부채총계', '자본총계', '매출', '영업이익', '당기순이익', '매출총이익']
        for item in key_items:
            match_count = 0
            mismatch_count = 0
            mismatches = []
            for r in success:
                m = r.get('key_metrics', {}).get(item)
                if not m:
                    continue
                if m['match']:
                    match_count += 1
                else:
                    if m['v1'] != 0:  # Skip if v1 has no data
                        mismatch_count += 1
                        mismatches.append(f"{r['name']}(v1={m['v1']:,},v2={m['v2']:,})")

            total_checked = match_count + mismatch_count
            if total_checked > 0:
                rate = match_count / total_checked * 100
                mark = f"{GREEN}✓{RESET}" if rate >= 90 else f"{YELLOW}△{RESET}" if rate >= 70 else f"{RED}✗{RESET}"
                print(f"  {mark} {item}: {match_count}/{total_checked} ({rate:.1f}%)")
                if mismatches:
                    for mm in mismatches[:5]:
                        print(f"      불일치: {mm}")

        # Arithmetic checks
        print(f"\n산술 검증:")
        for check in ['asset_eq', 'balance']:
            pass_count = sum(1 for r in success if r.get('arithmetic', {}).get(check, {}).get('pass'))
            fail_count = sum(1 for r in success if check in r.get('arithmetic', {}) and not r['arithmetic'][check]['pass'])
            total_checked = pass_count + fail_count
            if total_checked:
                rate = pass_count / total_checked * 100
                check_name = '자산=유동+비유동' if check == 'asset_eq' else '부채와자본=부채+자본'
                mark = f"{GREEN}✓{RESET}" if rate >= 95 else f"{RED}✗{RESET}"
                print(f"  {mark} {check_name}: {pass_count}/{total_checked} ({rate:.1f}%)")

        # Timing stats
        extract_times = [r['extract_time'] for r in success if r['extract_time'] > 0]
        v2_times = [r['v2_time'] for r in success if r['v2_time'] > 0]
        if extract_times:
            print(f"\n타이밍:")
            print(f"  추출: avg {sum(extract_times)/len(extract_times):.0f}s, "
                  f"min {min(extract_times):.0f}s, max {max(extract_times):.0f}s")
        if v2_times:
            print(f"  V2:   avg {sum(v2_times)/len(v2_times):.1f}s, "
                  f"min {min(v2_times):.1f}s, max {max(v2_times):.1f}s")

    # Save results to JSON
    output_file = f"/home/servermanager/pers-dev2/batch_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n결과 저장: {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="VCM v2 배치 검증")
    parser.add_argument('--url', default='http://localhost:8002')
    parser.add_argument('--parallel', type=int, default=5, help='동시 실행 수')
    parser.add_argument('--category', help='특정 카테고리만 (e.g., listed.standard)')
    parser.add_argument('--all', action='store_true', help='소형주 배치까지 전부 포함')
    parser.add_argument('--limit', type=int, default=0, help='최대 기업 수 (0=전체)')
    parser.add_argument('--names', help='특정 기업명 (콤마 구분, e.g., 앱코,피코그램)')
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url

    companies = load_companies(category_filter=args.category, include_all=args.all)
    if args.names:
        name_list = [n.strip() for n in args.names.split(',')]
        companies = [c for c in companies if c['name'] in name_list]
    if args.limit > 0:
        companies = companies[:args.limit]

    print(f"VCM v2 배치 검증 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"서버: {args.url} | 기업 수: {len(companies)} | 동시: {args.parallel}")

    # Group by category
    by_cat = defaultdict(int)
    for c in companies:
        by_cat[c['category']] += 1
    for cat, cnt in sorted(by_cat.items()):
        print(f"  {cat}: {cnt}개")

    # Check server
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/", timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    print(f"{RED}서버 응답 오류: {r.status}{RESET}")
                    return
    except Exception as e:
        print(f"{RED}서버 연결 실패: {e}{RESET}")
        return

    print(f"\n테스트 시작...")
    semaphore = asyncio.Semaphore(args.parallel)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = [test_single_company(session, co, semaphore, results) for co in companies]
        await asyncio.gather(*tasks)

    # Sort results by input order
    result_map = {r['corp_code']: r for r in results}
    sorted_results = [result_map.get(c['corp_code'], {'name': c['name'], 'status': 'missing'})
                      for c in companies]

    print_batch_summary(sorted_results)


if __name__ == '__main__':
    asyncio.run(main())
