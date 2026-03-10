#!/usr/bin/env python3
"""배치 추출 테스트 - 미검증 상장사 대상"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8001"
TARGETS_FILE = "/tmp/new_test_targets.json"

def test_one(name, stock_code, market, timeout=360):
    """한 기업 추출 테스트"""
    result = {
        "name": name,
        "stock_code": stock_code,
        "market": market,
        "status": "UNKNOWN",
        "corp_code": "",
        "fs_type": "",
        "balance_check": "",
        "asset_eq_check": "",
        "error": ""
    }

    try:
        # 서버 준비 대기 (이전 추출 완료 확인)
        for retry in range(10):
            try:
                r = requests.get(f"{BASE_URL}/", timeout=10)
                if r.status_code == 200:
                    break
            except:
                pass
            time.sleep(10)

        # 1) 기업 검색 (재시도 포함)
        resp = None
        for retry in range(3):
            try:
                resp = requests.post(f"{BASE_URL}/api/search", json={"company_name": name}, timeout=120)
                break
            except Exception as e:
                if retry < 2:
                    time.sleep(10)
                else:
                    raise
        search_data = resp.json()
        if isinstance(search_data, dict):
            search_results = search_data.get('results', search_data.get('data', []))
        elif isinstance(search_data, list):
            search_results = search_data
        else:
            search_results = []

        # stock_code로 매칭
        corp_code = None
        for r in search_results:
            if r.get('stock_code') == stock_code:
                corp_code = r['corp_code']
                break

        if not corp_code:
            # 이름으로 재시도
            for r in search_results:
                if name in r.get('corp_name', ''):
                    corp_code = r['corp_code']
                    break

        if not corp_code:
            result["status"] = "RESOLVE_FAIL"
            result["error"] = f"corp_code 미발견 (검색결과 {len(search_results)}건)"
            return result

        result["corp_code"] = corp_code

        # 2) 추출
        resp = requests.post(f"{BASE_URL}/api/extract", json={
            "corp_code": corp_code,
            "corp_name": name,
            "start_year": 2021,
            "end_year": 2024,
            "company_info": {}
        }, timeout=30)
        task_id = resp.json().get('task_id')

        # 3) 폴링
        for i in range(int(timeout / 3)):
            time.sleep(3)
            resp = requests.get(f"{BASE_URL}/api/status/{task_id}", timeout=10)
            sd = resp.json()
            status = sd.get('status', '?')

            if status == 'completed':
                preview = sd.get('preview_data', {})
                vcm = preview.get('vcm', [])
                vcm_display = preview.get('vcm_display', [])
                bs = preview.get('bs', [])
                is_data = preview.get('is', [])

                if not vcm_display and not vcm:
                    result["status"] = "NO_DATA"
                    result["error"] = "VCM 데이터 없음"
                    return result

                # 연결/별도 확인
                result["fs_type"] = "연결" if sd.get('message', '').find('별도') < 0 else "별도"

                # Balance check: 부채와자본총계 = 부채총계 + 자본총계
                result["status"] = "SUCCESS"

                # VCM에서 수치 확인
                def find_vcm_val(items, item_name, year_col=None):
                    for row in items:
                        if isinstance(row, dict) and item_name in str(row.get('항목', '')):
                            if year_col:
                                for k, v in row.items():
                                    if year_col in k and v:
                                        try:
                                            return float(str(v).replace(',', ''))
                                        except:
                                            pass
                            return row
                        elif isinstance(row, list) and len(row) > 0 and item_name in str(row[0]):
                            return row
                    return None

                # BS 원본에서 balance check
                if bs:
                    자산총계 = None
                    유동자산 = None
                    비유동자산 = None
                    부채자본총계 = None
                    부채총계 = None
                    자본총계 = None

                    for row in bs:
                        if isinstance(row, dict):
                            name_val = row.get('계정과목', row.get('항목', ''))
                        elif isinstance(row, list) and len(row) > 0:
                            name_val = str(row[0])
                        else:
                            continue

                        def get_last_year_val(r):
                            """마지막 연도 값 가져오기"""
                            if isinstance(r, dict):
                                for k in sorted(r.keys(), reverse=True):
                                    if '20' in k:
                                        try:
                                            v = str(r[k]).replace(',', '').strip()
                                            if v and v != '-' and v != 'None':
                                                neg = '(' in v
                                                v = v.replace('(', '').replace(')', '')
                                                return -float(v) if neg else float(v)
                                        except:
                                            pass
                            return None

                        if '자산총계' in name_val and '부채' not in name_val:
                            자산총계 = get_last_year_val(row)
                        elif '유동자산' == name_val.strip():
                            유동자산 = get_last_year_val(row)
                        elif '비유동자산' == name_val.strip():
                            비유동자산 = get_last_year_val(row)
                        elif '부채와자본총계' in name_val or '부채및자본총계' in name_val:
                            부채자본총계 = get_last_year_val(row)
                        elif '부채총계' in name_val:
                            부채총계 = get_last_year_val(row)
                        elif '자본총계' in name_val and '부채' not in name_val:
                            자본총계 = get_last_year_val(row)

                    # Balance check
                    if 부채총계 is not None and 자본총계 is not None and (부채자본총계 is not None or 자산총계 is not None):
                        total = 부채자본총계 or 자산총계
                        diff = abs(total - (부채총계 + 자본총계))
                        if diff <= 2:
                            result["balance_check"] = "PASS"
                        else:
                            result["balance_check"] = f"FAIL (diff={diff:,.0f})"
                    else:
                        result["balance_check"] = "SKIP (데이터 부족)"

                    # Asset equation
                    if 자산총계 is not None and 유동자산 is not None and 비유동자산 is not None:
                        diff = abs(자산총계 - (유동자산 + 비유동자산))
                        if diff <= 2:
                            result["asset_eq_check"] = "PASS"
                        else:
                            result["asset_eq_check"] = f"FAIL (diff={diff:,.0f})"
                    else:
                        result["asset_eq_check"] = "SKIP"

                return result

            elif status == 'failed':
                result["status"] = "EXTRACT_FAIL"
                result["error"] = sd.get('message', '')[:100]
                return result

        result["status"] = "TIMEOUT"
        return result

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)[:100]
        return result


def main():
    with open(TARGETS_FILE) as f:
        targets = json.load(f)

    print(f"\n{'='*80}")
    print(f"  배치 추출 테스트 - {len(targets)}개 종목")
    print(f"  서버: {BASE_URL}")
    print(f"{'='*80}")

    results = []
    counts = {"SUCCESS": 0, "NO_DATA": 0, "RESOLVE_FAIL": 0, "EXTRACT_FAIL": 0, "TIMEOUT": 0, "ERROR": 0}

    for i, t in enumerate(targets):
        name = t['name']
        stock_code = t['stock_code']
        market = t['market']

        print(f"\n[{i+1}/{len(targets)}] {name} ({stock_code}) {market}", end=" ... ", flush=True)

        start = time.time()
        result = test_one(name, stock_code, market)
        elapsed = time.time() - start

        status = result['status']
        counts[status] = counts.get(status, 0) + 1

        bal = result.get('balance_check', '')
        asset = result.get('asset_eq_check', '')
        extra = f" | Balance={bal} Asset={asset}" if status == "SUCCESS" else f" | {result.get('error', '')[:50]}"

        status_icon = {"SUCCESS": "✓", "NO_DATA": "△", "RESOLVE_FAIL": "✗", "EXTRACT_FAIL": "✗", "TIMEOUT": "⏱", "ERROR": "✗"}.get(status, "?")
        print(f"{status_icon} {status} ({elapsed:.0f}s){extra}")

        result["elapsed"] = round(elapsed, 1)
        results.append(result)

        # 서버 부하 줄이기 — 추출 작업 간 충분한 대기
        time.sleep(5)

    # 요약
    print(f"\n{'='*80}")
    print(f"  결과 요약")
    print(f"{'='*80}")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        if v > 0:
            print(f"  {k:15s}: {v}개")
    print(f"  {'─'*30}")
    print(f"  {'합계':15s}: {len(results)}개")

    # Balance/Asset check 요약
    bal_pass = sum(1 for r in results if r.get('balance_check') == 'PASS')
    bal_fail = sum(1 for r in results if 'FAIL' in r.get('balance_check', ''))
    asset_pass = sum(1 for r in results if r.get('asset_eq_check') == 'PASS')
    asset_fail = sum(1 for r in results if 'FAIL' in r.get('asset_eq_check', ''))

    success_count = counts.get('SUCCESS', 0)
    if success_count > 0:
        print(f"\n  Balance Check: {bal_pass}/{success_count} PASS, {bal_fail} FAIL")
        print(f"  Asset Eq Check: {asset_pass}/{success_count} PASS, {asset_fail} FAIL")

    # 실패 상세
    failures = [r for r in results if r['status'] not in ('SUCCESS',)]
    if failures:
        print(f"\n  실패/이슈 상세:")
        for r in failures:
            print(f"    {r['name']:20s} ({r['stock_code']}) → {r['status']}: {r.get('error','')[:60]}")

    # 저장
    output = {
        "test_date": time.strftime("%Y-%m-%d %H:%M"),
        "server": BASE_URL,
        "total": len(results),
        "counts": counts,
        "balance_pass": bal_pass,
        "balance_fail": bal_fail,
        "asset_pass": asset_pass,
        "asset_fail": asset_fail,
        "results": results
    }
    with open("/home/servermanager/pers-dev/test_batch_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: test_batch_results.json")

    return counts.get('EXTRACT_FAIL', 0) == 0 and counts.get('ERROR', 0) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
