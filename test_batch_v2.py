#!/usr/bin/env python3
"""배치 추출 테스트 v2 - corp_code 사전 조회 완료, 추출만 순차 실행"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8001"

def test_extract(name, corp_code, stock_code, market, timeout=300):
    """추출 & 검증"""
    result = {
        "name": name, "stock_code": stock_code, "corp_code": corp_code,
        "market": market, "status": "UNKNOWN",
        "fs_type": "", "balance_check": "", "asset_eq_check": "", "error": ""
    }

    try:
        # 추출 시작
        resp = requests.post(f"{BASE_URL}/api/extract", json={
            "corp_code": corp_code, "corp_name": name,
            "start_year": 2021, "end_year": 2024,
            "company_info": {}
        }, timeout=30)

        if resp.status_code != 200:
            result["status"] = "EXTRACT_FAIL"
            result["error"] = f"HTTP {resp.status_code}"
            return result

        task_id = resp.json().get('task_id')
        if not task_id:
            result["status"] = "EXTRACT_FAIL"
            result["error"] = "no task_id"
            return result

        # 폴링
        for i in range(int(timeout / 3)):
            time.sleep(3)
            try:
                resp = requests.get(f"{BASE_URL}/api/status/{task_id}", timeout=15)
                sd = resp.json()
            except:
                continue

            status = sd.get('status', '?')
            if status == 'completed':
                preview = sd.get('preview_data', {})
                vcm = preview.get('vcm', [])
                vcm_display = preview.get('vcm_display', [])

                if not vcm_display and not vcm:
                    result["status"] = "NO_DATA"
                    result["error"] = "VCM 없음"
                    return result

                result["status"] = "SUCCESS"

                # 별도/연결 확인
                msg = sd.get('message', '')
                result["fs_type"] = "별도" if '별도' in msg else "연결"

                # BS에서 balance check
                bs = preview.get('bs', [])
                if bs:
                    vals = {}
                    for row in bs:
                        if isinstance(row, dict):
                            nm = row.get('계정과목', row.get('항목', ''))
                        elif isinstance(row, list) and row:
                            nm = str(row[0])
                        else:
                            continue

                        def last_val(r):
                            if isinstance(r, dict):
                                for k in sorted(r.keys(), reverse=True):
                                    if '20' in k:
                                        try:
                                            v = str(r[k]).replace(',','').strip()
                                            if v and v not in ('-','None',''):
                                                neg = '(' in v
                                                v = v.replace('(','').replace(')','')
                                                return -float(v) if neg else float(v)
                                        except:
                                            pass
                            return None

                        for key in ['자산총계','유동자산','비유동자산','부채총계','자본총계','부채와자본총계','부채및자본총계']:
                            if key in nm and key not in vals:
                                v = last_val(row)
                                if v is not None:
                                    vals[key] = v

                    # Balance: 부채+자본 = 부채와자본총계 or 자산총계
                    total = vals.get('부채와자본총계', vals.get('부채및자본총계', vals.get('자산총계')))
                    debt = vals.get('부채총계')
                    equity = vals.get('자본총계')
                    if total and debt is not None and equity is not None:
                        diff = abs(total - (debt + equity))
                        result["balance_check"] = "PASS" if diff <= 2 else f"FAIL({diff:,.0f})"
                    else:
                        result["balance_check"] = "SKIP"

                    # Asset eq: 자산 = 유동 + 비유동
                    ta = vals.get('자산총계')
                    ca = vals.get('유동자산')
                    nca = vals.get('비유동자산')
                    if ta and ca is not None and nca is not None:
                        diff = abs(ta - (ca + nca))
                        result["asset_eq_check"] = "PASS" if diff <= 2 else f"FAIL({diff:,.0f})"
                    else:
                        result["asset_eq_check"] = "SKIP"

                return result

            elif status == 'failed':
                result["status"] = "EXTRACT_FAIL"
                result["error"] = sd.get('message', '')[:80]
                return result

        result["status"] = "TIMEOUT"
        return result

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)[:80]
        return result


def main():
    with open('/tmp/resolved_targets.json') as f:
        targets = json.load(f)

    print(f"\n{'='*80}")
    print(f"  배치 추출 테스트 v2 - {len(targets)}개 종목 (corp_code 사전 조회 완료)")
    print(f"{'='*80}")

    results = []
    counts = {}

    for i, t in enumerate(targets):
        name = t['name']
        corp_code = t['corp_code']
        stock_code = t['stock_code']
        market = t['market']

        print(f"\n[{i+1}/{len(targets)}] {name:16s} ({stock_code}) {market:6s}", end=" ... ", flush=True)

        start = time.time()
        r = test_extract(name, corp_code, stock_code, market)
        elapsed = time.time() - start

        s = r['status']
        counts[s] = counts.get(s, 0) + 1

        icon = {"SUCCESS":"✓","NO_DATA":"△","EXTRACT_FAIL":"✗","TIMEOUT":"⏱","ERROR":"✗"}.get(s,"?")
        extra = f"Balance={r['balance_check']} Asset={r['asset_eq_check']}" if s == "SUCCESS" else r.get('error','')[:40]
        print(f"{icon} {s} ({elapsed:.0f}s) {extra}")

        r["elapsed"] = round(elapsed, 1)
        results.append(r)

        # 서버 안정화 대기
        time.sleep(3)

    # 요약
    print(f"\n{'='*80}")
    print(f"  결과 요약")
    print(f"{'='*80}")
    total = len(results)
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v}개")
    print(f"  {'합계':15s}: {total}개")

    success = [r for r in results if r['status'] == 'SUCCESS']
    bal_pass = sum(1 for r in success if r['balance_check'] == 'PASS')
    bal_fail = sum(1 for r in success if 'FAIL' in r['balance_check'])
    asset_pass = sum(1 for r in success if r['asset_eq_check'] == 'PASS')
    asset_fail = sum(1 for r in success if 'FAIL' in r['asset_eq_check'])

    if success:
        print(f"\n  추출 성공 {len(success)}개 중:")
        print(f"    Balance Check: {bal_pass} PASS, {bal_fail} FAIL")
        print(f"    Asset Eq:      {asset_pass} PASS, {asset_fail} FAIL")
        print(f"    별도: {sum(1 for r in success if r['fs_type']=='별도')}개, 연결: {sum(1 for r in success if r['fs_type']=='연결')}개")

    failures = [r for r in results if r['status'] not in ('SUCCESS',)]
    if failures:
        print(f"\n  실패/이슈:")
        for r in failures:
            print(f"    {r['name']:16s} ({r['stock_code']}) → {r['status']}: {r.get('error','')}")

    with open("/home/servermanager/pers-dev/test_batch_v2_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "counts": counts, "total": total}, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: test_batch_v2_results.json")


if __name__ == "__main__":
    main()
