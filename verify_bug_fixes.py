#!/usr/bin/env python3
"""Comprehensive verification test for 15 companies with data issues."""

import asyncio
import aiohttp
import json
import time
import sys

BASE_URL = "http://localhost:8001"
TIMEOUT = 180  # seconds per company
CONCURRENCY = 3

COMPANIES = [
    # ASSET mismatch (BUG 1 + BUG 2+5)
    {"corp_code": "01168143", "corp_name": "케이만금세기차륜집단유한공사", "category": "ASSET"},
    {"corp_code": "00153418", "corp_name": "티케이지태광", "category": "ASSET"},
    {"corp_code": "00495554", "corp_name": "와이앤넥스트", "category": "ASSET"},
    {"corp_code": "00857231", "corp_name": "비에스에셋매니지먼트", "category": "ASSET"},
    {"corp_code": "00215976", "corp_name": "무궁화인포메이션테크놀로지", "category": "ASSET"},
    {"corp_code": "00907323", "corp_name": "자일자동차", "category": "ASSET"},
    {"corp_code": "00683283", "corp_name": "LS전선", "category": "ASSET"},
    {"corp_code": "00149266", "corp_name": "씨앤에이치", "category": "ASSET"},
    {"corp_code": "00852175", "corp_name": "SK에너지", "category": "ASSET"},
    {"corp_code": "01212921", "corp_name": "비바리퍼블리카", "category": "ASSET"},
    {"corp_code": "00136642", "corp_name": "에스유앤피", "category": "ASSET"},
    # BALANCE mismatch (BUG 1)
    {"corp_code": "00181934", "corp_name": "플랜텍", "category": "BALANCE"},
    # NO_BS/NO_VCM (BUG 6)
    {"corp_code": "00687085", "corp_name": "포천파워", "category": "NO_BS"},
    {"corp_code": "00652159", "corp_name": "코오롱머티리얼", "category": "NO_BS"},
    {"corp_code": "00123967", "corp_name": "부산도시가스", "category": "NO_BS"},
]


def parse_num(v):
    """Parse a numeric value from vcm_display."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", ""))
    except (ValueError, TypeError):
        return None


async def extract_and_poll(session, company, sem):
    """Extract data for a company and poll until complete."""
    async with sem:
        name = company["corp_name"]
        code = company["corp_code"]
        category = company["category"]
        result = {
            "name": name,
            "code": code,
            "category": category,
            "overall": "PENDING",
            "error": None,
            "vcm_rows": 0,
            "years": [],
            "year_results": {},
            "elapsed": 0,
        }

        try:
            # Step 1: POST extract
            payload = {
                "corp_code": code,
                "corp_name": name,
                "start_year": 2023,
                "end_year": 2024,
            }
            print(f"[START] {name} ({code}) [{category}]")
            async with session.post(
                f"{BASE_URL}/api/extract", json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    result["overall"] = "EXTRACT_FAIL"
                    result["error"] = f"HTTP {resp.status}"
                    print(f"  [ERROR] Extract failed HTTP {resp.status}")
                    return result
                data = await resp.json()
                task_id = data.get("task_id")
                if not task_id:
                    result["overall"] = "NO_TASK_ID"
                    result["error"] = "No task_id"
                    return result

            # Step 2: Poll status + heartbeat
            start_time = time.time()
            poll_count = 0
            status_data = None
            while time.time() - start_time < TIMEOUT:
                await asyncio.sleep(4)
                poll_count += 1

                # Send heartbeat every 3rd poll
                if poll_count % 3 == 0:
                    try:
                        async with session.post(
                            f"{BASE_URL}/api/heartbeat/{task_id}",
                            timeout=aiohttp.ClientTimeout(total=10),
                        ):
                            pass
                    except Exception:
                        pass

                try:
                    async with session.get(
                        f"{BASE_URL}/api/status/{task_id}",
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as status_resp:
                        if status_resp.status != 200:
                            continue
                        status_data = await status_resp.json()
                        task_status = status_data.get("status", "")

                        if task_status == "completed":
                            elapsed = time.time() - start_time
                            result["elapsed"] = round(elapsed, 1)
                            print(f"  [DONE] {name} in {elapsed:.0f}s")
                            break
                        elif task_status in ("failed", "error", "cancelled"):
                            result["overall"] = "TASK_FAIL"
                            result["error"] = status_data.get("message", "Unknown")[:120]
                            print(f"  [FAIL] {name}: {result['error']}")
                            return result
                except Exception as e:
                    continue
            else:
                result["overall"] = "TIMEOUT"
                result["error"] = f">{TIMEOUT}s"
                print(f"  [TIMEOUT] {name}")
                return result

            # Step 3: Get vcm_display from preview_data
            preview_data = status_data.get("preview_data", {})
            vcm_display = preview_data.get("vcm_display", [])
            result["vcm_rows"] = len(vcm_display)

            if category == "NO_BS" and len(vcm_display) == 0:
                result["overall"] = "FAIL"
                result["error"] = "Still no BS data (0 vcm rows)"
                return result

            if len(vcm_display) == 0:
                result["overall"] = "NO_DATA"
                result["error"] = "vcm_display empty"
                return result

            # Step 4: Determine year columns
            sample_keys = list(vcm_display[0].keys())
            year_cols = [k for k in sample_keys if k.startswith("FY")]
            result["years"] = year_cols

            # Step 5: Build lookup by account name
            lookup = {}
            for row in vcm_display:
                acct = row.get("항목", "").strip()
                if acct:
                    lookup[acct] = row

            # Step 6: Validate for each year
            all_pass = True
            for ycol in year_cols:
                yr = ycol.replace("FY", "")
                yr_result = {
                    "자산총계": None,
                    "유동자산": None,
                    "비유동자산": None,
                    "매각예정자산": None,
                    "부채총계": None,
                    "자본총계": None,
                    "매각예정부채": None,
                    "asset_check": "N/A",
                    "balance_check": "N/A",
                }

                # Extract values
                for key in yr_result.keys():
                    if key.endswith("_check"):
                        continue
                    row = lookup.get(key)
                    if row:
                        yr_result[key] = parse_num(row.get(ycol))

                total_asset = yr_result["자산총계"]
                current = yr_result["유동자산"]
                noncurrent = yr_result["비유동자산"]
                disposal_asset = yr_result["매각예정자산"] or 0
                total_debt = yr_result["부채총계"]
                total_equity = yr_result["자본총계"]

                # Check 1: 자산총계 ~= 유동자산 + 비유동자산 + 매각예정자산 (tolerance ±2 for rounding)
                if total_asset is not None and current is not None and noncurrent is not None:
                    expected = current + noncurrent + disposal_asset
                    diff = abs(total_asset - expected)
                    if diff <= 2:
                        yr_result["asset_check"] = "PASS"
                    else:
                        yr_result["asset_check"] = f"FAIL(d={diff:.0f})"
                        all_pass = False
                else:
                    missing = []
                    if total_asset is None: missing.append("자산총계")
                    if current is None: missing.append("유동자산")
                    if noncurrent is None: missing.append("비유동자산")
                    yr_result["asset_check"] = f"MISS({','.join(missing)})"
                    all_pass = False

                # Check 2: 자산총계 ~= 부채총계 + 자본총계 (tolerance ±2)
                if total_asset is not None and total_debt is not None and total_equity is not None:
                    expected = total_debt + total_equity
                    diff = abs(total_asset - expected)
                    if diff <= 2:
                        yr_result["balance_check"] = "PASS"
                    else:
                        yr_result["balance_check"] = f"FAIL(d={diff:.0f})"
                        all_pass = False
                else:
                    missing = []
                    if total_asset is None: missing.append("자산총계")
                    if total_debt is None: missing.append("부채총계")
                    if total_equity is None: missing.append("자본총계")
                    yr_result["balance_check"] = f"MISS({','.join(missing)})"
                    all_pass = False

                result["year_results"][yr] = yr_result

            # For NO_BS, having data at all is a pass even if equations are off
            if category == "NO_BS":
                if len(vcm_display) > 0:
                    result["overall"] = "PASS" if all_pass else "PARTIAL"
                else:
                    result["overall"] = "FAIL"
            else:
                result["overall"] = "PASS" if all_pass else "FAIL"

        except Exception as e:
            result["overall"] = "ERROR"
            result["error"] = str(e)[:200]
            print(f"  [ERROR] {name}: {e}")

        return result


async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Verify server is up
        try:
            async with session.get(f"{BASE_URL}/", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                print(f"Server is up at {BASE_URL} (HTTP {resp.status})")
        except Exception as e:
            print(f"FATAL: Cannot reach server: {e}")
            sys.exit(1)

        print(f"\nTesting {len(COMPANIES)} companies, {CONCURRENCY} concurrent, timeout {TIMEOUT}s each")
        print("=" * 100)

        tasks = [extract_and_poll(session, c, sem) for c in COMPANIES]
        results = await asyncio.gather(*tasks)

    # Print detailed results
    print("\n" + "=" * 100)
    print("DETAILED RESULTS")
    print("=" * 100)

    for r in results:
        print(f"\n{'='*80}")
        print(f"  {r['name']} ({r['code']}) [{r['category']}] => {r['overall']}  ({r['elapsed']}s)")
        if r.get("error"):
            print(f"  Error: {r['error']}")
        print(f"  VCM rows: {r['vcm_rows']}, Years: {r['years']}")

        for yr, yr_data in r.get("year_results", {}).items():
            total_a = yr_data['자산총계']
            curr = yr_data['유동자산']
            noncurr = yr_data['비유동자산']
            disp = yr_data['매각예정자산']
            debt = yr_data['부채총계']
            equity = yr_data['자본총계']

            ta_s = f"{total_a:,.0f}" if total_a is not None else "N/A"
            cu_s = f"{curr:,.0f}" if curr is not None else "N/A"
            nc_s = f"{noncurr:,.0f}" if noncurr is not None else "N/A"
            di_s = f"{disp:,.0f}" if disp is not None else "-"
            de_s = f"{debt:,.0f}" if debt is not None else "N/A"
            eq_s = f"{equity:,.0f}" if equity is not None else "N/A"

            print(f"  FY{yr}: 자산={ta_s} | 유동={cu_s} + 비유동={nc_s} + 매각={di_s} | 부채={de_s} + 자본={eq_s}")
            print(f"         Asset check: {yr_data['asset_check']}  |  Balance check: {yr_data['balance_check']}")

    # Summary table
    print("\n" + "=" * 130)
    print("SUMMARY TABLE")
    print("=" * 130)
    hdr = f"{'#':>2} {'Company':<25} {'Code':<10} {'Cat':<8} {'Result':<8} {'VCM':>4} {'Time':>5}"
    hdr += f" {'Asset2023':<14} {'Balance2023':<14} {'Asset2024':<14} {'Balance2024':<14}"
    print(hdr)
    print("-" * 130)

    pass_c = fail_c = other_c = 0
    for i, r in enumerate(results, 1):
        short_name = r["name"][:23]
        status = r["overall"]

        a23 = b23 = a24 = b24 = "---"
        for yr, yd in r.get("year_results", {}).items():
            if yr == "2023":
                a23 = yd["asset_check"]
                b23 = yd["balance_check"]
            elif yr == "2024":
                a24 = yd["asset_check"]
                b24 = yd["balance_check"]

        # For NO_BS with data, show bs_exists
        if r["category"] == "NO_BS" and r["vcm_rows"] > 0 and a23 == "---":
            a23 = f"rows={r['vcm_rows']}"

        row = f"{i:>2} {short_name:<25} {r['code']:<10} {r['category']:<8} {status:<8} {r['vcm_rows']:>4} {r['elapsed']:>5.0f}s"
        row += f" {a23:<14} {b23:<14} {a24:<14} {b24:<14}"
        print(row)

        if status == "PASS":
            pass_c += 1
        elif status in ("FAIL", "TASK_FAIL", "EXTRACT_FAIL", "TIMEOUT", "ERROR", "NO_DATA"):
            fail_c += 1
        else:
            other_c += 1

    print("-" * 130)
    print(f"TOTAL: {pass_c} PASS / {fail_c} FAIL / {other_c} OTHER  (out of {len(results)})")
    print("=" * 130)

    if fail_c > 0:
        print("\nFAILED COMPANIES:")
        for r in results:
            if r["overall"] in ("FAIL", "TASK_FAIL", "EXTRACT_FAIL", "TIMEOUT", "ERROR", "NO_DATA"):
                print(f"  - {r['name']} ({r['code']}): {r['overall']} {r.get('error','')}")

    sys.exit(0 if fail_c == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
