#!/usr/bin/env python3
"""Retry timed-out companies with longer timeout."""

import asyncio
import aiohttp
import json
import time
import sys

BASE_URL = "http://localhost:8001"
TIMEOUT = 360  # 6 minutes
CONCURRENCY = 2  # Lower concurrency for big companies

COMPANIES = [
    {"corp_code": "00215976", "corp_name": "무궁화인포메이션테크놀로지", "category": "ASSET"},
    {"corp_code": "00683283", "corp_name": "LS전선", "category": "ASSET"},
    {"corp_code": "00149266", "corp_name": "씨앤에이치", "category": "ASSET"},
    {"corp_code": "00852175", "corp_name": "SK에너지", "category": "ASSET"},
]


def parse_num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", ""))
    except (ValueError, TypeError):
        return None


async def extract_and_poll(session, company, sem):
    async with sem:
        name = company["corp_name"]
        code = company["corp_code"]
        result = {"name": name, "code": code, "overall": "PENDING", "error": None, "vcm_rows": 0, "elapsed": 0, "year_results": {}}

        try:
            payload = {"corp_code": code, "corp_name": name, "start_year": 2023, "end_year": 2024}
            print(f"[START] {name} ({code})")
            async with session.post(f"{BASE_URL}/api/extract", json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    result["overall"] = "EXTRACT_FAIL"
                    return result
                data = await resp.json()
                task_id = data.get("task_id")

            start_time = time.time()
            status_data = None
            poll_count = 0
            while time.time() - start_time < TIMEOUT:
                await asyncio.sleep(5)
                poll_count += 1
                if poll_count % 3 == 0:
                    try:
                        async with session.post(f"{BASE_URL}/api/heartbeat/{task_id}", timeout=aiohttp.ClientTimeout(total=10)):
                            pass
                    except:
                        pass

                try:
                    async with session.get(f"{BASE_URL}/api/status/{task_id}", timeout=aiohttp.ClientTimeout(total=15)) as sr:
                        if sr.status != 200:
                            continue
                        status_data = await sr.json()
                        ts = status_data.get("status", "")
                        prog = status_data.get("progress", 0)
                        msg = status_data.get("message", "")
                        if poll_count % 6 == 0:
                            print(f"  [{name[:10]}] {ts} {prog}% - {msg[:60]}")
                        if ts == "completed":
                            result["elapsed"] = round(time.time() - start_time, 1)
                            print(f"  [DONE] {name} in {result['elapsed']}s")
                            break
                        elif ts in ("failed", "error"):
                            result["overall"] = "TASK_FAIL"
                            result["error"] = msg[:120]
                            return result
                except:
                    continue
            else:
                result["overall"] = "TIMEOUT"
                result["error"] = f">{TIMEOUT}s"
                return result

            # Analyze
            pd = status_data.get("preview_data", {})
            vd = pd.get("vcm_display", [])
            result["vcm_rows"] = len(vd)
            if len(vd) == 0:
                result["overall"] = "NO_DATA"
                return result

            lookup = {}
            for row in vd:
                acct = row.get("항목", "").strip()
                if acct:
                    lookup[acct] = row

            year_cols = [k for k in vd[0].keys() if k.startswith("FY")]
            all_pass = True

            for ycol in year_cols:
                yr = ycol.replace("FY", "")
                vals = {}
                for key in ["자산총계", "유동자산", "비유동자산", "매각예정자산", "부채총계", "자본총계"]:
                    row = lookup.get(key)
                    vals[key] = parse_num(row.get(ycol)) if row else None

                ta = vals["자산총계"]
                cu = vals["유동자산"]
                nc = vals["비유동자산"]
                da = vals["매각예정자산"] or 0
                de = vals["부채총계"]
                eq = vals["자본총계"]

                asset_check = "N/A"
                if ta is not None and cu is not None and nc is not None:
                    diff = abs(ta - cu - nc - da)
                    asset_check = "PASS" if diff <= 2 else f"FAIL(d={diff:.0f})"
                    if diff > 2:
                        all_pass = False

                balance_check = "N/A"
                if ta is not None and de is not None and eq is not None:
                    diff = abs(ta - de - eq)
                    balance_check = "PASS" if diff <= 2 else f"FAIL(d={diff:.0f})"
                    if diff > 2:
                        all_pass = False

                ta_s = f"{ta:,.0f}" if ta else "N/A"
                cu_s = f"{cu:,.0f}" if cu else "N/A"
                nc_s = f"{nc:,.0f}" if nc else "N/A"
                da_s = f"{da:,.0f}" if da else "-"
                de_s = f"{de:,.0f}" if de else "N/A"
                eq_s = f"{eq:,.0f}" if eq else "N/A"

                print(f"  FY{yr}: A={ta_s} | C={cu_s}+NC={nc_s}+D={da_s} | L={de_s}+E={eq_s}")
                print(f"         Asset: {asset_check} | Balance: {balance_check}")
                result["year_results"][yr] = {"asset": asset_check, "balance": balance_check}

            result["overall"] = "PASS" if all_pass else "FAIL"

        except Exception as e:
            result["overall"] = "ERROR"
            result["error"] = str(e)[:200]

        return result


async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"Retrying {len(COMPANIES)} timed-out companies, timeout={TIMEOUT}s, concurrency={CONCURRENCY}")
        print("=" * 80)
        tasks = [extract_and_poll(session, c, sem) for c in COMPANIES]
        results = await asyncio.gather(*tasks)

    print("\n" + "=" * 80)
    print("RETRY RESULTS")
    print("=" * 80)
    for r in results:
        a23 = r["year_results"].get("2023", {}).get("asset", "---")
        b23 = r["year_results"].get("2023", {}).get("balance", "---")
        a24 = r["year_results"].get("2024", {}).get("asset", "---")
        b24 = r["year_results"].get("2024", {}).get("balance", "---")
        print(f"  {r['name']:<25} {r['overall']:<8} VCM={r['vcm_rows']:>3} {r['elapsed']:>5.0f}s | A23={a23:<14} B23={b23:<14} A24={a24:<14} B24={b24}")
        if r.get("error"):
            print(f"    Error: {r['error']}")


if __name__ == "__main__":
    asyncio.run(main())
