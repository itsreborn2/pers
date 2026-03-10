#!/usr/bin/env python3
"""Final comprehensive verification - all 15 companies with improved checks."""

import asyncio
import aiohttp
import json
import time
import sys

BASE_URL = "http://localhost:8001"
TIMEOUT = 300
CONCURRENCY = 3

COMPANIES = [
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
    {"corp_code": "00181934", "corp_name": "플랜텍", "category": "BALANCE"},
    {"corp_code": "00687085", "corp_name": "포천파워", "category": "NO_BS"},
    {"corp_code": "00652159", "corp_name": "코오롱머티리얼", "category": "NO_BS"},
    {"corp_code": "00123967", "corp_name": "부산도시가스", "category": "NO_BS"},
]

def pn(v):
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    try: return float(str(v).replace(",", ""))
    except: return None

def fs(v):
    return f"{v:,.0f}" if v is not None else "N/A"


async def test_company(session, company, sem):
    async with sem:
        name = company["corp_name"]
        code = company["corp_code"]
        cat = company["category"]
        R = {"name": name, "code": code, "cat": cat, "status": "PENDING", "err": None, "rows": 0, "time": 0, "yrs": {}}

        try:
            payload = {"corp_code": code, "corp_name": name, "start_year": 2023, "end_year": 2024}
            print(f"[START] {name} ({code})")
            async with session.post(f"{BASE_URL}/api/extract", json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    R["status"] = "EXTRACT_FAIL"; return R
                task_id = (await resp.json()).get("task_id")

            t0 = time.time()
            sd = None
            for i in range(int(TIMEOUT / 4)):
                await asyncio.sleep(4)
                if i % 3 == 0:
                    try:
                        async with session.post(f"{BASE_URL}/api/heartbeat/{task_id}", timeout=aiohttp.ClientTimeout(total=10)): pass
                    except: pass
                try:
                    async with session.get(f"{BASE_URL}/api/status/{task_id}", timeout=aiohttp.ClientTimeout(total=20)) as sr:
                        sd = await sr.json()
                        st = sd.get("status", "")
                        if st == "completed":
                            R["time"] = round(time.time() - t0, 1)
                            print(f"  [DONE] {name} in {R['time']}s")
                            break
                        elif st in ("failed", "error"):
                            R["status"] = "TASK_FAIL"; R["err"] = sd.get("message", "")[:120]; return R
                except: continue
            else:
                R["status"] = "TIMEOUT"; R["err"] = f">{TIMEOUT}s"; return R

            pd = sd.get("preview_data", {})
            vd = pd.get("vcm_display", [])
            R["rows"] = len(vd)

            if cat == "NO_BS" and len(vd) == 0:
                R["status"] = "FAIL"; R["err"] = "No BS data"; return R
            if len(vd) == 0:
                R["status"] = "NO_DATA"; return R

            # Build lookup
            lk = {}
            for row in vd:
                a = row.get("항목", "").strip()
                if a: lk[a] = row

            year_cols = [k for k in vd[0].keys() if k.startswith("FY")]
            all_pass = True

            for yc in year_cols:
                yr = yc.replace("FY", "")
                v = {}
                for key in ["자산총계", "유동자산", "비유동자산", "매각예정자산",
                           "부채총계", "유동부채", "비유동부채", "자본총계", "부채와자본총계"]:
                    r = lk.get(key)
                    v[key] = pn(r.get(yc)) if r else None

                ta = v["자산총계"]
                cu = v["유동자산"]; nc = v["비유동자산"]; da = v["매각예정자산"] or 0
                dt = v["부채총계"]; cdt = v["유동부채"]; ncdt = v["비유동부채"]
                eq = v["자본총계"]; dce = v["부채와자본총계"]

                # Fallback: compute 부채총계 from 유동부채 + 비유동부채
                if dt is None and cdt is not None and ncdt is not None:
                    dt = cdt + ncdt
                # Fallback: compute from 부채와자본총계 - 자본총계
                if dt is None and dce is not None and eq is not None:
                    dt = dce - eq

                # Check 1: Asset sum (자산총계 = 유동 + 비유동 + 매각예정)
                ac = "N/A"
                if ta is not None and cu is not None and nc is not None:
                    diff = abs(ta - cu - nc - da)
                    ac = "PASS" if diff <= 2 else f"FAIL(d={diff:.0f})"
                    if diff > 2: all_pass = False
                else:
                    ac = "MISS"
                    all_pass = False

                # Check 2: Balance (자산총계 = 부채총계 + 자본총계)
                bc = "N/A"
                if ta is not None and dt is not None and eq is not None:
                    diff = abs(ta - dt - eq)
                    bc = "PASS" if diff <= 2 else f"FAIL(d={diff:.0f})"
                    if diff > 2: all_pass = False
                elif dce is not None and ta is not None:
                    diff = abs(ta - dce)
                    bc = "PASS" if diff <= 2 else f"FAIL(d={diff:.0f})"
                    if diff > 2: all_pass = False
                else:
                    bc = "MISS"
                    all_pass = False

                R["yrs"][yr] = {
                    "ac": ac, "bc": bc,
                    "ta": ta, "cu": cu, "nc": nc, "da": da, "dt": dt, "eq": eq
                }

            if cat == "NO_BS":
                R["status"] = "PASS" if all_pass else ("PARTIAL" if R["rows"] > 0 else "FAIL")
            else:
                R["status"] = "PASS" if all_pass else "FAIL"

        except Exception as e:
            R["status"] = "ERROR"; R["err"] = str(e)[:200]

        return R


async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    conn = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=conn) as s:
        try:
            async with s.get(f"{BASE_URL}/", timeout=aiohttp.ClientTimeout(total=5)) as r:
                print(f"Server OK ({r.status})")
        except Exception as e:
            print(f"FATAL: {e}"); sys.exit(1)

        print(f"\nTesting {len(COMPANIES)} companies, concurrency={CONCURRENCY}, timeout={TIMEOUT}s\n{'='*100}")
        tasks = [test_company(s, c, sem) for c in COMPANIES]
        results = await asyncio.gather(*tasks)

    # Detailed output
    print(f"\n{'='*110}")
    print("DETAILED RESULTS")
    print(f"{'='*110}")
    for r in results:
        print(f"\n  {r['name']} ({r['code']}) [{r['cat']}] => {r['status']} ({r['time']}s, {r['rows']} rows)")
        if r.get("err"): print(f"    Error: {r['err']}")
        for yr, d in r["yrs"].items():
            print(f"    FY{yr}: Asset={fs(d['ta'])} = Current({fs(d['cu'])}) + NonCurrent({fs(d['nc'])}) + Disposal({fs(d['da'])})")
            print(f"           Debt={fs(d['dt'])} + Equity={fs(d['eq'])}")
            print(f"           Asset check: {d['ac']}  |  Balance check: {d['bc']}")

    # Summary
    print(f"\n{'='*150}")
    h = f"{'#':>2} {'Company':<28} {'Code':<10} {'Cat':<8} {'Result':<8} {'Rows':>4} {'Sec':>5} {'Asset23':<15} {'Bal23':<15} {'Asset24':<15} {'Bal24':<15}"
    print(h)
    print("-" * 150)

    pc = fc = oc = 0
    for i, r in enumerate(results, 1):
        a23 = r["yrs"].get("2023", {}).get("ac", "---")
        b23 = r["yrs"].get("2023", {}).get("bc", "---")
        a24 = r["yrs"].get("2024", {}).get("ac", "---")
        b24 = r["yrs"].get("2024", {}).get("bc", "---")
        st = r["status"]
        err_hint = ""
        if r.get("err"): err_hint = f" [{r['err'][:30]}]"
        if st == "PASS": pc += 1
        elif st in ("PARTIAL",): oc += 1
        else: fc += 1
        nm = r["name"][:26]
        print(f"{i:>2} {nm:<28} {r['code']:<10} {r['cat']:<8} {st:<8} {r['rows']:>4} {r['time']:>5.0f}s {a23:<15} {b23:<15} {a24:<15} {b24:<15}{err_hint}")

    print("-" * 150)
    print(f"TOTAL: {pc} PASS / {fc} FAIL / {oc} PARTIAL  (out of {len(results)})")

    # Failures detail
    if fc > 0:
        print(f"\nFAILURES DETAIL:")
        for r in results:
            if r["status"] not in ("PASS", "PARTIAL"):
                print(f"  {r['name']} ({r['code']}): {r['status']}")
                if r.get("err"): print(f"    => {r['err']}")
                for yr, d in r["yrs"].items():
                    if "FAIL" in d.get("ac","") or "FAIL" in d.get("bc","") or "MISS" in d.get("ac","") or "MISS" in d.get("bc",""):
                        print(f"    FY{yr}: ac={d['ac']} bc={d['bc']} | A={fs(d['ta'])} C={fs(d['cu'])} NC={fs(d['nc'])} D={fs(d['da'])} L={fs(d['dt'])} E={fs(d['eq'])}")

    sys.exit(0)

asyncio.run(main())
