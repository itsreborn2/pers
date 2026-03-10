#!/usr/bin/env python3
"""Batch verification of ~50 listed companies sorted by market cap (lowest first).
Uses server search API to resolve stock_code -> corp_code, then tests extraction."""

import asyncio
import aiohttp
import json
import time
import sys
from datetime import datetime

BASE_URL = "http://localhost:8001"
TIMEOUT = 300
CONCURRENCY = 3

# Companies to test (from pykrx, sorted by market cap ascending, KOSPI/KOSDAQ only)
COMPANIES = [
    {"stock_code": "313760", "name": "캐리", "market": "KOSDAQ", "cap_eok": 64},
    {"stock_code": "079970", "name": "투비소프트", "market": "KOSDAQ", "cap_eok": 76},
    {"stock_code": "031860", "name": "디에이치엑스컴퍼니", "market": "KOSDAQ", "cap_eok": 82},
    {"stock_code": "052770", "name": "아이톡시", "market": "KOSDAQ", "cap_eok": 88},
    {"stock_code": "044180", "name": "KD", "market": "KOSDAQ", "cap_eok": 95},
    {"stock_code": "060240", "name": "스타코링크", "market": "KOSDAQ", "cap_eok": 115},
    {"stock_code": "065570", "name": "삼영이엔씨", "market": "KOSDAQ", "cap_eok": 123},
    {"stock_code": "054180", "name": "메디콕스", "market": "KOSDAQ", "cap_eok": 127},
    {"stock_code": "050090", "name": "비케이홀딩스", "market": "KOSDAQ", "cap_eok": 130},
    {"stock_code": "188260", "name": "세니젠", "market": "KOSDAQ", "cap_eok": 132},
    {"stock_code": "244460", "name": "올리패스", "market": "KOSDAQ", "cap_eok": 134},
    {"stock_code": "058450", "name": "한주에이알티", "market": "KOSDAQ", "cap_eok": 134},
    {"stock_code": "065420", "name": "에스아이리소스", "market": "KOSDAQ", "cap_eok": 136},
    {"stock_code": "352770", "name": "셀레스트라", "market": "KOSDAQ", "cap_eok": 136},
    {"stock_code": "225430", "name": "케이엠제약", "market": "KOSDAQ", "cap_eok": 137},
    {"stock_code": "140910", "name": "에이리츠", "market": "KOSPI", "cap_eok": 140},
    {"stock_code": "900120", "name": "씨엑스아이", "market": "KOSDAQ", "cap_eok": 143},
    {"stock_code": "060260", "name": "뉴보텍", "market": "KOSDAQ", "cap_eok": 145},
    {"stock_code": "121850", "name": "코이즈", "market": "KOSDAQ", "cap_eok": 147},
    {"stock_code": "227100", "name": "프로브잇", "market": "KOSDAQ", "cap_eok": 148},
    {"stock_code": "065770", "name": "CS", "market": "KOSDAQ", "cap_eok": 151},
    {"stock_code": "008290", "name": "원풍물산", "market": "KOSDAQ", "cap_eok": 151},
    {"stock_code": "026910", "name": "광진실업", "market": "KOSDAQ", "cap_eok": 151},
    {"stock_code": "030350", "name": "드래곤플라이", "market": "KOSDAQ", "cap_eok": 152},
    {"stock_code": "069330", "name": "유아이디", "market": "KOSDAQ", "cap_eok": 153},
    {"stock_code": "035290", "name": "골드앤에스", "market": "KOSDAQ", "cap_eok": 156},
    {"stock_code": "250930", "name": "예선테크", "market": "KOSDAQ", "cap_eok": 156},
    {"stock_code": "106080", "name": "케이이엠텍", "market": "KOSDAQ", "cap_eok": 157},
    {"stock_code": "106520", "name": "노블엠앤비", "market": "KOSDAQ", "cap_eok": 159},
    {"stock_code": "027040", "name": "서울전자통신", "market": "KOSDAQ", "cap_eok": 161},
    {"stock_code": "215790", "name": "이노인스트루먼트", "market": "KOSDAQ", "cap_eok": 163},
    {"stock_code": "368970", "name": "오에스피", "market": "KOSDAQ", "cap_eok": 164},
    {"stock_code": "065060", "name": "지엔코", "market": "KOSDAQ", "cap_eok": 164},
    {"stock_code": "275630", "name": "에스에스알", "market": "KOSDAQ", "cap_eok": 166},
    {"stock_code": "241820", "name": "피씨엘", "market": "KOSDAQ", "cap_eok": 168},
    {"stock_code": "073190", "name": "듀오백", "market": "KOSDAQ", "cap_eok": 169},
    {"stock_code": "322780", "name": "코퍼스코리아", "market": "KOSDAQ", "cap_eok": 171},
    {"stock_code": "304840", "name": "피플바이오", "market": "KOSDAQ", "cap_eok": 173},
    {"stock_code": "032800", "name": "판타지오", "market": "KOSDAQ", "cap_eok": 174},
    {"stock_code": "145210", "name": "다이나믹디자인", "market": "KOSPI", "cap_eok": 175},
    {"stock_code": "067010", "name": "에스코넥", "market": "KOSDAQ", "cap_eok": 176},
    {"stock_code": "003380", "name": "하림지주", "market": "KOSPI", "cap_eok": 177},
    {"stock_code": "170920", "name": "엘티씨", "market": "KOSDAQ", "cap_eok": 177},
    {"stock_code": "039740", "name": "한국정보공학", "market": "KOSDAQ", "cap_eok": 178},
    {"stock_code": "310870", "name": "디와이씨", "market": "KOSDAQ", "cap_eok": 178},
    {"stock_code": "226440", "name": "한국IT전문학교", "market": "KOSDAQ", "cap_eok": 179},
    {"stock_code": "086040", "name": "바이오노트", "market": "KOSDAQ", "cap_eok": 180},
    {"stock_code": "041020", "name": "폴라리스오피스", "market": "KOSDAQ", "cap_eok": 181},
    {"stock_code": "064260", "name": "다이얼로그", "market": "KOSDAQ", "cap_eok": 183},
    {"stock_code": "083640", "name": "인콘", "market": "KOSDAQ", "cap_eok": 184},
]


def pn(v):
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    try: return float(str(v).replace(",", ""))
    except: return None

def fs(v):
    return f"{v:,.0f}" if v is not None else "N/A"


async def resolve_corp_code(session, company):
    """Use server search API to resolve stock_code -> corp_code."""
    try:
        payload = {"company_name": company["name"], "market": "YKN"}
        async with session.post(f"{BASE_URL}/api/search", json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            results = data.get("data", [])
            for r in results:
                if r.get("stock_code") == company["stock_code"]:
                    return r["corp_code"]
            # If exact stock_code match not found, try first result
            if results and len(results) == 1:
                return results[0]["corp_code"]
    except Exception as e:
        print(f"  Search error for {company['name']}: {e}")
    return None


async def test_company(session, company, sem):
    async with sem:
        name = company["name"]
        stock_code = company["stock_code"]
        R = {"name": name, "stock_code": stock_code, "market": company["market"],
             "cap_eok": company["cap_eok"], "corp_code": "",
             "status": "PENDING", "err": None, "rows": 0, "time": 0,
             "fs_type": "", "yrs": {}, "note": ""}

        try:
            # Step 1: Resolve corp_code
            corp_code = await resolve_corp_code(session, company)
            if not corp_code:
                R["status"] = "RESOLVE_FAIL"
                R["err"] = f"Cannot find corp_code for {stock_code}"
                print(f"[SKIP] {name} ({stock_code}) - corp_code not found")
                return R

            R["corp_code"] = corp_code

            # Step 2: Start extraction
            payload = {"corp_code": corp_code, "corp_name": name, "start_year": 2023, "end_year": 2024}
            print(f"[START] {name} ({corp_code}, {stock_code})")
            async with session.post(f"{BASE_URL}/api/extract", json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    R["status"] = "EXTRACT_FAIL"
                    R["err"] = f"HTTP {resp.status}: {body[:100]}"
                    return R
                resp_data = await resp.json()
                task_id = resp_data.get("task_id")

            t0 = time.time()
            sd = None
            for i in range(int(TIMEOUT / 4)):
                await asyncio.sleep(4)
                if i % 3 == 0:
                    try:
                        async with session.post(f"{BASE_URL}/api/heartbeat/{task_id}", timeout=aiohttp.ClientTimeout(total=30)): pass
                    except: pass
                try:
                    async with session.get(f"{BASE_URL}/api/status/{task_id}", timeout=aiohttp.ClientTimeout(total=60)) as sr:
                        sd = await sr.json()
                        st = sd.get("status", "")
                        if st == "completed":
                            R["time"] = round(time.time() - t0, 1)
                            print(f"  [DONE] {name} in {R['time']}s")
                            break
                        elif st in ("failed", "error"):
                            R["status"] = "TASK_FAIL"
                            R["err"] = sd.get("message", "")[:120]
                            return R
                except: continue
            else:
                R["status"] = "TIMEOUT"
                R["err"] = f">{TIMEOUT}s"
                return R

            pd_data = sd.get("preview_data", {})
            vd = pd_data.get("vcm_display", [])
            R["rows"] = len(vd)

            # Detect fs_type from vcm_display structure or meta
            meta = pd_data.get("meta", {})
            R["fs_type"] = meta.get("fs_type", "")
            if not R["fs_type"]:
                # Try to infer from data
                for row in vd:
                    item = row.get("항목", "")
                    if "연결" in str(sd.get("preview_data", {}).get("sheets", [])):
                        R["fs_type"] = "연결"
                        break
                if not R["fs_type"]:
                    R["fs_type"] = "별도"

            if len(vd) == 0:
                R["status"] = "NO_DATA"
                return R

            # Build lookup
            lk = {}
            for row in vd:
                a = row.get("항목", "").strip()
                if a: lk[a] = row

            year_cols = [k for k in vd[0].keys() if k.startswith("FY")]
            all_pass = True
            note_parts = []

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

                if dt is None and cdt is not None and ncdt is not None:
                    dt = cdt + ncdt
                if dt is None and dce is not None and eq is not None:
                    dt = dce - eq

                ac = "N/A"
                if ta is not None and cu is not None and nc is not None:
                    diff = abs(ta - cu - nc - da)
                    ac = "PASS" if diff <= 2 else f"FAIL(d={diff:.0f})"
                    if diff > 2: all_pass = False
                else:
                    ac = "MISS"

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

                capital_impaired = eq is not None and eq < 0
                if capital_impaired:
                    note_parts.append(f"자본잠식(FY{yr}:{eq:,.0f}M)")

                R["yrs"][yr] = {
                    "ac": ac, "bc": bc,
                    "ta": ta, "cu": cu, "nc": nc, "da": da, "dt": dt, "eq": eq,
                    "capital_impaired": capital_impaired
                }

            R["note"] = "; ".join(note_parts)
            R["status"] = "PASS" if all_pass else "FAIL"

        except Exception as e:
            R["status"] = "ERROR"
            R["err"] = str(e)[:200]

        return R


async def main():
    print(f"{'='*80}")
    print(f"Batch verification: {len(COMPANIES)} companies, concurrency={CONCURRENCY}")
    print(f"{'='*80}")

    sem = asyncio.Semaphore(CONCURRENCY)
    conn = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=conn) as s:
        try:
            async with s.get(f"{BASE_URL}/", timeout=aiohttp.ClientTimeout(total=30)) as r:
                print(f"Server OK ({r.status})")
        except Exception as e:
            print(f"FATAL: {e}"); sys.exit(1)

        tasks = [test_company(s, c, sem) for c in COMPANIES]
        results = await asyncio.gather(*tasks)

    # Summary table
    print(f"\n{'='*170}")
    h = f"{'#':>2} {'Company':<22} {'StockCode':<10} {'CorpCode':<10} {'Mkt':<7} {'시총':>6} {'Result':<12} {'FS':>4} {'Rows':>4} {'Sec':>5} {'A23':<10} {'B23':<10} {'A24':<10} {'B24':<10} {'Note'}"
    print(h)
    print("-" * 170)

    pc = fc = ndc = ec = rfc = tfc = 0
    for i, r in enumerate(results, 1):
        a23 = r["yrs"].get("2023", {}).get("ac", "---")
        b23 = r["yrs"].get("2023", {}).get("bc", "---")
        a24 = r["yrs"].get("2024", {}).get("ac", "---")
        b24 = r["yrs"].get("2024", {}).get("bc", "---")
        st = r["status"]
        note = r.get("note", "")
        if r.get("err"): note = (note + " " + r["err"][:50]).strip()

        if st == "PASS": pc += 1
        elif st == "NO_DATA": ndc += 1
        elif st == "FAIL": fc += 1
        elif st == "RESOLVE_FAIL": rfc += 1
        elif st == "TASK_FAIL": tfc += 1
        else: ec += 1

        fs_label = r.get("fs_type", "")[:4]
        nm = r["name"][:20]
        print(f"{i:>2} {nm:<22} {r['stock_code']:<10} {r['corp_code']:<10} {r['market']:<7} {r['cap_eok']:>6} {st:<12} {fs_label:>4} {r['rows']:>4} {r['time']:>5.0f}s {a23:<10} {b23:<10} {a24:<10} {b24:<10} {note}")

    print("-" * 170)
    total = len(results)
    print(f"TOTAL: {pc} PASS / {fc} FAIL / {ndc} NO_DATA / {tfc} TASK_FAIL / {rfc} RESOLVE_FAIL / {ec} ERROR  (out of {total})")
    if total > 0:
        tested = total - rfc
        if tested > 0:
            print(f"Success rate (tested): {pc}/{tested} = {pc/tested*100:.1f}%")

    # Save results
    output = {
        "test_date": datetime.now().strftime("%Y-%m-%d"),
        "total": total,
        "pass_count": pc,
        "fail_count": fc,
        "no_data": ndc,
        "task_fail": tfc,
        "resolve_fail": rfc,
        "error": ec,
        "results": []
    }
    for r in results:
        entry = {
            "name": r["name"],
            "stock_code": r["stock_code"],
            "corp_code": r["corp_code"],
            "market": r["market"],
            "cap_eok": r["cap_eok"],
            "status": r["status"],
            "fs_type": r["fs_type"],
            "rows": r["rows"],
            "time": r["time"],
            "err": r["err"],
            "note": r.get("note", ""),
            "years": {}
        }
        for yr, d in r["yrs"].items():
            entry["years"][yr] = {
                "asset_check": d["ac"],
                "balance_check": d["bc"],
                "capital_impaired": d.get("capital_impaired", False),
                "equity": d.get("eq")
            }
        output["results"].append(entry)

    with open("/home/servermanager/pers-dev/verify_batch_50_results.json", "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to verify_batch_50_results.json")

    # Failures detail
    fail_results = [r for r in results if r["status"] == "FAIL"]
    if fail_results:
        print(f"\nFAILURES DETAIL:")
        for r in fail_results:
            print(f"\n  {r['name']} ({r['stock_code']}) [{r['market']}]:")
            for yr, d in r["yrs"].items():
                if "FAIL" in d.get("ac","") or "FAIL" in d.get("bc",""):
                    print(f"    FY{yr}: ac={d['ac']} bc={d['bc']} | A={fs(d['ta'])} C={fs(d['cu'])} NC={fs(d['nc'])} D={fs(d['da'])} L={fs(d['dt'])} E={fs(d['eq'])}")


asyncio.run(main())
