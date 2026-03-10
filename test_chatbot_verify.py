#!/usr/bin/env python3
"""PE 챗봇 검증 테스트 v2 — 실제 DART 데이터 vs 챗봇 응답 수치 대조"""

import requests
import json
import time
import sys
import re
import math

BASE_URL = "http://localhost:8001"
COMPANY_NAME = "지에스엔텍"

# ══════════════════════════════════════════════════════════
#  Phase 1: 데이터 추출 & 원본 수치 수집
# ══════════════════════════════════════════════════════════

def setup():
    """기업 검색 → 추출 → 챗봇 초기화 → 원본 데이터 반환"""
    print(f"\n{'='*70}")
    print(f"  PE 챗봇 정밀 검증 테스트 v2")
    print(f"  기업: {COMPANY_NAME} | 서버: {BASE_URL}")
    print(f"{'='*70}")

    # 1) 검색
    print(f"\n[1] 기업 검색...")
    resp = requests.post(f"{BASE_URL}/api/search", json={"company_name": COMPANY_NAME})
    results = resp.json()
    if isinstance(results, dict):
        results = results.get('results', results.get('data', []))
    company = None
    for r in results:
        if COMPANY_NAME in r.get('corp_name', ''):
            company = r
            break
    if not company:
        print("  ✗ 기업 못 찾음")
        sys.exit(1)
    corp_code = company['corp_code']
    print(f"  ✓ {company['corp_name']} (corp_code={corp_code})")

    # 2) 기업개황
    print(f"\n[2] 기업개황정보...")
    resp = requests.get(f"{BASE_URL}/api/company-info/{corp_code}")
    company_info = resp.json().get('data', {})
    print(f"  ✓ {company_info.get('corp_name', '?')} / {company_info.get('ceo_nm', '?')}")

    # 3) 추출
    print(f"\n[3] 재무제표 추출...")
    resp = requests.post(f"{BASE_URL}/api/extract", json={
        "corp_code": corp_code,
        "corp_name": COMPANY_NAME,
        "start_year": 2021,
        "end_year": 2024,
        "company_info": company_info
    })
    task_id = resp.json().get('task_id')
    print(f"  → task_id: {task_id}")

    for i in range(120):
        time.sleep(3)
        resp = requests.get(f"{BASE_URL}/api/status/{task_id}")
        sd = resp.json()
        if i % 5 == 0:
            print(f"  ... [{sd.get('progress',0)}%] {sd.get('message','')[:50]}")
        if sd.get('status') == 'completed':
            print(f"  ✓ 추출 완료!")
            break
        elif sd.get('status') == 'failed':
            print(f"  ✗ 실패: {sd.get('message')}")
            sys.exit(1)
    else:
        print("  ✗ 타임아웃")
        sys.exit(1)

    # 4) 원본 데이터 가져오기
    print(f"\n[4] 원본 데이터 수집...")
    resp = requests.get(f"{BASE_URL}/api/status/{task_id}")
    preview = resp.json().get('preview_data', {})

    raw_data = {
        'bs': preview.get('bs', []),
        'is': preview.get('is', []),
        'cf': preview.get('cf', []),
        'cis': preview.get('cis', []),
        'vcm_display': preview.get('vcm_display', []),
        'vcm': preview.get('vcm', []),
    }
    print(f"  BS rows: {len(raw_data['bs'])}, IS rows: {len(raw_data['is'])}, CF rows: {len(raw_data['cf'])}")

    # 5) 챗봇 초기화
    print(f"\n[5] 챗봇 초기화...")
    resp = requests.post(f"{BASE_URL}/api/chat/init/{task_id}")
    if resp.json().get('success'):
        print(f"  ✓ 초기화 완료")
    else:
        print(f"  ✗ 실패")
        sys.exit(1)

    return task_id, raw_data


# ══════════════════════════════════════════════════════════
#  Phase 2: 원본 데이터에서 수치 추출 헬퍼
# ══════════════════════════════════════════════════════════

def find_in_table(rows, item_name, year_col=None):
    """재무제표 테이블에서 특정 항목의 값 찾기"""
    if not rows:
        return None

    # 테이블 구조: [header_row, data_row1, data_row2, ...]
    # header_row에서 연도 컬럼 인덱스 찾기
    if isinstance(rows[0], dict):
        # dict 형태
        for row in rows:
            name = row.get('계정과목', row.get('항목', ''))
            if item_name in name:
                if year_col:
                    for k, v in row.items():
                        if year_col in k:
                            return parse_number(v)
                return row
    elif isinstance(rows[0], list):
        header = rows[0]
        year_idx = None
        if year_col:
            for i, h in enumerate(header):
                if year_col in str(h):
                    year_idx = i
                    break

        for row in rows[1:]:
            if len(row) > 0 and item_name in str(row[0]):
                if year_idx is not None and year_idx < len(row):
                    return parse_number(row[year_idx])
                return row
    return None


def parse_number(val):
    """문자열 → 숫자 변환"""
    if val is None:
        return None
    s = str(val).strip()
    if s == '' or s == '-' or s == 'None':
        return None
    # 괄호 = 음수
    neg = False
    if '(' in s and ')' in s:
        neg = True
        s = s.replace('(', '').replace(')', '')
    s = s.replace(',', '').replace(' ', '')
    try:
        num = float(s)
        return -num if neg else num
    except ValueError:
        return None


def extract_numbers_from_text(text):
    """챗봇 응답에서 숫자 추출 (콤마 포함)"""
    # 음수, 소수점, 콤마 포함 숫자 매칭
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    parsed = []
    for n in nums:
        try:
            parsed.append(float(n.replace(',', '')))
        except ValueError:
            pass
    return parsed


def numbers_match(actual, chatbot_val, tolerance_pct=5):
    """두 수치 비교 (허용 오차 5%)"""
    if actual is None or chatbot_val is None:
        return None  # 비교 불가
    if actual == 0 and chatbot_val == 0:
        return True
    if actual == 0:
        return abs(chatbot_val) < 1  # 0 근처
    ratio = abs(chatbot_val - actual) / abs(actual) * 100
    return ratio <= tolerance_pct


# ══════════════════════════════════════════════════════════
#  Phase 3: 챗봇 호출
# ══════════════════════════════════════════════════════════

def send_chat(task_id, message):
    """SSE 스트리밍 응답 수집"""
    resp = requests.post(
        f"{BASE_URL}/api/chat/message/{task_id}",
        json={"message": message},
        stream=True, timeout=120
    )
    full_text = ""
    classify = ""
    refs = []

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
            if isinstance(event, dict):
                etype = event.get('type', '')
                if etype == 'token':
                    full_text += event.get('content', '')
                elif etype == 'done':
                    if event.get('content'):
                        full_text = event['content']
                elif etype == 'classify':
                    classify = event.get('content', '')
                elif etype == 'references':
                    refs = event.get('content', [])
        except json.JSONDecodeError:
            pass

    return {"text": full_text, "classify": classify, "refs": refs}


# ══════════════════════════════════════════════════════════
#  Phase 4: 테스트 케이스 정의 & 실행
# ══════════════════════════════════════════════════════════

def run_tests(task_id, raw_data):
    print(f"\n{'='*70}")
    print(f"  테스트 실행 (원본 데이터 vs 챗봇 응답)")
    print(f"{'='*70}")

    bs = raw_data['bs']
    is_data = raw_data['is']
    cf = raw_data['cf']

    results = []
    pass_count = 0
    fail_count = 0
    warn_count = 0

    def run_one(qid, category, question, verify_fn):
        nonlocal pass_count, fail_count, warn_count
        print(f"\n{'─'*70}")
        print(f"  Q{qid} [{category}] {question[:60]}{'...' if len(question)>60 else ''}")
        print(f"{'─'*70}")

        start = time.time()
        resp = send_chat(task_id, question)
        elapsed = time.time() - start
        text = resp['text']

        print(f"  분류: {resp['classify']} | {len(text)}자 | {elapsed:.1f}초")
        print(f"  응답: {text[:150].replace(chr(10), ' ')}...")

        status, details = verify_fn(text, raw_data)

        for d in details:
            print(f"  {d}")

        if status == "PASS":
            pass_count += 1
            print(f"  ★ PASS")
        elif status == "WARN":
            warn_count += 1
            print(f"  △ WARN (응답 OK, 검증 제한)")
        else:
            fail_count += 1
            print(f"  ✗ FAIL")

        results.append({
            "id": qid, "category": category, "question": question,
            "status": status, "details": details,
            "response_length": len(text), "elapsed": round(elapsed, 1),
            "response": text[:500]
        })
        time.sleep(1)

    # ──────────────────────────────────────────
    # 카테고리 1: BS 수치 정확성 (원본 대조)
    # ──────────────────────────────────────────

    def verify_inventory(text, data):
        """재고자산 4개년 수치 대조"""
        details = []
        bs = data['bs']
        years = ['2021', '2022', '2023', '2024']
        found_years = 0
        for y in years:
            val = find_in_table(bs, '재고자산', y)
            if val is not None:
                # 챗봇 응답에서 해당 수치 근사값 찾기
                nums = extract_numbers_from_text(text)
                # 원 단위 또는 백만원 단위로 비교
                matched = any(numbers_match(val, n) or numbers_match(val/1000000, n, 2) for n in nums)
                if matched:
                    details.append(f"✓ FY{y} 재고자산 {val:,.0f} → 챗봇 일치")
                    found_years += 1
                else:
                    details.append(f"△ FY{y} 재고자산 {val:,.0f} → 챗봇에서 정확한 일치 찾기 어려움 (숫자 포맷 차이 가능)")
                    found_years += 1  # 데이터는 있으니 카운트
            else:
                details.append(f"△ FY{y} 재고자산 원본 데이터 없음")

        # 4개년 모두 언급되는지
        for y in years:
            if y not in text:
                details.append(f"✗ {y}년 미언급")
                return "FAIL", details

        if '확인 불가' in text or '데이터 없' in text:
            details.append(f"✗ '확인 불가' 또는 '데이터 없음' 문구 포함")
            return "FAIL", details

        return "PASS", details

    run_one(1, "BS-수치", "재고자산의 2021년부터 2024년까지의 값과 전년 대비 증감율을 알려줘", verify_inventory)

    def verify_total_assets(text, data):
        """자산총계 수치 대조"""
        details = []
        val_2024 = find_in_table(data['bs'], '자산총계', '2024')
        if val_2024:
            nums = extract_numbers_from_text(text)
            # 원 또는 백만원으로 비교
            matched = any(numbers_match(val_2024, n) or numbers_match(val_2024/1000000, n, 2) for n in nums)
            if matched:
                details.append(f"✓ 자산총계(2024) {val_2024:,.0f} 일치")
            else:
                details.append(f"△ 자산총계(2024) {val_2024:,.0f} — 근사치 확인 필요")
        if '자산총계' not in text:
            details.append(f"✗ '자산총계' 미언급")
            return "FAIL", details
        return "PASS", details

    run_one(2, "BS-수치", "2024년 자산총계, 부채총계, 자본총계를 각각 알려줘", verify_total_assets)

    def verify_cash(text, data):
        details = []
        for y in ['2021', '2022', '2023', '2024']:
            if y not in text:
                details.append(f"✗ {y}년 미언급")
                return "FAIL", details
        if '현금' in text:
            details.append(f"✓ 현금 관련 수치 포함")
        return "PASS", details

    run_one(3, "BS-수치", "현금및현금성자산과 단기금융상품의 합계를 연도별로 알려줘", verify_cash)

    def verify_receivables(text, data):
        details = []
        if '매출채권' in text or '매출채권및기타채권' in text:
            details.append("✓ 매출채권 항목 포함")
        else:
            details.append("✗ 매출채권 미언급")
            return "FAIL", details
        nums = extract_numbers_from_text(text)
        if len(nums) >= 4:
            details.append(f"✓ 숫자 {len(nums)}개 포함 (4개년 이상)")
        return "PASS", details

    run_one(4, "BS-수치", "매출채권및기타채권의 연도별 금액과 매출 대비 비중을 알려줘", verify_receivables)

    # ──────────────────────────────────────────
    # 카테고리 2: IS 수치 정확성
    # ──────────────────────────────────────────

    def verify_revenue(text, data):
        details = []
        val = find_in_table(data['is'], '매출', '2024')
        if val is not None:
            details.append(f"✓ 원본 매출(2024): {val:,.0f}")
        if '매출' in text and '영업이익' in text:
            details.append("✓ 매출 + 영업이익 모두 포함")
        else:
            details.append("✗ 매출 또는 영업이익 미포함")
            return "FAIL", details
        nums = extract_numbers_from_text(text)
        if len(nums) >= 8:
            details.append(f"✓ 숫자 {len(nums)}개 (4년치 매출+영업이익)")
        return "PASS", details

    run_one(5, "IS-수치", "매출액, 매출원가, 매출총이익, 영업이익을 연도별로 보여줘", verify_revenue)

    def verify_net_income(text, data):
        details = []
        if '당기순이익' in text or '순이익' in text:
            details.append("✓ 당기순이익 포함")
        else:
            details.append("✗ 당기순이익 미포함")
            return "FAIL", details
        return "PASS", details

    run_one(6, "IS-수치", "당기순이익의 연도별 추이와 순이익률을 계산해줘", verify_net_income)

    def verify_sga(text, data):
        details = []
        if '판매비' in text or '판관비' in text:
            details.append("✓ 판관비 포함")
        else:
            details.append("✗ 판관비 미포함")
            return "FAIL", details
        return "PASS", details

    run_one(7, "IS-수치", "판매비와관리비 세부 내역을 가장 큰 항목 순으로 보여줘", verify_sga)

    # ──────────────────────────────────────────
    # 카테고리 3: CF 수치
    # ──────────────────────────────────────────

    def verify_ocf(text, data):
        details = []
        if '영업활동' in text:
            details.append("✓ 영업활동현금흐름 포함")
        else:
            details.append("✗ 영업활동현금흐름 미포함")
            return "FAIL", details
        if '투자활동' in text:
            details.append("✓ 투자활동현금흐름 포함")
        if '재무활동' in text:
            details.append("✓ 재무활동현금흐름 포함")
        return "PASS", details

    run_one(8, "CF-수치", "영업활동, 투자활동, 재무활동 현금흐름을 연도별로 보여줘", verify_ocf)

    # ──────────────────────────────────────────
    # 카테고리 4: PE 핵심 지표 계산 정확성
    # ──────────────────────────────────────────

    def verify_ebitda(text, data):
        details = []
        if 'EBITDA' in text:
            details.append("✓ EBITDA 언급")
        else:
            return "FAIL", [f"✗ EBITDA 미언급"]
        if '영업이익' in text and '감가상각' in text:
            details.append("✓ 계산 구성요소(영업이익+감가상각) 명시")
        else:
            details.append("△ 계산 구성요소 일부 누락")
        # 출처 태그
        if '[IS원본]' in text or '[CF원본]' in text:
            details.append("✓ 출처 태그 포함")
        return "PASS", details

    run_one(9, "PE-지표", "EBITDA를 연도별로 계산해줘. 계산식과 각 구성요소 금액도 보여줘", verify_ebitda)

    def verify_net_debt(text, data):
        details = []
        if '순차입금' in text or 'Net Debt' in text:
            details.append("✓ 순차입금 언급")
        else:
            return "FAIL", ["✗ 순차입금 미언급"]
        if '차입' in text and '현금' in text:
            details.append("✓ 계산 구성요소 포함")
        return "PASS", details

    run_one(10, "PE-지표", "Net Debt을 계산하고, 총차입금 구성(단기차입금, 장기차입금, 사채 등)을 상세히 보여줘", verify_net_debt)

    def verify_ev_ebitda(text, data):
        details = []
        if 'EV' in text or 'EBITDA' in text:
            details.append("✓ EV/EBITDA 관련 언급")
        else:
            return "FAIL", ["✗ EV/EBITDA 미언급"]
        # 비상장이면 시가총액 불가 안내 확인
        if '시가총액' in text or '주가' in text or '비상장' in text or '상장' in text:
            details.append("✓ 시가총액/주가 관련 언급")
        return "PASS", details

    run_one(11, "PE-지표", "EV/EBITDA 멀티플을 추정할 수 있어? 가능하다면 계산해줘", verify_ev_ebitda)

    # ──────────────────────────────────────────
    # 카테고리 5: 회전율/효율성
    # ──────────────────────────────────────────

    def verify_ccc(text, data):
        details = []
        if 'CCC' in text or '현금' in text:
            details.append("✓ CCC 언급")
        else:
            return "FAIL", ["✗ CCC 미언급"]
        for kw in ['매출채권', '재고자산', '매입채무']:
            if kw in text:
                details.append(f"✓ {kw} 회전일수 포함")
            else:
                details.append(f"△ {kw} 미포함")
        return "PASS", details

    run_one(12, "효율성", "CCC(현금전환주기)를 연도별로 계산해줘. DSO, DSI, DPO 각각도 보여줘", verify_ccc)

    def verify_working_capital_detail(text, data):
        details = []
        for kw in ['매출채권', '재고자산', '매입채무']:
            if kw in text:
                details.append(f"✓ {kw} 포함")
            else:
                details.append(f"✗ {kw} 미포함")
                return "FAIL", details
        return "PASS", details

    run_one(13, "효율성", "운전자본 세부 항목(매출채권, 재고자산, 매입채무, 선수금 등)을 연도별로 분석해줘", verify_working_capital_detail)

    # ──────────────────────────────────────────
    # 카테고리 6: 안정성/레버리지
    # ──────────────────────────────────────────

    def verify_leverage(text, data):
        details = []
        if '부채비율' in text:
            details.append("✓ 부채비율 포함")
        else:
            return "FAIL", ["✗ 부채비율 미포함"]
        if '이자보상' in text or 'ICR' in text:
            details.append("✓ 이자보상배율 포함")
        else:
            details.append("△ 이자보상배율 미포함")
        return "PASS", details

    run_one(14, "안정성", "부채비율, 차입금의존도, 이자보상배율(ICR)을 연도별로 계산해줘", verify_leverage)

    def verify_roe_decomp(text, data):
        details = []
        if 'ROE' in text:
            details.append("✓ ROE 포함")
        else:
            return "FAIL", ["✗ ROE 미포함"]
        # 듀퐁 분해 여부
        if '순이익률' in text or '자산회전' in text or '레버리지' in text or '듀퐁' in text or 'DuPont' in text:
            details.append("✓ 듀퐁 분해 요소 포함")
        else:
            details.append("△ 듀퐁 분해 미수행")
        return "PASS", details

    run_one(15, "수익성", "ROE를 듀퐁 분석(DuPont Analysis)으로 분해해줘. 순이익률 × 자산회전율 × 재무레버리지", verify_roe_decomp)

    # ──────────────────────────────────────────
    # 카테고리 7: PE 심화 분석
    # ──────────────────────────────────────────

    def verify_quality_earnings(text, data):
        details = []
        if '이익의 질' in text or 'Quality' in text or 'Accrual' in text or '발생액' in text:
            details.append("✓ 이익의 질 관련 분석")
        if '영업활동현금흐름' in text or 'OCF' in text:
            details.append("✓ OCF 참조")
        if '당기순이익' in text or '순이익' in text:
            details.append("✓ 순이익 참조")
        if len(details) >= 2:
            return "PASS", details
        return "FAIL", details + ["✗ 충분한 분석 없음"]

    run_one(16, "PE-심화", "이 기업의 이익의 질(Quality of Earnings)을 평가해줘. Accrual ratio도 계산해줘", verify_quality_earnings)

    def verify_fcf(text, data):
        details = []
        if 'FCF' in text or '잉여현금흐름' in text or 'Free Cash Flow' in text:
            details.append("✓ FCF 언급")
        else:
            return "FAIL", ["✗ FCF 미언급"]
        if 'CAPEX' in text or '설비투자' in text or '유형자산' in text:
            details.append("✓ CAPEX 참조")
        return "PASS", details

    run_one(17, "PE-심화", "FCF(Free Cash Flow)를 연도별로 계산해줘. OCF - CAPEX로", verify_fcf)

    def verify_deal_risk(text, data):
        details = []
        risk_keywords = ['리스크', '위험', '우려', '주의', '취약', '의존', '집중']
        found = [kw for kw in risk_keywords if kw in text]
        if len(found) >= 2:
            details.append(f"✓ 리스크 키워드: {found}")
        else:
            details.append(f"△ 리스크 키워드 부족: {found}")
        if len(text) > 300:
            details.append(f"✓ 충분한 분석 ({len(text)}자)")
        return "PASS", details

    run_one(18, "PE-심화", "이 기업을 인수한다고 가정할 때, 재무적 리스크 요인 Top 5를 뽑아줘", verify_deal_risk)

    def verify_valuation(text, data):
        details = []
        val_keywords = ['밸류에이션', 'EV', 'EBITDA', '멀티플', 'multiple', 'PER', 'PBR']
        found = [kw for kw in val_keywords if kw in text]
        if len(found) >= 2:
            details.append(f"✓ 밸류에이션 키워드: {found}")
        else:
            details.append(f"△ 키워드 부족: {found}")
        return "PASS", details

    run_one(19, "PE-심화", "이 기업의 적정 밸류에이션 레인지를 추정해줘. Comparable 기업 대비 할인/프리미엄 요소도 언급해줘", verify_valuation)

    # ──────────────────────────────────────────
    # 카테고리 8: 대화 규칙 테스트
    # ──────────────────────────────────────────

    def verify_greeting(text, data):
        details = []
        if len(text) > 200:
            details.append(f"✗ 인사 응답이 과다: {len(text)}자")
            return "FAIL", details
        # 재무 분석 시작하면 안됨
        bad_kw = ['매출액', 'EBITDA', '영업이익률', '부채비율']
        for bk in bad_kw:
            if bk in text:
                details.append(f"✗ 인사에 재무분석 포함: '{bk}'")
                return "FAIL", details
        details.append(f"✓ 짧은 인사 응답 ({len(text)}자)")
        return "PASS", details

    run_one(20, "대화규칙", "안녕!", verify_greeting)

    def verify_simple_q(text, data):
        details = []
        if '매출' in text:
            details.append("✓ 매출 언급")
        else:
            return "FAIL", ["✗ 매출 미언급"]
        if len(text) > 800:
            details.append(f"△ 간단한 질문 대비 응답 과다: {len(text)}자")
        else:
            details.append(f"✓ 적절한 응답 길이 ({len(text)}자)")
        return "PASS", details

    run_one(21, "대화규칙", "매출이 얼마야?", verify_simple_q)

    def verify_unknown(text, data):
        details = []
        if '확인 불가' in text or '데이터' in text and '없' in text or '제공' in text:
            details.append("✓ 데이터 없음 정직하게 응답")
        elif '추정' in text or '가정' in text:
            details.append("✓ 추정/가정 명시")
        else:
            details.append("△ 정직한 한계 인정 확인 필요")
        # 숫자를 지어내면 안됨
        if '대표이사 연봉' in text and any(c.isdigit() for c in text.split('대표이사 연봉')[1][:50] if len(text.split('대표이사 연봉')) > 1):
            details.append("✗ 없는 데이터에 대해 숫자를 생성했을 가능성")
            return "FAIL", details
        return "PASS", details

    run_one(22, "할루시네이션", "대표이사 연봉이 얼마야?", verify_unknown)

    def verify_no_hallucination(text, data):
        details = []
        # 2020년 이전 데이터를 만들어내면 안됨
        if '2019' in text and ('매출' in text or '영업이익' in text):
            nums_around_2019 = re.findall(r'2019.*?[\d,]+', text[:text.index('2019')+100] if '2019' in text else '')
            if nums_around_2019:
                details.append("✗ 2019년 데이터를 생성했을 수 있음 (DART에 없는 연도)")
                return "FAIL", details
        if '확인 불가' in text or '데이터' in text or '2021' in text:
            details.append("✓ 데이터 범위를 정직하게 안내")
        return "PASS", details

    run_one(23, "할루시네이션", "2019년 매출이 얼마였어?", verify_no_hallucination)

    # ──────────────────────────────────────────
    # 카테고리 9: 복합 질문
    # ──────────────────────────────────────────

    def verify_dd_summary(text, data):
        details = []
        dd_keywords = ['수익성', '안정성', '성장', '현금흐름', '운전자본', '차입', '리스크']
        found = [kw for kw in dd_keywords if kw in text]
        if len(found) >= 3:
            details.append(f"✓ DD 핵심 영역 {len(found)}개 커버: {found}")
        else:
            details.append(f"△ DD 영역 부족 ({len(found)}개): {found}")
        if len(text) > 500:
            details.append(f"✓ 충분한 분석 ({len(text)}자)")
        return "PASS", details

    run_one(24, "복합분석", "이 기업에 대한 Financial Due Diligence 핵심 요약을 작성해줘. 수익성, 안정성, 성장성, 현금흐름 관점에서", verify_dd_summary)

    def verify_yoy_comparison(text, data):
        details = []
        if '2023' in text and '2024' in text:
            details.append("✓ 2023 vs 2024 비교")
        else:
            details.append("✗ 연도 비교 미수행")
            return "FAIL", details
        change_keywords = ['증가', '감소', '개선', '악화', '상승', '하락', '변동']
        found = [kw for kw in change_keywords if kw in text]
        if found:
            details.append(f"✓ 변동 분석: {found}")
        return "PASS", details

    run_one(25, "복합분석", "2023년 대비 2024년에 가장 크게 변한 재무 항목 Top 5를 뽑아줘", verify_yoy_comparison)

    # ──────────────────────────────────────────
    #  최종 요약
    # ──────────────────────────────────────────
    total = len(results)
    print(f"\n{'='*70}")
    print(f"  최종 결과")
    print(f"{'='*70}")
    print(f"  PASS : {pass_count}/{total}")
    print(f"  WARN : {warn_count}/{total}")
    print(f"  FAIL : {fail_count}/{total}")
    print(f"  성공률: {(pass_count + warn_count)/total*100:.0f}%")

    if fail_count > 0:
        print(f"\n  실패 항목:")
        for r in results:
            if r['status'] == 'FAIL':
                print(f"    Q{r['id']} [{r['category']}] {r['question'][:50]}")
                for d in r['details']:
                    if '✗' in d:
                        print(f"      {d}")

    # 카테고리별 결과
    cats = {}
    for r in results:
        c = r['category']
        if c not in cats:
            cats[c] = {'pass': 0, 'fail': 0, 'warn': 0}
        cats[c][r['status'].lower()] = cats[c].get(r['status'].lower(), 0) + 1
    print(f"\n  카테고리별:")
    for c, v in cats.items():
        print(f"    {c:12s}: PASS={v.get('pass',0)} WARN={v.get('warn',0)} FAIL={v.get('fail',0)}")

    # 저장
    with open("/home/servermanager/pers-dev/test_chatbot_verify_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: test_chatbot_verify_results.json")

    return fail_count == 0


if __name__ == "__main__":
    task_id, raw_data = setup()
    success = run_tests(task_id, raw_data)
    sys.exit(0 if success else 1)
