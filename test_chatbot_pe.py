#!/usr/bin/env python3
"""PE 챗봇 자동 테스트 - 20개 PE 관점 질문 + 응답 검증"""

import requests
import json
import time
import sys
import re

BASE_URL = "http://localhost:8001"
COMPANY_NAME = "지에스엔텍"
STOCK_CODE = "208350"

# ── 1. 기업 검색 & 추출 ──
def search_company():
    print(f"\n{'='*60}")
    print(f"[1/4] 기업 검색: {COMPANY_NAME}")
    print(f"{'='*60}")
    resp = requests.post(f"{BASE_URL}/api/search", json={"company_name": COMPANY_NAME})
    data = resp.json()
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
    elif isinstance(data, list):
        results = data
    else:
        results = data.get('results', data.get('data', []))

    for r in results:
        if r.get('stock_code') == STOCK_CODE or COMPANY_NAME in r.get('corp_name', ''):
            print(f"  → 찾음: {r['corp_name']} (corp_code={r['corp_code']})")
            return r

    # Try direct extraction with stock code
    print(f"  → 검색 결과에서 못 찾음. 직접 추출 시도...")
    return None

def get_company_info(corp_code):
    print(f"\n[2/4] 기업개황정보 조회: {corp_code}")
    resp = requests.get(f"{BASE_URL}/api/company-info/{corp_code}")
    data = resp.json()
    if data.get('success'):
        info = data['data']
        print(f"  → {info['corp_name']} / 대표: {info.get('ceo_nm','?')} / 시장: {info.get('market_name','?')}")
        return info
    return {}

def extract_financial_data(corp_code, company_info):
    print(f"\n[3/4] 재무제표 추출 시작")
    resp = requests.post(f"{BASE_URL}/api/extract", json={
        "corp_code": corp_code,
        "corp_name": COMPANY_NAME,
        "start_year": 2021,
        "end_year": 2024,
        "company_info": company_info
    })
    data = resp.json()
    task_id = data.get('task_id')
    if not task_id:
        print(f"  ✗ 추출 실패: {data}")
        return None

    print(f"  → task_id: {task_id}")

    # 폴링
    for i in range(120):  # 최대 6분
        time.sleep(3)
        resp = requests.get(f"{BASE_URL}/api/status/{task_id}")
        status_data = resp.json()
        status = status_data.get('status', '?')
        progress = status_data.get('progress', 0)
        msg = status_data.get('message', '')

        if i % 5 == 0:
            print(f"  ... [{progress}%] {msg[:60]}")

        if status == 'completed':
            print(f"  ✓ 추출 완료!")
            return task_id
        elif status == 'failed':
            print(f"  ✗ 추출 실패: {msg}")
            return None

    print(f"  ✗ 타임아웃")
    return None

def init_chatbot(task_id):
    print(f"\n[4/4] 챗봇 초기화")
    resp = requests.post(f"{BASE_URL}/api/chat/init/{task_id}")
    data = resp.json()
    if data.get('success'):
        print(f"  ✓ 챗봇 초기화 완료: {data.get('company_name')}")
        return True
    print(f"  ✗ 초기화 실패: {data}")
    return False

# ── 2. 챗봇 질문 & 응답 수집 ──
def send_chat(task_id, message):
    """SSE 스트리밍 응답 수집"""
    resp = requests.post(
        f"{BASE_URL}/api/chat/message/{task_id}",
        json={"message": message},
        stream=True,
        timeout=120
    )

    full_text = ""
    refs = []
    classify = ""

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
                elif etype == 'error':
                    full_text = f"[ERROR] {event.get('content','')}"
        except json.JSONDecodeError:
            pass

    return {
        "text": full_text,
        "classify": classify,
        "refs": refs,
        "length": len(full_text)
    }

# ── 3. PE 관점 질문 20개 ──
PE_QUESTIONS = [
    # --- 기본 재무 수치 정확성 (BS) ---
    {
        "id": 1,
        "question": "재고자산의 2021년부터 2024년까지의 값과 전년 대비 증감율을 알려줘",
        "check": "4개년 수치가 모두 있어야 함. FY2021, FY2022, FY2023, FY2024",
        "keywords": ["2021", "2022", "2023", "2024", "재고자산"],
        "fail_keywords": ["확인 불가", "데이터 없", "제공되지"]
    },
    {
        "id": 2,
        "question": "매출채권및기타채권의 연도별 추이를 알려줘",
        "check": "BS원본에서 매출채권 데이터 제공",
        "keywords": ["매출채권"],
        "fail_keywords": ["데이터 없"]
    },
    {
        "id": 3,
        "question": "현금및현금성자산이 최근 4년간 어떻게 변했어?",
        "check": "현금 추이 정보",
        "keywords": ["현금"],
        "fail_keywords": []
    },
    # --- IS 수치 ---
    {
        "id": 4,
        "question": "매출액과 영업이익의 연도별 추이 및 영업이익률을 계산해줘",
        "check": "매출, 영업이익, 영업이익률 계산 필요",
        "keywords": ["매출", "영업이익"],
        "fail_keywords": []
    },
    {
        "id": 5,
        "question": "매출원가율과 판관비율의 연도별 추이를 분석해줘",
        "check": "원가율 = 매출원가/매출, 판관비율 = 판관비/매출",
        "keywords": ["매출원가", "판관비"],
        "fail_keywords": []
    },
    # --- PE 핵심 지표 계산 ---
    {
        "id": 6,
        "question": "EBITDA를 계산해줘. 영업이익 + 감가상각비 + 무형자산상각비로",
        "check": "EBITDA 계산 정확성",
        "keywords": ["EBITDA", "감가상각"],
        "fail_keywords": []
    },
    {
        "id": 7,
        "question": "Net Debt(순차입금)을 계산해줘",
        "check": "순차입금 = 차입금 - 현금성자산",
        "keywords": ["차입", "현금"],
        "fail_keywords": []
    },
    {
        "id": 8,
        "question": "NWC(순운전자본)를 계산하고 연도별 변화를 분석해줘",
        "check": "NWC = 유동자산 - 유동부채",
        "keywords": ["유동자산", "유동부채"],
        "fail_keywords": []
    },
    # --- 회전율/효율성 ---
    {
        "id": 9,
        "question": "매출채권 회전율과 회전일수를 계산해줘",
        "check": "회전율 = 매출/매출채권, 회전일수 = 365/회전율",
        "keywords": ["회전율", "회전일수"],
        "fail_keywords": ["불가"]
    },
    {
        "id": 10,
        "question": "재고자산 회전율과 회전일수를 계산해줘",
        "check": "재고자산 회전율",
        "keywords": ["회전율"],
        "fail_keywords": ["불가"]
    },
    {
        "id": 11,
        "question": "매입채무 회전율과 회전일수를 계산해줘",
        "check": "매입채무 회전율. BS원본에서 매입채무 데이터 찾아야 함",
        "keywords": ["회전율"],
        "fail_keywords": ["불가", "기말 잔액 부재"]
    },
    {
        "id": 12,
        "question": "CCC(현금순환주기)를 계산해줘",
        "check": "CCC = 매출채권 회전일수 + 재고자산 회전일수 - 매입채무 회전일수",
        "keywords": ["CCC", "현금순환"],
        "fail_keywords": []
    },
    # --- 안정성/레버리지 ---
    {
        "id": 13,
        "question": "부채비율과 유동비율의 연도별 추이를 분석해줘",
        "check": "부채비율 = 부채/자본, 유동비율 = 유동자산/유동부채",
        "keywords": ["부채비율", "유동비율"],
        "fail_keywords": []
    },
    {
        "id": 14,
        "question": "차입금 의존도를 계산해줘. 총차입금/자산총계로",
        "check": "차입금 의존도",
        "keywords": ["차입금"],
        "fail_keywords": []
    },
    # --- 수익성 ---
    {
        "id": 15,
        "question": "ROE와 ROA를 연도별로 계산해줘",
        "check": "ROE = 순이익/자본, ROA = 순이익/자산",
        "keywords": ["ROE", "ROA"],
        "fail_keywords": []
    },
    # --- PE 관점 심화 ---
    {
        "id": 16,
        "question": "이 기업의 매출 성장률 대비 영업이익 성장률의 차이를 분석해줘. 스케일업 효과가 있는지 판단해줘",
        "check": "operating leverage 분석",
        "keywords": ["매출", "영업이익", "성장"],
        "fail_keywords": []
    },
    {
        "id": 17,
        "question": "유형자산의 연도별 추이와 CAPEX 추정치를 알려줘",
        "check": "유형자산 추이, CF에서 CAPEX",
        "keywords": ["유형자산"],
        "fail_keywords": []
    },
    {
        "id": 18,
        "question": "영업활동현금흐름과 당기순이익의 괴리를 분석해줘. 이익의 질(Quality of Earnings)이 어떤지 판단해줘",
        "check": "OCF vs 순이익 비교",
        "keywords": ["영업활동현금흐름", "당기순이익"],
        "fail_keywords": []
    },
    # --- 인사/간단 질문 ---
    {
        "id": 19,
        "question": "안녕하세요",
        "check": "짧은 인사 응답. 재무분석 시작하지 않아야 함",
        "keywords": [],
        "fail_keywords": ["EBITDA", "매출액", "영업이익률"],
        "max_length": 200  # 인사는 짧아야 함
    },
    {
        "id": 20,
        "question": "자산총계가 얼마야?",
        "check": "간단한 수치 질문 → 짧게 핵심만 답변",
        "keywords": ["자산총계"],
        "fail_keywords": [],
        "max_length": 500  # 간단한 질문은 짧아야 함
    },
]

# ── 4. 결과 검증 ──
def verify_response(q, response):
    """응답 검증"""
    text = response["text"]
    results = []
    passed = True

    # 1) 필수 키워드 포함 여부
    for kw in q.get("keywords", []):
        if kw in text:
            results.append(f"  ✓ '{kw}' 포함")
        else:
            results.append(f"  ✗ '{kw}' 미포함")
            passed = False

    # 2) 실패 키워드 미포함 여부
    for fk in q.get("fail_keywords", []):
        if fk in text:
            results.append(f"  ✗ 실패 키워드 '{fk}' 포함됨!")
            passed = False

    # 3) 응답 길이 제한 확인
    max_len = q.get("max_length")
    if max_len and len(text) > max_len:
        results.append(f"  ✗ 응답 과다: {len(text)}자 > 최대 {max_len}자")
        passed = False

    # 4) 에러 응답 확인
    if text.startswith("[ERROR]"):
        results.append(f"  ✗ 에러 응답!")
        passed = False

    # 5) 빈 응답 확인
    if len(text) < 10:
        results.append(f"  ✗ 응답이 너무 짧음: {len(text)}자")
        passed = False

    # 6) 숫자 포함 확인 (재무 질문인 경우)
    if q["id"] <= 18:  # 재무 질문
        numbers = re.findall(r'[\d,]+\.?\d*', text)
        if len(numbers) >= 2:
            results.append(f"  ✓ 숫자 {len(numbers)}개 포함")
        else:
            results.append(f"  △ 숫자가 적음 ({len(numbers)}개)")

    # 7) 출처 태그 확인 (재무 질문인 경우)
    if q["id"] <= 18:
        source_tags = re.findall(r'\[(BS원본|IS원본|CF원본|CIS원본|주석|웹검색)\]', text)
        if source_tags:
            results.append(f"  ✓ 출처 태그: {set(source_tags)}")
        else:
            results.append(f"  △ 출처 태그 없음")

    return passed, results


def main():
    print("\n" + "="*60)
    print("  PE 챗봇 자동 테스트 (20개 질문)")
    print("="*60)

    # Step 1: 기업 검색
    company = search_company()
    if not company:
        print("✗ 기업을 찾을 수 없습니다.")
        sys.exit(1)

    corp_code = company['corp_code']

    # Step 2: 기업개황정보
    company_info = get_company_info(corp_code)

    # Step 3: 재무제표 추출
    task_id = extract_financial_data(corp_code, company_info)
    if not task_id:
        print("✗ 추출 실패")
        sys.exit(1)

    # Step 4: 챗봇 초기화
    if not init_chatbot(task_id):
        print("✗ 챗봇 초기화 실패")
        sys.exit(1)

    # Step 5: 20개 질문 테스트
    print(f"\n{'='*60}")
    print(f"  질문 테스트 시작 (총 {len(PE_QUESTIONS)}개)")
    print(f"{'='*60}")

    total_pass = 0
    total_fail = 0
    all_results = []

    for i, q in enumerate(PE_QUESTIONS):
        print(f"\n--- Q{q['id']}/{len(PE_QUESTIONS)}: {q['question'][:50]}{'...' if len(q['question'])>50 else ''} ---")
        print(f"  검증: {q['check']}")

        start = time.time()
        response = send_chat(task_id, q["question"])
        elapsed = time.time() - start

        print(f"  분류: {response['classify']} | 응답 길이: {response['length']}자 | 시간: {elapsed:.1f}초")

        # 응답 미리보기 (처음 200자)
        preview = response["text"][:200].replace('\n', ' ')
        print(f"  응답: {preview}...")

        # 검증
        passed, details = verify_response(q, response)
        for d in details:
            print(d)

        status = "PASS" if passed else "FAIL"
        print(f"  → 결과: {status}")

        if passed:
            total_pass += 1
        else:
            total_fail += 1

        all_results.append({
            "id": q["id"],
            "question": q["question"],
            "status": status,
            "classify": response["classify"],
            "length": response["length"],
            "elapsed": round(elapsed, 1),
            "response_preview": response["text"][:300]
        })

        # 너무 빠른 연속 요청 방지
        time.sleep(1)

    # Step 6: 최종 요약
    print(f"\n{'='*60}")
    print(f"  최종 결과 요약")
    print(f"{'='*60}")
    print(f"  PASS: {total_pass}/{len(PE_QUESTIONS)}")
    print(f"  FAIL: {total_fail}/{len(PE_QUESTIONS)}")
    print(f"  성공률: {total_pass/len(PE_QUESTIONS)*100:.0f}%")

    if total_fail > 0:
        print(f"\n  ✗ 실패한 질문:")
        for r in all_results:
            if r["status"] == "FAIL":
                print(f"    Q{r['id']}: {r['question'][:50]}")

    # 결과 JSON 저장
    with open("/home/servermanager/pers-dev/test_chatbot_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: test_chatbot_results.json")

    return total_fail == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
