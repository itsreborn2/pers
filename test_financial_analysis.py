"""
재무제표 분석 에이전트 테스트

넥시스디자인그룹의 재무상태표와 손익계산서 데이터로 테스트합니다.
"""

import os
import sys
import pandas as pd
from io import StringIO

# 모듈 import
try:
    from financial_analysis_agent import analyze_financial_statements
except ImportError:
    print("Error: financial_analysis_agent.py를 찾을 수 없습니다.")
    sys.exit(1)


# 넥시스디자인그룹 재무상태표 데이터
BALANCE_SHEET_DATA = """
항목,FY2020,FY2021,FY2022,FY2023,FY2024
유동자산,4317,4245,5081,5870,5351
현금및현금성자산,490,364,318,191,50.3
단기금융상품,21,,,,
재고자산,2261,2581,3578,3994,3322
매출채권및기타채권,1569,1291,1199,1694,2000
기타비금융자산,36.4,49.1,46.3,67.4,66.6
비유동자산,1443,1165,1022,1067,1069
유형자산,610,543,476,532,628
무형자산,138,111,97.7,63.5,31.4
장기투자자산,206,55.3,28.4,26.3,30.6
보증금,228,234,220,305,291
자산총계,5759,5409,6103,6937,6420
유동부채,3593,3176,3430,4128,4042
매입채무및기타채무,1983,1592,1709,2192,2134
단기차입금,1362,1299,1466,1699,1474
유동성장기차입금,224,242,222,194,119
유동성사채,,,,0,300
기타유동부채,24.6,42.2,33.2,44.1,14.9
비유동부채,285,267,570,546,327
매입채무및기타채무[비유동],0.4,0.4,0,0,193
사채,,,,0,0
장기차입금,256,219,195,187,67.8
퇴직급여채무,,,,0,0
부채총계,3879,3443,4000,4674,4369
자본금,60,60,60,60,60
이익잉여금,1626,1712,1849,2009,1722
기타자본구성요소,195,195,195,195,269
자본총계,1881,1966,2103,2263,2051
부채와자본총계,5759,5409,6103,6937,6420
NWC,724,1069,1651,1742,1309
Net Debt,1331,1397,1565,1888,1911
"""

# 넥시스디자인그룹 손익계산서 데이터
INCOME_STATEMENT_DATA = """
항목,FY2020,FY2021,FY2022,FY2023,FY2024
매출,12041,8534,11968,12952,15512
상품매출,1905,2841,4079,2947,4633
제품매출,10036,5693,7889,10003,10841
기타매출,100,0,0,2.1,38.2
매출원가,10511,7093,10386,11081,13269
상품매출원가,1294,1859,3131,2289,3526
제품매출원가,9217,5234,7255,8792,9743
매출총이익,1530,1441,1582,1871,2243
% of Sales,0.127,0.169,0.132,0.144,0.145
판매비와관리비,1265,1167,1300,1492,1720
인건비,587,579,666,685,769
수수료비용,172,72.9,77.1,185,199
임차료비용,85.1,102,117,129,172
여비교통비,33.4,39.8,78.7,75.2,94.3
보험료,67.7,49.2,45.1,52.3,59.6
접대비,41.3,49.6,51.1,47.5,54.7
감가상각비,34.6,46.8,32.5,27.1,89.1
연구비,57.3,35.6,26.5,30.9,29
기타판매비와관리비,187,192,206,261,253
영업이익,265,274,282,379,523
% of Sales,0.022,0.032,0.024,0.029,0.034
영업외수익,93.8,56.1,90.7,64.6,48.9
금융수익,7.3,4.8,3.2,9.5,8.2
영업외비용,187,104,163,202,265
금융비용,74.6,50,86.3,116,150
법인세비용차감전이익,171,227,210,241,307
법인세비용,52.8,50.6,69.9,69.8,45.6
당기순이익,118,176,140,172,261
% of Sales,0.010,0.021,0.012,0.013,0.017
EBITDA,299,321,315,406,612
% of Sales,0.025,0.038,0.026,0.031,0.039
"""


def main():
    """테스트 실행"""
    print("="*80)
    print("재무제표 분석 에이전트 테스트")
    print("="*80)
    print()

    # 환경변수 확인
    print("[1] 환경변수 확인...")
    required_env = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]

    missing_env = []
    for env_var in required_env:
        value = os.getenv(env_var)
        if not value or value.startswith("your_"):
            missing_env.append(env_var)
            print(f"  ⚠️  {env_var}: 설정되지 않음")
        else:
            # API 키는 일부만 표시
            if "KEY" in env_var:
                display_value = value[:10] + "..." if len(value) > 10 else value
            else:
                display_value = value
            print(f"  ✓ {env_var}: {display_value}")

    if missing_env:
        print()
        print("❌ 다음 환경변수를 .env 파일에 설정해주세요:")
        for env_var in missing_env:
            print(f"  - {env_var}")
        print()
        print("테스트를 종료합니다.")
        return

    print()

    # 데이터 로드
    print("[2] 재무제표 데이터 로드...")
    try:
        balance_sheet = pd.read_csv(StringIO(BALANCE_SHEET_DATA.strip()))
        income_statement = pd.read_csv(StringIO(INCOME_STATEMENT_DATA.strip()))
        print(f"  ✓ 재무상태표: {balance_sheet.shape[0]} 행 x {balance_sheet.shape[1]} 열")
        print(f"  ✓ 손익계산서: {income_statement.shape[0]} 행 x {income_statement.shape[1]} 열")
    except Exception as e:
        print(f"  ❌ 데이터 로드 실패: {e}")
        return

    print()

    # 재무제표 분석
    print("[3] AI 분석 실행 (이 작업은 1-2분 소요될 수 있습니다)...")
    print()

    try:
        result = analyze_financial_statements(
            balance_sheet=balance_sheet,
            income_statement=income_statement,
            company_name='넥시스디자인그룹',
            search_news=True  # 뉴스 검색 활성화
        )

        print()
        print("[4] 분석 결과:")
        print()
        print(result)

        # 결과를 파일로 저장
        output_file = "output/nexis_analysis_result.txt"
        os.makedirs("output", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)

        print()
        print(f"✓ 분석 결과가 {output_file}에 저장되었습니다.")

    except Exception as e:
        print(f"  ❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("="*80)
    print("테스트 완료!")
    print("="*80)


if __name__ == "__main__":
    # .env 파일 로드
    from dotenv import load_dotenv
    load_dotenv()

    main()
