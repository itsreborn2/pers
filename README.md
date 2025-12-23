# DART 재무제표 추출기

dart-fss 오픈소스 라이브러리를 사용하여 상장/비상장 기업의 재무제표를 추출하는 도구입니다.

## 빠른 시작 (웹 UI)

```powershell
# 1. 패키지 설치
pip install -r requirements.txt

# 2. DART API 키 설정
$env:DART_API_KEY = "YOUR_API_KEY"

# 3. 서버 실행
python server.py
```

브라우저에서 http://localhost:8000 접속

## 설치

```bash
pip install -r requirements.txt
```

## DART API 키 발급

1. [DART 오픈API](https://opendart.fss.or.kr/) 접속
2. 회원가입 후 API 키 발급
3. 환경변수 설정:
   ```powershell
   # Windows PowerShell
   $env:DART_API_KEY = "YOUR_API_KEY"
   
   # 영구 설정
   [Environment]::SetEnvironmentVariable("DART_API_KEY", "YOUR_API_KEY", "User")
   ```

## 사용법

### 기본 사용

```python
from dart_financial_extractor import DartFinancialExtractor

# 초기화 (환경변수에서 API 키 자동 로드)
extractor = DartFinancialExtractor()

# 또는 직접 API 키 전달
extractor = DartFinancialExtractor(api_key='YOUR_API_KEY')
```

### 회사 검색

```python
# 회사명으로 검색 (상장/비상장 모두)
companies = extractor.search_company("삼성전자")

# 정확히 일치하는 회사만 검색
companies = extractor.search_company("삼성전자", exactly=True)

# 시장별 필터링
# 'Y': 코스피, 'K': 코스닥, 'N': 코넥스, 'E': 기타(비상장)
companies = extractor.search_company("삼성", market='YK')  # 코스피+코스닥

# DART 고유번호로 검색
corp = extractor.search_by_corp_code('00126380')

# 주식 종목코드로 검색 (상장사만)
corp = extractor.search_by_stock_code('005930')
```

### 재무제표 추출

```python
# 재무제표 추출
fs_data = extractor.extract_financial_statements(
    corp_code='00126380',      # DART 고유번호
    start_date='20200101',     # 시작일
    end_date='20241231',       # 종료일 (생략시 오늘)
    fs_types=['bs', 'is'],     # 재무상태표, 손익계산서
    report_tp='annual',        # 연간 보고서
    separate=False             # 연결재무제표 (True: 개별)
)

# 재무상태표
balance_sheet = fs_data['bs']

# 손익계산서
income_statement = fs_data['is']
```

### 재무제표 유형

| 코드 | 재무제표 |
|------|----------|
| `bs` | 재무상태표 (Balance Sheet) |
| `is` | 손익계산서 (Income Statement) |
| `cis` | 포괄손익계산서 (Comprehensive Income Statement) |
| `cf` | 현금흐름표 (Cash Flow Statement) |

### 보고서 유형

| 값 | 설명 |
|----|------|
| `annual` | 연간 (사업보고서) |
| `half` | 연간 + 반기 |
| `quarter` | 연간 + 반기 + 분기 |

### 엑셀 저장

```python
extractor.save_to_excel(fs_data, "삼성전자_재무제표", path="./output")
```

## 추출 가능한 항목

### 재무상태표 (이미지 기준)
- ✅ 유동자산, 비유동자산, 자산총계
- ✅ 현금및현금성자산, 재고자산, 매출채권
- ✅ 유형자산, 무형자산
- ✅ 유동부채, 비유동부채, 부채총계
- ✅ 매입채무, 단기차입금, 장기차입금
- ✅ 자본금, 이익잉여금, 자본총계
- ⚠️ NWC, Net Debt: 직접 계산 필요

### 손익계산서 (이미지 기준)
- ✅ 매출, 매출원가, 매출총이익
- ✅ 판매비와관리비, 영업이익
- ✅ 영업외수익, 영업외비용
- ✅ 법인세비용차감전이익, 법인세비용
- ✅ 당기순이익
- ⚠️ EBITDA: 직접 계산 필요 (영업이익 + 감가상각비)
- ⚠️ % of Sales: 직접 계산 필요

## 제한사항

1. **비상장사 재무제표**: DART에 공시된 기업만 조회 가능
   - 비상장사도 외부감사 대상이면 DART에 공시됨
   - 소규모 비상장사는 공시 의무가 없어 조회 불가

2. **데이터 기간**: 2015년 이후 데이터만 제공

3. **금융업 제외**: 상장법인 재무정보 API는 금융업 제외

## 참고 자료

- [dart-fss GitHub](https://github.com/josw123/dart-fss)
- [dart-fss 문서](https://dart-fss.readthedocs.io/)
- [DART 오픈API](https://opendart.fss.or.kr/)
