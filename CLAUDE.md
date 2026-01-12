# PERS 프로젝트 개발 규칙

## 문서화 규칙 (★필수★)

### 핵심 원칙
**메이저 수정, 새 스크립트 추가, API 변경 시 반드시 이 CLAUDE.md 파일에 기록하라.**

### 기록 대상
- 새로운 Python 스크립트 추가
- 새로운 API 엔드포인트 추가
- UI 구조 변경 (새 섹션, 탭 추가 등)
- 데이터 흐름 변경
- 엑셀 시트 구조 변경
- 주요 버그 수정 및 해결 방법

### 기록 형식
```markdown
## [기능명]

### 개요
[1-2줄 설명]

### 데이터 흐름
1. 프론트엔드: ...
2. 백엔드: ...

### 관련 파일
- `파일명`: 역할

### API (해당 시)
- `METHOD /api/endpoint`: 설명
```

---

## 백엔드/프론트엔드 동기화 규칙

### 핵심 원칙
**백엔드(server.py)를 수정할 때 반드시 프론트엔드(index.html)도 함께 확인하라.**

### VCM 데이터 흐름

1. **백엔드 (server.py)**
   - `create_vcm_format()` 함수에서 VCM 데이터 생성
   - Excel 파일에 2개 시트 저장:
     - `VCM전용포맷`: 메타데이터 포함 (항목, 타입, 부모, FY연도...)
     - `복사용테이블`: 프론트 표시용 (항목, FY연도... 단위: 천만원)
   - `preview_data['vcm']`, `preview_data['vcm_display']`로 API 응답에 포함

2. **프론트엔드 (index.html)**
   - `vcm_display` 데이터가 있으면 **그대로 렌더링** (단순화된 방식)
   - `vcm` 메타데이터에서 타입 정보 참조 (category, subitem, total, highlight)
   - 빈 행은 백엔드에서 필터링되어 프론트에 전달되지 않음

---

## VCM 표시 규칙 (★중요★)

### 핵심 원칙
**프론트엔드는 복사용테이블을 그대로 렌더링한다. 모든 필터링/그룹핑은 백엔드에서 처리.**

### 항목 표시 개수 제한

| 섹션 | 최대 항목 수 | 나머지 처리 |
|------|-------------|------------|
| 유동자산 | 6개 | 기타유동자산으로 합산 |
| 비유동자산 | 6개 | 기타비유동자산으로 합산 |
| 유동부채 | 6개 | 기타유동부채로 합산 |
| 비유동부채 | 6개 | 기타비유동부채로 합산 |
| 판관비 | 8개 | 기타판매비와관리비로 합산 |
| 매출 하위항목 | 전체 | (제한 없음) |
| 매출원가 하위항목 | 전체 | (제한 없음) |

### 선택 기준
1. **필수 항목** 먼저 포함 (예: 현금및현금성자산, 매출채권, 유형자산 등)
2. **금액 큰 순**으로 나머지 슬롯 채움
3. **나머지**는 "기타XXX"로 합산
4. 합산된 항목들은 **툴팁**에 세부 내역 표시

### 복사용테이블 생성 규칙
- 단위: **천만원** (백엔드에서 변환 완료)
- 숫자 포맷: 천 단위 콤마 (예: 1,234)
- 빈 행: **제외** (모든 연도에 값이 없는 행)
- 타입별 스타일링은 프론트엔드에서 vcm 메타데이터 참조

### 복사용테이블 필터링 규칙

**표시되는 항목:**
1. `부모` 없는 항목 → 표시 (메인 항목)
2. IS 섹션 하위항목 → 표시:
   - `부모='매출'`
   - `부모='매출원가'`
   - `부모='판매비와관리비'`
   - `부모='영업외수익'`
   - `부모='영업외비용'`

**제외되는 항목 (툴팁용):**
- BS 세부항목 (예: 현금및현금성자산 하위의 현금, 당좌예금 등)
- 기타XXX 하위항목 (예: 기타유동자산 하위 항목들)
- NWC, Net Debt 하위항목

**코드 위치:** `server.py` 내 `create_vcm_format()` 함수의 복사용테이블 생성 부분

### 주의사항

- 백엔드에서 VCM 항목명을 변경하면 프론트엔드 매칭 로직도 확인
- 새로운 카테고리 추가 시 프론트엔드의 `categoryNames`, `totalNames` 배열 확인
- 세부항목의 `부모` 필드가 올바르게 설정되어야 툴팁이 정상 작동
- **항목명 매칭 시 `startsWith()` 대신 `includes()` 또는 정확한 매칭 사용**
  - 예: `'매출'`로 startsWith 체크하면 `'매출채권및기타채권'`도 매칭됨 (버그)

### 파일 구조

```
/home/servermanager/pers/
├── server.py                    # 백엔드 API + VCM 생성
├── index.html                   # 프론트엔드 UI
├── .env                         # API 키 (DART, Gemini)
├── dart_financial_extractor.py  # 재무제표 추출 모듈 (복잡)
├── dart_company_info.py         # 기업개황정보 전용 모듈 (경량)
├── financial_insight_analyzer.py # AI 인사이트 분석 모듈
├── output/                      # 생성된 Excel 파일
└── CLAUDE.md                    # 이 파일
```

---

## 기업개황정보 기능

### 개요
DART API에서 기업개황정보만 따로 가져오는 경량 기능. 재무제표 추출 전에 회사 기본 정보를 빠르게 확인할 수 있음.

### 데이터 흐름

1. **프론트엔드 (index.html)**
   - 회사 검색 → 회사 클릭 시 `selectCompany()` 호출
   - `loadCompanyInfo(corpCode)` 함수로 API 호출
   - `#companyInfoWrapper` (독립 섹션)에 테이블 렌더링
   - `companyInfoData` 전역 변수에 저장 → 추출 시 서버로 전송

2. **백엔드 (server.py)**
   - `GET /api/company-info/{corp_code}` 엔드포인트로 조회
   - `POST /api/extract` 요청 시 `company_info` 필드로 전달받음
   - `save_to_excel()` 함수에서 "기업개황" 시트로 저장

### UI 구조
- `#companyInfoWrapper`: 독립 섹션 (progressSection 밖)
- 회사 선택 시 표시, 추출 진행/완료 중에도 **계속 유지**
- "새로 검색" 시에만 숨김

### 엑셀 시트 순서
1. **기업개황** ← 기업개황정보 (신규)
2. 재무상태표
3. 손익계산서
4. 포괄손익계산서
5. 현금흐름표
6. 주석들 (손익주석, 재무주석, 현금주석)
7. VCM전용포맷
8. 복사용테이블

### API 응답 필드

| 필드 | 설명 |
|------|------|
| corp_code | DART 고유번호 |
| corp_name | 회사명 |
| corp_name_eng | 영문 회사명 |
| stock_code | 종목코드 (상장사만) |
| ceo_nm | 대표자명 |
| market_name | 시장구분 (코스피/코스닥/코넥스/기타) |
| jurir_no | 법인번호 |
| bizr_no | 사업자번호 |
| adres | 주소 |
| hm_url | 홈페이지 |
| phn_no | 전화번호 |
| induty_code | 업종코드 |
| est_dt | 설립일 (YYYYMMDD) |
| est_dt_formatted | 설립일 (YYYY-MM-DD) |
| acc_mt | 결산월 |
| acc_mt_formatted | 결산월 (예: 12월) |

### 전용 스크립트 (dart_company_info.py)

기존 `dart_financial_extractor.py`가 재무제표 추출에 특화되어 무거우므로, 기업개황정보만 빠르게 조회하는 경량 스크립트.

```python
from dart_company_info import DartCompanyInfo

client = DartCompanyInfo()

# 회사 검색
results = client.search("삼성전자")

# 기업개황정보 조회
info = client.get_info(results[0]['corp_code'])
print(info.corp_name, info.ceo_nm)

# 종목코드로 조회 (상장사)
info = client.get_info_by_stock_code("005930")
```

---

## AI 인사이트 분석 기능

### 개요
LLM을 활용하여 재무제표의 이상 패턴을 감지하고, 웹 검색을 통해 원인을 파악하여 M&A 실사용 인사이트를 생성합니다.

### 아키텍처
```
[재무 데이터 + 기업개황]
         ↓
[LLM 0: 업종 파악] (Gemini Flash + Search)
         ↓
[LLM 1: 이상 감지] (Gemini Pro)
         ↓
[LLM 2: 검색 태스크 생성] (Gemini Pro)
         ↓
[병렬 검색 에이전트들] (Gemini Flash + Search)
├── 기업 특정 검색
├── 산업 동향 검색
├── 거시경제 검색
└── 경쟁사 비교 검색
         ↓
[LLM 3: 종합 보고서 생성] (Gemini Pro)
```

### 관련 파일
- `financial_insight_analyzer.py`: AI 분석 메인 스크립트
- `.env`: Gemini API 키 (`GEMINI_API_KEY`)

### API 엔드포인트
- `POST /api/analyze/{task_id}`: 분석 시작
- `GET /api/analyze-status/{task_id}`: 분석 상태 조회

### UI
- 추출 완료 후 "AI 인사이트 분석" 버튼 표시
- `#analysisSection`: 분석 결과 표시 영역
- 마크다운 형식의 보고서를 HTML로 렌더링

### 감지 대상 (M&A 실사 관점)
- 전년 대비 급격한 변동 (±20% 이상)
- 흑자 ↔ 적자 전환
- 영업이익 vs 당기순이익 괴리 (일회성 비용/수익)
- 매출 성장 vs 이익률 불일치
- 차입금/부채 급변
- 운전자본(NWC) 이상 변동
- 영업외비용 급증 (과징금, 소송 등)

---

## 서버/Nginx 포트 규칙

### 핵심 원칙
**서버는 항상 포트 8000에서 실행된다. Nginx도 반드시 8000으로 프록시해야 한다.**

### 포트 설정
- **서버 (server.py)**: `uvicorn.run(app, host="0.0.0.0", port=8000)`
- **Nginx**: `proxy_pass http://localhost:8000;`

### 502 Bad Gateway 발생 시
nginx가 잘못된 포트(예: 8080)로 프록시 중일 가능성이 높음.

```bash
# 확인
sudo grep proxy_pass /etc/nginx/sites-available/pers.moatai.app

# 수정 (8080 → 8000)
sudo sed -i 's/8080/8000/g' /etc/nginx/sites-available/pers.moatai.app
sudo nginx -t && sudo systemctl reload nginx
```

### 서버 시작 명령어
```bash
# 방법 1: uvicorn 직접 실행
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# 방법 2: python 실행 (server.py 내부에서 uvicorn 호출)
nohup python3 server.py > /tmp/server.log 2>&1 &
```

### 상세 문서
`/home/servermanager/pers/SERVER_SETUP.md` 참조
