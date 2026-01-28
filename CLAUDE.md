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
- **시스템이 알아야 할 중요한 이슈나 문제점**
- **기존 동작 방식에 큰 변화가 있는 경우**

### 서버 재시작 규칙 (★필수★)
**server.py, database.py 등 백엔드 파일을 수정한 후에는 반드시 서버를 재시작해야 변경사항이 적용된다.**

```bash
pkill -9 -f "uvicorn server:app"
cd /home/servermanager/pers
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

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
     - `Frontdata`: 메타데이터 포함 (항목, 타입, 부모, FY연도...)
     - `Financials`: 프론트 표시용 (항목, FY연도... 단위: 백만원)
   - `preview_data['vcm']`, `preview_data['vcm_display']`로 API 응답에 포함

2. **프론트엔드 (index.html)**
   - `vcm_display` 데이터가 있으면 **그대로 렌더링** (단순화된 방식)
   - `vcm` 메타데이터에서 타입 정보 참조 (category, subitem, total, highlight)
   - 빈 행은 백엔드에서 필터링되어 프론트에 전달되지 않음

### 엑셀 시트 순서
1. 기업개황
2. Financials (표시용)
3. 재무분석 AI (분석 후 추가)
4. 재무상태표
5. 손익계산서
6. 현금흐름표
7. 주석들...
8. Frontdata (메타데이터) ← 맨 끝

---

## 데이터-엑셀 동기화 규칙 (★필수★)

### 핵심 원칙
**프론트엔드/API에 데이터 필드를 추가하거나 변경할 때, 엑셀 저장 함수도 함께 수정해야 한다.**
**UI 텍스트(탭명, 버튼명 등)를 변경할 때, 해당하는 엑셀 시트명도 반드시 함께 변경해야 한다.**

### 동기화 대상
- 기업개황정보 필드 추가/변경 → `save_to_excel()` 함수의 "기업개황" 시트
- 재무제표 데이터 필드 추가 → 해당 시트 저장 로직
- VCM 포맷 변경 → `create_vcm_format()` 함수
- **UI 탭명/메뉴명 변경 → 엑셀 시트명도 동일하게 변경**

### UI-엑셀 시트명 매핑
| UI 탭/메뉴명 | 엑셀 시트명 | 설명 |
|-------------|------------|------|
| Financials | Financials | 표시용 (단위: 백만원) |
| 재무분석 AI | 재무분석 AI | AI 분석 보고서 |
| - | Frontdata | 메타데이터 (타입, 부모 등) |

### 체크리스트
새 데이터 필드 추가 시:
1. ☐ API 응답에 필드 추가
2. ☐ 프론트엔드 UI에 표시
3. ☐ **엑셀 저장 함수에 추가** ← 누락하기 쉬움!

UI 텍스트 변경 시:
1. ☐ index.html UI 텍스트 변경
2. ☐ **server.py 엑셀 시트명 변경** ← 누락하기 쉬움!

### 관련 코드 위치
- 엑셀 저장: `server.py` 내 `save_to_excel()` 함수
- 기업개황 시트: `info_rows` 리스트 (line 4323 부근)

### 참고
- 애매한 경우 사용자에게 "엑셀에도 저장해야 할까요?" 질문하기
- 예시: 업종명(induty_name) 추가 시 프론트엔드에만 표시하고 엑셀 누락 → 버그

---

## VCM 표시 규칙 (★중요★)

### 핵심 원칙
**프론트엔드는 Financials 시트(표시용)를 그대로 렌더링한다. 모든 필터링/그룹핑은 백엔드에서 처리.**

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

### Financials 시트 생성 규칙
- 단위: **백만원** (백엔드에서 변환 완료, 정수로 반올림)
- 숫자 포맷: 천 단위 콤마 (예: 1,234)
- 빈 행: **제외** (모든 연도에 값이 없는 행)
- 타입별 스타일링은 프론트엔드에서 Frontdata 메타데이터 참조

### 엑셀-프론트엔드 데이터 연결 (★핵심 원칙★)

**원칙: Financials 시트가 기준! 프론트는 이 숫자를 그대로 표시!**

```
[1단계] 백엔드 (server.py create_vcm_format)
    ├── Frontdata 시트: 원 단위 숫자 + 메타데이터(타입, 부모)
    └── Financials 시트: 백만원 정수 (이 숫자가 기준!)

[2단계] API 응답
    ├── vcm (= Frontdata): 원 단위 - 툴팁 세부항목용
    └── vcm_display (= Financials): 백만원 정수 - 테이블 표시용

[3단계] 프론트엔드 (index.html)
    ├── 테이블 셀: vcm_display 값을 그대로 표시 (변환 금지!)
    └── 툴팁만: vcm(원 단위) → 백만원 변환 (100만 이상이면 /1000000)
```

**주의사항:**
- 프론트 테이블 숫자와 엑셀 Financials 시트 숫자가 반드시 일치해야 함
- 테이블 값에 추가 변환/계산 절대 금지
- 툴팁 데이터(getChildItems)는 vcm에서 가져오고, 표시 시 백만원 변환

**코드 위치:**
- 백엔드 변환: `server.py` 4498-4507줄 (`create_vcm_format` 함수)
- 프론트 툴팁 변환: `index.html` 4288-4297줄 (BS), 4393-4405줄 (IS), 4352-4372줄 (EBITDA)
- 툴팁용 데이터 조회: `index.html` getChildItems 함수 (4186-4203줄)

### Financials 시트 필터링 규칙

**표시되는 항목:**
1. `부모` 없는 항목 → 표시 (메인 항목)
2. BS/IS 섹션 하위항목 → 표시 (들여쓰기):
   - BS: `부모='유동자산'`, `'비유동자산'`, `'유동부채'`, `'비유동부채'`, `'자본'`
   - IS: `부모='매출'`, `'매출원가'`, `'판매비와관리비'`, `'영업외수익'`, `'영업외비용'`

**제외되는 항목 (툴팁용):**
- 기타XXX 하위항목 (예: 기타유동자산 하위 항목들)
- NWC, Net Debt 하위항목 ([NWC], [NetDebt] 접미사 포함)

**코드 위치:** `server.py` 내 `create_vcm_format()` 함수의 Financials 시트 생성 부분

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
├── database.py                  # SQLite DB 모듈 (사용자/사용량 관리)
├── financial_data.db            # SQLite 데이터베이스 파일
├── index.html                   # 프론트엔드 UI
├── .env                         # API 키 (DART, Gemini)
├── dart_financial_extractor.py  # 재무제표 추출 모듈 (복잡)
├── dart_company_info.py         # 기업개황정보 전용 모듈 (경량)
├── financial_insight_analyzer.py # 재무분석 AI 분석 모듈
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
7. Financials
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

## 재무분석 AI 분석 기능

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
- 추출 완료 후 "재무분석 AI 분석" 버튼 표시
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

## 사용자 인증 및 사용량 관리

### 개요
SQLite 기반의 사용자 인증 및 사용량 추적 시스템. 회원가입/로그인, 추출/AI 사용량 기록.

### 데이터베이스 스키마 (database.py)

| 테이블 | 용도 |
|--------|------|
| users | 사용자 계정 (이메일, 비밀번호 해시, 등급, 사용량 한도) |
| sessions | 세션 토큰 관리 |
| login_history | 로그인 기록 (IP, User-Agent) |
| search_history | 기업 검색 기록 |
| extraction_history | 재무제표 추출 기록 |
| llm_usage | AI 분석 사용량 (모델, 토큰, 비용) |

### 사용자 등급 (tier)

| 등급 | 검색 한도 | 추출 한도 | AI 한도 | 비고 |
|------|----------|----------|--------|------|
| free | 10회/월 | 5회/월 | 3회/월 | 기본값 |
| basic | 100회/월 | 50회/월 | 20회/월 | 유료 |
| pro | 무제한 | 무제한 | 100회/월 | 유료 |

**결제**: 계좌이체 후 관리자가 수동으로 등급 변경

### 세션 정책
- **로그인 유지**: 브라우저 세션 쿠키 (브라우저 닫으면 자동 로그아웃)
- **쿠키 설정**: `httponly=True`, `samesite=lax`, `max_age=None`

### API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/auth/register` | 회원가입 |
| POST | `/api/auth/login` | 로그인 (세션 쿠키 발급) |
| POST | `/api/auth/logout` | 로그아웃 (세션 삭제) |
| GET | `/api/auth/me` | 현재 사용자 정보 + 사용량 통계 |
| GET | `/api/admin/users` | 전체 사용자 조회 (관리자 전용) |

### 인증 헬퍼 함수

```python
# 선택적 인증 (없으면 None)
user = Depends(get_current_user)

# 필수 인증 (없으면 401)
user = Depends(require_auth)

# 관리자 필수 (없으면 403)
admin = Depends(require_admin)
```

### 사용량 로깅 함수 (database.py)

```python
# 기업 검색 기록 + 검색 횟수 증가
db.log_search(user_id, corp_code, corp_name, market)

# 추출 기록 + 추출 횟수 증가
db.log_extraction(user_id, corp_code, corp_name, start_year, end_year, file_path)

# AI 분석 기록 + AI 사용 횟수 증가
db.log_llm_usage(user_id, corp_code, corp_name, model_name, input_tokens, output_tokens, cost)
```

### 관리자 계정 생성

```bash
# database.py 직접 실행
cd /home/servermanager/pers
python3 database.py  # admin@example.com / admin123 생성

# 또는 코드로 생성
from database import create_user
create_user('admin@mysite.com', 'secure_password', role='admin', tier='pro')
```

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
