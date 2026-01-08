# PERS 프로젝트 개발 규칙

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
├── server.py          # 백엔드 API + VCM 생성
├── index.html         # 프론트엔드 UI
├── output/            # 생성된 Excel 파일
└── CLAUDE.md          # 이 파일
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
