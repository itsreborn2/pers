# PERS 수정 히스토리

> **규칙**: 모든 수정 후 반드시 이 파일에 기록. 수정 전에는 반드시 이 히스토리를 참고하여 수정 방향을 결정.
> 과거 수정이 어떤 문제를 해결했고, 어떤 부작용을 야기했는지 확인 후 수정.

---

## 2026-03-21

### [FIX] V2 비유동차입부채 누락 수정 (동일계정명 유동/비유동 구분)
- **문제**: DART BS에서 '차입금', '리스부채' 등이 유동/비유동 양쪽에 동일 이름 존재 → V2 LLM 분류가 `current_liability`로만 처리 → 비유동차입부채=0 (CJ대한통운 ~2조원 GAP)
- **원인**: `bs_accounts = bs_df[acc_col].unique()`로 중복 제거 + LLM 캐시가 `account_name_raw` 기준 하나만 저장
- **수정**:
  - BS 데이터 섹션 순서 추적 (`_bs_section_map`)
  - 비유동 섹션 중복 항목에 `[비유동]` 접미사 추가
  - `get_value()`에서 접미사 항목의 비유동 행 값 반환
  - `_group_override_map`으로 유동차입부채→비유동차입부채, 기타금융부채→기타비유동금융부채 자동 매핑
- **영향 범위**: `server.py` create_vcm_format_v2() (line ~6650-6820)
- **검증**: CJ대한통운 비유동부채 GAP 81%→0%, Net Debt 404K→1,504K
- **회귀**: 삼성전자/E1/패스트파이브/한화에어로스페이스 전원 PASS
- **부작용**: 없음
- **관련 파일**: `server.py`

### [FIX] EBITDA Notes 상장사 감사보고서 검색 시도 → 원복
- **문제**: 상장사 XBRL에 Notes 없는 경우 EBITDA D&A=0
- **시도**: 상장사도 사업보고서(pblntf_ty='A')에서 주석 추출
- **부작용 발견**: 사업보고서 XBRL 주석의 단위(원)와 기존 HTML 주석(천원) 불일치 → 감가상각비 1000배 증폭
- **결정**: 원복. EBITDA Notes는 DART HTML 주석의 FY 커버리지 한계 (별도 이슈)
- **교훈**: **주석 데이터의 단위(원 vs 천원)는 소스마다 다름. 새 소스 추가 시 반드시 단위 검증 필요.**

### [DOC] 검증 체크리스트 V1→V2 회귀 검증 48항목 추가
- CLAUDE.md에 #15-48 검증 항목 추가
- V1 보장 BS/IS 필수 항목, EBITDA 구성요소, Net Debt 구성요소 등
- V2 수정 시 회귀 보호 원칙 명시

---

## 2026-03-19

### [FIX] IS 매출 감지 substring→exact 매칭 수정
- **문제**: CJ대한통운 매출=None (실제 약 11.7조원)
- **원인**: 7곳의 `'매출' in text` substring 매칭이 주석 데이터 '매출채권', '매출원가' 등과 오매칭
- **수정**: `{'매출', '매출액', '영업수익', '수익(매출액)'}` exact match + `endswith('매출액')`
- **부작용**: 없음 (19개 기업 테스트 통과)
- **교훈**: **substring 매칭은 주석 병합 후 false positive 발생 → exact match 필수**

### [FIX] BS/IS 탭 분리 버그 수정
- **문제**: IS 항목이 BS 탭에 표시됨 (CJ대한통운, 패스트파이브)
- **원인**: 프론트 BS/IS 분리가 `itemName === '매출'`에만 의존 → 매출=0이면 분리 불가
- **수정**:
  - 프론트: `isStartNames` 확장 + Net Debt 이후 첫 항목을 IS 시작으로 인식
  - 백엔드: IS 핵심 행 항상 유지 (값=0이어도 필터 안 함)
- **교훈**: **419개 기업 검증은 값 정확도만 테스트, 프론트 레이아웃은 미검증이었음**

### [FIX] CIS→IS Fallback Fix
- **문제**: XBRL IS 비어있을 때 CIS에 매출 있어도 HTML IS 폴백 사용 → IS 데이터 손실
- **수정**: HTML IS 폴백 전 CIS에 매출 확인 → CIS를 IS로 사용

### [FIX] 패스트파이브 HTML IS 폴백 실패
- **문제**: DART HTML 컬럼명 '당 기'/'전 기' (공백 포함) 미매칭
- **수정**: regex에 `\s*` 추가

### [FEAT] 기업 리서치 뉴스 검색 개선
- 25개 카테고리별 쿼리 → 3소스 병렬 (Firecrawl + Naver + Gemini)
- 결과: 1건→27~40건

---

## 2026-03-17

### [FEAT] V2 프로덕션 활성화
- V1→V2 전환: extract_financial_data, save_to_excel, run_financial_analysis
- **부작용 주의**: V2는 LLM 분류에 의존 → 분류 오류 시 데이터 누락 가능 (→ 2026-03-21 비유동차입부채 사례)

### [FIX] EBITDA Notes 감가상각비 사용
- SGA D&A → Notes "비용의 성격별 분류" 전체 D&A로 변경 (EBITDA 정확도 향상)
- 용마로지스: 19,445M→56,408M (AI chatbot 56,407M 일치)
- **주의**: Notes 미추출 시 SGA fallback → D&A=0 가능

### [FIX] 판관비 합계행 판별
- group=None 항목 중 '판매비' display_name 우선, 없으면 max(abs)
- 이전 버그: 사용권자산상각비(32M)가 마지막 group=None이라 판관비로 덮어씀

---

## 2026-03-16

### [FIX] BS Overflow Fixes (Bug #1-5, H1-H4)
- 다중 연도 overflow 불일치 → 2-pass global selection
- overflow 합계=0 조건 제거 (양/음 상쇄 방지)
- 동적 충돌 감지 — 동명 항목 `[섹션명]` 접미사
- **교훈**: overflow 로직은 전체 연도를 동시에 처리해야 연도 간 항목 불일치 방지

---

## 2026-03-14

### [TEST] VCM V2 LLM Classification 181개 기업 테스트
- 68/181 성공 (DART 실패 제외)
- Perfect match: 57 (84%), V2 better: 7 (10%)
- FY column union, sign preference 등 수정

---

## 2026-02-26

### [FEAT] Model upgrade gemini-3.1-pro-preview
- **부작용**: 서버 장애 발생 → gemini-2.5-pro로 롤백

### [FEAT] API key split
- `GEMINI_CHATBOT_API_KEY` 챗봇 전용 분리

### [FIX] Excel 동적 컬럼 너비
- 하드코딩 → 콘텐츠 기반 동적 (한글 2x, min 12, max 40)

---

## 2026-02-10

### [FIX] 재무분석 AI 6가지 오류 패턴 수정
1. **Solution A**: Pre-computed YoY (LLM 계산 대체)
2. **Solution B**: normalize_account_name() (계정명 정규화)
3. **Solution C-2**: VCM/IS 네임스페이스 태그
4. **Solution D**: _validate_report_numbers() 후처리 검증
5. **Solution D-2**: Summary hallucination fix
6. **P1/P2**: API timeout, 단일연도 처리, 입력 sanitize

---

## 2026-01-29

### [FIX] VCM 초기 버그 수정 다수
- 영업손실 양수 표시 → 괄호→음수 변환
- 자본잠식 자본총계 오류 → 괄호 처리 + excludes
- 비유동자산 소계 중복 → find_bs_val 우선
- 기타유동자산 툴팁 오류 → 원래 items 사용
- IS 항목 UI 미표시 → 좌우→상하 변환
- **교훈**: 본체 값 계산과 Frontdata 하위항목 추가를 반드시 동기화

---

## 미해결 이슈 (Open)

| 이슈 | 심각도 | 상태 | 설명 |
|------|--------|------|------|
| EBITDA Notes FY2023+ | HIGH | Open | DART HTML 주석의 FY 커버리지 한계. V2 FY 컬럼 매핑 개선 필요 |
| Normalized earnings bridge | P2 | Open | M&A critical |
| Auditor opinion extraction | P3 | Open | |
| YoY from pre-rounding values | P3 | Open | |
