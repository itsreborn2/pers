# VCM v2 배치 검증 테스트 체크리스트

## 개요

VCM v1(하드코딩 키워드 매칭)과 v2(LLM 계정 분류) 결과를 비교하여 정확도를 검증하는 자동화 테스트.

## 테스트 스크립트

```bash
# 스크립트 위치
/home/servermanager/pers-dev2/test_vcm_v2_batch.py

# 기업 목록
/home/servermanager/pers-dev/test_companies.json
```

## 실행 방법

```bash
cd /home/servermanager/pers-dev2

# 기본 (standard + edge_cases + bottom50 = 40개)
PYTHONUNBUFFERED=1 python3 -u test_vcm_v2_batch.py

# 특정 배치만
PYTHONUNBUFFERED=1 python3 -u test_vcm_v2_batch.py --category listed.small_cap_batch8

# 여러 배치 동시에 (콤마 구분)
PYTHONUNBUFFERED=1 python3 -u test_vcm_v2_batch.py --category listed.small_cap_batch8,listed.small_cap_batch9,listed.small_cap_batch10

# 전체 (소형주 배치 포함)
PYTHONUNBUFFERED=1 python3 -u test_vcm_v2_batch.py --all

# 특정 기업만
PYTHONUNBUFFERED=1 python3 -u test_vcm_v2_batch.py --names 삼성전자,E1

# 동시 실행 수 조절 (★ --parallel 1 필수, 2 이상은 메모리 타임아웃 발생)
PYTHONUNBUFFERED=1 python3 -u test_vcm_v2_batch.py --parallel 1
```

## 사전 조건

1. **dev2 서버 실행 중** (port 8002)
2. **메모리 확인** — 서버 RSS < 500MB (테스트 전 `free -m` 확인, 여유 1GB+ 필요)
3. **서버 신규 시작 권장** — 이전 작업으로 메모리 누적 방지

```bash
# 서버 재시작
kill $(lsof -i :8002 -t 2>/dev/null) 2>/dev/null
sleep 3
nohup python3 server.py > /tmp/pers-dev2-server.log 2>&1 &
sleep 5
curl -s http://localhost:8002/ | head -1  # 확인
free -m | grep Mem  # 메모리 확인
```

## 테스트 단계별 상세

### Step 1: DART 추출 (Extract)

| 항목 | 설명 |
|------|------|
| API | `POST /api/extract` |
| 입력 | `corp_code`, `corp_name`, `start_year=2023`, `end_year=2024` |
| 타임아웃 | 360초 (5초 간격 72회 폴링) |
| 성공 조건 | `status == 'completed'` |
| 실패 유형 | `extract_fail` (task_id 미반환), `extract_error` (DART 오류), `extract_timeout` |

**주요 실패 원인:**
- `NotFoundConsolidated`: 연결재무제표 없는 기업 (소형주에 빈번)
- DART API 타임아웃: 서버 과부하
- Guest limit: `guest_usage` 테이블 초과 시

### Step 2: VCM v2 실행 (LLM 분류)

| 항목 | 설명 |
|------|------|
| API | `POST /api/vcm-v2/{task_id}` |
| 타임아웃 | 300초 |
| 성공 조건 | `success == true` |
| 출력 | `display_v2` (Financials 형식 데이터) |

**캐시 동작:** 동일 기업 재실행 시 DB 캐시 히트 → LLM 호출 없이 즉시 완료 (0.4~1s)

### Step 3: V1 vs V2 비교 (Compare)

| 항목 | 설명 |
|------|------|
| API | `POST /api/vcm-compare/{task_id}` |
| 비교 대상 | V1(`vcm_display`)과 V2(`display_v2`)의 모든 셀 |
| 일치 기준 | 수치 차이 ±1 이내 |
| 출력 | `match_rate`, `diffs` (불일치 목록) |

### Step 4: 주요 지표 비교 (7개 핵심 항목)

최신 FY 기준, V1 값과 V2 값을 비교:

| # | 항목 | 허용 오차 | 설명 |
|---|------|----------|------|
| 1 | **자산총계** | ±1 | BS 합계 |
| 2 | **부채총계** | ±1 | BS 합계 |
| 3 | **자본총계** | ±1 | BS 합계 |
| 4 | **매출** | ±1 | IS 최상위 |
| 5 | **영업이익** | ±1 | IS 핵심 |
| 6 | **당기순이익** | ±1 | IS 최종 |
| 7 | **매출총이익** | ±1 | IS 중간 합계 |

**판정 기준:**
- `✓` (Match): 7개 모두 ±1 이내
- `△` (Mismatch): 1개 이상 차이 → 원인 조사 필요
- V1=0인 경우 mismatch에서 제외 (V1 추출 실패한 항목)

### Step 5: 산술 검증 (V2 데이터 자체 검증)

| # | 검증 항목 | 수식 | 허용 오차 |
|---|----------|------|----------|
| 1 | **자산 등식** (asset_eq) | 자산총계 = 유동자산 + 비유동자산 | ±2 |
| 2 | **대차 균형** (balance) | 부채와자본총계 = 부채총계 + 자본총계 | ±2 |

### Step 6: 엑셀 파일 검증

엑셀 다운로드 및 시트/데이터 정합성 검증. (`test_full_pipeline.py` 에서 수행)

| # | 검증 항목 | 설명 |
|---|----------|------|
| 1 | **엑셀 다운로드** | `GET /api/download/{task_id}` → 파일 수신, 크기 > 0 |
| 2 | **필수 시트 존재** | `Financials`, `Frontdata` 시트 존재 |
| 3 | **선택 시트 확인** | `기업개황` (company_info 전달 시), `재무분석 AI` (분석 후) |
| 4 | **Financials 행 수** | 10행 이상 데이터 존재 |
| 5 | **Frontdata 행 수** | 10행 이상 데이터 존재 |
| 6 | **Financials-프론트 일치** | 엑셀 수치 = API `vcm_display` 수치 (백만원 단위) |
| 7 | **Frontdata 부모-자식 정합성** | 하위 항목의 `부모` 필드가 유효한 항목을 참조 |
| 8 | **BS 시트 존재** | 재무상태표 시트 데이터 확인 |
| 9 | **IS 시트 존재** | 손익계산서 시트 데이터 확인 |
| 10 | **CF 시트 존재** | 현금흐름표 시트 데이터 확인 |

**실행 방법:**
```bash
cd /home/servermanager/pers-dev2
PYTHONUNBUFFERED=1 python3 -u test_full_pipeline.py                    # 기본 (추출+엑셀)
PYTHONUNBUFFERED=1 python3 -u test_full_pipeline.py --full             # 전체 (AI분석+리서치 포함)
PYTHONUNBUFFERED=1 python3 -u test_full_pipeline.py --company E1 --corp-code 00356361
```

### Step 7: 재무분석 AI 검증

| # | 검증 항목 | 설명 |
|---|----------|------|
| 1 | **분석 시작** | `POST /api/analyze/{task_id}` → 정상 응답 |
| 2 | **분석 완료** | 상태 폴링 → `completed` (타임아웃 600초) |
| 3 | **보고서 생성** | `report` 필드에 마크다운 보고서 존재 |
| 4 | **엑셀 AI 시트 추가** | `POST /api/add-insight/{task_id}` → 성공 |
| 5 | **AI 시트 존재** | 엑셀에 `재무분석 AI` 시트 존재 |
| 6 | **수치 검증** | `_validate_report_numbers()` 통과 (보고서 내 수치 = 원본 데이터 ±5%) |
| 7 | **요약 검증** | `_validate_and_fix_summary()` 통과 (요약 내 수치 = 보고서 수치) |

**검증 대상 (AI 보고서 품질):**
- 숫자 할루시네이션 없음 (LLM이 만든 수치 vs 원본 데이터 비교)
- VCM/IS 혼동 없음 (집계 항목 vs 개별 항목 구분)
- 연도/컨텍스트 정확 (FY2024 데이터를 FY2023으로 오기재 없음)
- 단일 연도 데이터 시 YoY 비교 미수행

**실행 방법:**
```bash
# AI 분석 포함 전체 테스트
PYTHONUNBUFFERED=1 python3 -u test_full_pipeline.py --full
```

**주의:** AI 분석은 Gemini API 호출이 많아 비용 발생. 배치 테스트에는 포함하지 않고, 주요 수정 후 개별 기업으로 검증.

## 결과 판정 기준

### 기업별 판정

| 상태 | 의미 | 조치 |
|------|------|------|
| `success` + 7/7 Match | V2가 V1과 동일 | 정상 |
| `success` + Mismatch | V2와 V1 차이 있음 | 원인 조사 필요 (아래 분류) |
| `extract_fail` | DART 추출 시작 실패 | 인프라 문제 |
| `extract_error` | DART 추출 중 에러 | 보통 NotFoundConsolidated |
| `extract_timeout` | 360초 초과 | 서버 메모리/DART 과부하 |
| `v2_fail` | LLM 분류 실패 | 프롬프트/모델 문제 |
| `compare_fail` | 비교 실패 | V1/V2 데이터 구조 불일치 |

### Mismatch 분류

| 분류 | 의미 | 예시 |
|------|------|------|
| **V2>V1** | V2가 더 정확 | 소룩스 (지배주주 귀속), 스코넥 (중단사업 포함) |
| **V1>V2** | V1이 더 정확 | V2 LLM 분류 오류 → 프롬프트 수정 필요 |
| **Minor Diff** | 경미한 차이 (5% 미만) | 반올림, 비지배지분 등 |
| **Investigate** | 원인 불명 | 추가 조사 필요 |

## 배치 카테고리

```
listed.standard          —  3개 (삼성전자, E1, 아이티엠반도체)
listed.edge_cases        —  7개 (특수 IS/BS 구조)
listed.bottom50_small_cap — 30개 (시총 하위 소형주)
listed.small_cap_batch2  — 46개
listed.small_cap_batch3  — 39개
listed.small_cap_batch4~26 — 각 19~21개 (총 400+개)
```

## 주의사항

### 메모리 관리 (★필수★)
- 서버 RAM 3.8GB, prod(35MB) + dev(63MB) + dev2(~750MB) 상시 점유
- **`--parallel 1` 필수** — 동시 추출 2개 이상 시 메모리 부족으로 대부분 타임아웃
- `--parallel 1` 기준 기업당 약 2~3분 (추출 40~80초 + V2 80~120초)
- 배치 1개(20개) 약 40~60분 소요
- **배치 2~3개 단위로 실행**, 사이사이 서버 재시작 권장
- 서버 RSS 2GB 초과 시 재시작 (`kill $(lsof -i :8002 -t)`)

```bash
# 권장 실행 방법 (parallel 1 필수)
PYTHONUNBUFFERED=1 python3 -u test_vcm_v2_batch.py --category listed.small_cap_batch15 --parallel 1

# 메모리 모니터링
free -m | grep Mem
ps aux | grep server.py | grep -v grep | awk '{printf "RSS:%.0fMB\n", $6/1024}'
```

### 소형주 실패율
- `small_cap_batch5` 이후: NotFoundConsolidated 비율 50%+
- 이는 V2 문제가 아닌 DART 데이터 부재 (별도재무제표만 있는 기업)

### 결과 파일
- 테스트 완료 시 `batch_test_results_YYYYMMDD_HHMMSS.json` 자동 생성
- 상세 비교 데이터 포함 (기업별 v1/v2 값, 차이, 산술 검증)

## 누적 테스트 결과 (2026-03-15 기준)

| 배치 | 기업 수 | 추출 성공 | 7/7 Match | V2>V1 | Investigate |
|------|--------|----------|-----------|-------|------------|
| batch1-3 (standard+edge+bottom50+batch2+3) | 125 | 48 | 39 | 5 | 1 |
| batch4-7 | 81 | 20 | 18 | 3 | 2 |
| batch8-10 | 60 | 22 | 20 | 1 | 1 |
| batch11-13 | 59 | 22 | 18 | 1 | 0 |
| batch14 (부분) | 20 | 3 | 2 | 1 | 0 |
| batch15 | 20 | 20 | 16 | 0 | 4 |
| batch16 | 20 | 20 | 15 | 1 | 4 |
| batch17 | 19 | 18 | 17 | 0 | 1 |
| batch18 | 20 | 20 | 18 | 1 | 1 |
| batch19 | 20 | 19 | 17 | 0 | 2 |
| batch20 | 20 | 20 | 17 | 0 | 3 |
| batch21 | 20 | 20 | 19 | 0 | 1 |
| batch22 | 19 | 19 | 16 | 0 | 3 |
| batch23 | 20 | 20 | 17 | 2 | 1 |
| batch24 | 20 | 19 | 17 | 2 | 0 |
| batch25-26 | 미테스트 | - | - | - | - |
| **누적** | **543** | **313** | **268** | **20** | **24** |

### 확인된 V2>V1 케이스
1. 스코넥: 중단사업 당기순손익 포함 (V1 누락)
2. 소룩스: 지배주주 귀속 당기순이익 사용 (PE 관점 적절)
3. 다산솔루에타: DART 원본값 직접 사용
4. 앱코/피코그램/심플랫폼/티에스이: V1 오매칭, V2 정확히 0 반환
5. 자이언트스텝/바이오인프라: V1 VCM 빈값, V2 정상
6. 동성제약: V1 법인세비용 부호 오류 (EBT-Tax 계산 시 세금환급 잘못 처리)
7. 벡트: V1 IS 주석 데이터 오매칭, V2는 CIS 폴백으로 정확한 값 사용
8. 이노시뮬레이션: V1 IS 주석의 "매출원가 및 판매비와관리비 합계"를 매출원가로 오매칭 (14,150M), V2는 주석 필터링으로 정확히 None 반환
9. 뉴키즈온: V1 BS 자산총계=0 (BS 못 읽음), V2 자산총계=45,807 정상 추출
10. 엠브레인: V1 영업이익/당기순이익=0 (IS 못 읽음), V2 영업이익=-220,512, 당기순이익=-252,892 정상 추출
11. 라메디텍: V1 BS=0 (IS에 FY 컬럼 없어 조기종료), V2는 BS+IS FY 합집합으로 BS 정상 추출 (자산22,346)
12. KS인더스트리: V1 중단영업 구조에서 XBRL 파싱 실패→HTML 오인식 (매출85), V2는 DART CFS와 완벽 일치 (매출24,687)
13. 티엔엔터테인먼트: V1 IS=0, V2 매출=71,907, 영업이익=3,250 정상 추출
14. 케일럼: V1 전체=0 (BS+IS), V2 자산총계=128,870, 매출=86,003 정상 추출

### 확인된 DART 데이터 부재 케이스
- 남성: IS/CIS 모두 매출/영업이익 항목 없음
- 디지캡: FY2024 전체 None (아직 공시 안 됨)
- 앱코(batch23): IS DataFrame에 FY값 없음 (DART 추출 문제). V1 매출=1,062는 이자수익 오인식. V2=0이 정당

### 확인된 수정사항
1. `get_cat_first` sign preference (삼영에스앤씨 sign flip 수정)
2. `sign_applicable_cats` 코드 레벨 제한 (other_expense sign 오적용 방지)
3. `account_classifier.py` 프롬프트 sign 규칙 명확화
4. `create_vcm_format_v2` IS→CIS 폴백 추가: IS에 매출/영업수익 없으면 CIS 사용 (벡트 등)
5. `create_vcm_format_v2` IS 주석 필터링: 분류1 기반으로 주석(notes) 데이터 제외, 당기순이익은 주석에서 복구 (이노시뮬레이션 등)
6. BS overflow 이름 충돌 수정 (2026-03-16):
   - `len(items) > 1` → `>= 1`: 1개 항목 그룹의 툴팁 하위 표시
   - overflow 하위/selected 하위 이름 충돌 → `(개별)` 접미사
   - overflow 컨테이너와 selected 이름 충돌 → `(합산)` 접미사
   - master_order 첫 연도 dedup: 동일 이름 중복 row 방지
7. Bug #5: 다중 연도 BS overflow 불일치 수정 (2026-03-16):
   - 연도별 독립 selected/overflow 결정 → 전체 연도 max abs 기준 글로벌 결정
   - Pre-scan: 전체 연도 display_candidates 수집 → 항목별 max abs 계산
   - Global selection: 섹션별 max abs 상위 MAX_ITEMS개 결정
   - 모든 연도에서 동일한 selected/overflow → 부모 충돌 방지

---

## BS Overflow 검증 체크리스트 (★필수★)

### 개요

BS 각 섹션(유동자산/비유동자산/유동부채/비유동부채/자본)에서 항목이 MAX_ITEMS=6개를 초과하면 나머지가 "기타{섹션명}" 컨테이너로 합산됩니다. 이 과정에서 다양한 엣지케이스가 존재합니다.

### 검증 항목

| # | 검증 항목 | 설명 | 확인 방법 |
|---|----------|------|----------|
| 1 | **섹션별 합계 일치** | 각 섹션 헤더값 = 직접 하위항목 합계 | vcm에서 부모=섹션인 항목 합 vs 섹션 헤더 |
| 2 | **이름 중복 없음** | vcm 전체에서 동일 항목명이 2개 이상 없어야 함 | Counter(항목명) 중 v > 1 확인 |
| 3 | **기타 컨테이너 정합성** | 기타{섹션} 컨테이너의 값 = 하위항목 합계 + (세부) 잔여분 | 기타{섹션} 하위 합계 검증 |
| 4 | **이름 충돌 방지** | (개별)/(합산) 접미사가 올바르게 적용 | DART 계정 "기타비유동자산" 등이 충돌 시 접미사 확인 |
| 5 | **1개 항목 그룹 툴팁** | is_group=True이고 items=1인 그룹도 하위항목 생성 | Frontdata에서 해당 그룹의 하위 존재 확인 |
| 6 | **Financials 필터링** | overflow 하위항목은 Financials 시트에서 제외, Frontdata에만 존재 | vcm_display에 기타 하위 미포함 확인 |
| 7 | **자산총계 = 유동+비유동** | 자산총계가 유동자산 + 비유동자산과 일치 | ±2 오차 허용 |
| 8 | **부채와자본 = 부채+자본** | 대차 균형 | ±2 오차 허용 |

### 이름 충돌 시나리오

| 시나리오 | 조건 | 기대 결과 |
|----------|------|----------|
| **A: overflow 자식-컨테이너 충돌** | DART 계정 "기타비유동자산"이 overflow에 포함 | 자식 → "기타비유동자산(개별)", 컨테이너 유지 |
| **B: selected-컨테이너 충돌** | DART 계정 "기타비유동자산"이 selected(상위6)에 포함 | 컨테이너 → "기타비유동자산(합산)" |
| **C: selected 자식-부모 충돌** | 그룹 내 항목명이 그룹명과 동일 | 자식 → "{이름}(개별)" |
| **D: master_order 중복** | 같은 이름이 bs_items에 2회 등장 | 첫 번째만 master_order에 유지 |

### 알려진 미해결 이슈

| 이슈 | 심각도 | 설명 | 상태 |
|------|--------|------|------|
| **다중 연도 overflow 불일치** | 높음 | FY2024에 7개(overflow), FY2023에 5개(no overflow) → 같은 항목 부모 불일치 | ✅ 수정완료 (2-pass global selection, 방안 D) |
| **양/음 상쇄 시 overflow 소실** | 중간 | overflow 합계=0이면 컨테이너 미생성 → 항목 소실 | 미수정 |
| **다른 섹션 동명 항목** | 중간 | "보증금"이 유동+비유동 양쪽에 있으면 값 덮어쓰기 | 미수정 |

### 검증 스크립트 (수동)

```python
# VCM v2 결과에서 overflow 검증
import requests
from collections import Counter

r = requests.post('http://localhost:8002/api/vcm-v2/{task_id}')
vcm = r.json()['vcm']

# 1. 섹션별 합계
for section in ['유동자산', '비유동자산', '유동부채', '비유동부채']:
    header = next((r for r in vcm if r.get('항목', '').strip() == section), None)
    if not header: continue
    fy = sorted([k for k in header if k.startswith('FY')], reverse=True)[0]
    hv = header.get(fy) or 0
    cs = sum((r.get(fy) or 0) for r in vcm if r.get('부모', '').strip() == section)
    print(f'{section}: 헤더={hv/1e6:.0f}M 하위={cs/1e6:.0f}M 차={abs(hv-cs)/1e6:.0f}M {"✓" if abs(hv-cs)<1e6 else "✗"}')

# 2. 이름 중복
names = [r.get('항목', '') for r in vcm]
dupes = {k: v for k, v in Counter(names).items() if v > 1}
print(f'이름 중복: {dupes if dupes else "없음 ✓"}')

# 3. 충돌 방지 접미사
special = [n for n in names if '(개별)' in n or '(합산)' in n]
print(f'충돌방지: {special if special else "없음"}')
```
