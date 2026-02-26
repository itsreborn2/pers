# PERS 프로젝트 개발 규칙

## 개발/프로덕션 환경 분리 (★최우선★)

### 핵심 원칙
**모든 개발 작업은 개발 환경(pers-dev)에서 진행한다. 프로덕션 환경(pers)은 사용자 요청 시에만 배포한다.**

### 환경 구성

| 항목 | Production | Development |
|------|------------|-------------|
| 디렉토리 | `/home/servermanager/pers` | `/home/servermanager/pers-dev` |
| 브랜치 | `main` | `develop` |
| 포트 | 8000 | 8001 |
| URL | https://pers.moatai.app | https://pers-dev.moatai.app |
| systemd | `pers.service` | `pers-dev.service` |

### 서버 하드웨어 사양 (vm-docker-01)

| 항목 | 사양 | 비고 |
|------|------|------|
| CPU | Intel Xeon E5-2673 v4 @ 2.30GHz | 2코어 |
| RAM | 3.8GB | Swap 4GB 포함 |
| Disk | 29GB (16GB 가용) | Azure VM |
| OS | Ubuntu 22.04 LTS | - |

### 리소스 사용량 (참고)

| 프로세스 | RAM 사용량 | 비고 |
|----------|-----------|------|
| pers (프로덕션) | ~300MB | 분석 시 최대 500MB |
| pers-dev (개발) | ~110MB | 유휴 시 |
| 동시 운영 | ~500MB | 충분한 여유 있음 |

**주의:** RAM이 부족하면 Swap을 사용하므로 성능 저하 가능. 동시에 무거운 AI 분석 작업은 피할 것.

### 작업 흐름

```
[1] 개발 환경에서 작업
    cd /home/servermanager/pers-dev
    # 코드 수정...
    sudo systemctl restart pers-dev
    # https://pers-dev.moatai.app 에서 테스트

[2] 개발 완료 후 커밋
    git add . && git commit -m "feat: ..." && git push origin develop

[3] 사용자 요청 시 프로덕션 배포
    # GitHub에서 develop → main PR 생성 및 머지
    # 또는 로컬에서:
    cd /home/servermanager/pers
    git checkout main
    git merge develop
    git push origin main
    sudo systemctl restart pers
```

### 서버 관리 명령어

```bash
# 개발 서버
sudo systemctl status pers-dev    # 상태 확인
sudo systemctl restart pers-dev   # 재시작
sudo journalctl -u pers-dev -f    # 로그 실시간 확인

# 프로덕션 서버
sudo systemctl status pers        # 상태 확인
sudo systemctl restart pers       # 재시작
sudo journalctl -u pers -f        # 로그 실시간 확인
```

### 주의사항
- **개발 환경**: 자유롭게 수정/테스트 가능
- **프로덕션 환경**: 사용자 요청 없이 절대 수정 금지
- 두 환경은 **별도의 DB 파일** 사용 (각 디렉토리의 `financial_data.db`)
- `.env` 파일은 git에 포함되지 않으므로 수동 복사 필요

---

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
# 개발 환경 (일반적인 경우)
sudo systemctl restart pers-dev

# 프로덕션 환경 (배포 시에만)
sudo systemctl restart pers
```

### 통합 테스트 규칙 (★필수★)

**사용자가 "테스트해" 또는 "테스트 진행해"라고 하면, 반드시 아래 통합 테스트를 실행하라.**

#### 테스트 스크립트
```bash
# 기본 테스트 (DART 추출 + 데이터 검증 + 엑셀 다운로드 + heartbeat)
cd /home/servermanager/pers-dev
PYTHONUNBUFFERED=1 python3 -u test_full_pipeline.py

# 전체 테스트 (기본 + AI 분석 + 기업 리서치 + 최종 엑셀)
PYTHONUNBUFFERED=1 python3 -u test_full_pipeline.py --full

# 특정 기업으로 테스트
PYTHONUNBUFFERED=1 python3 -u test_full_pipeline.py --company E1 --corp-code 00356361

# 프로덕션 서버 테스트
PYTHONUNBUFFERED=1 python3 -u test_full_pipeline.py --url http://localhost:8000
```

#### 테스트 범위 (15+ 항목)
| 단계 | 테스트 항목 | 설명 |
|------|------------|------|
| 1 | 서버 상태 | GET / 200 OK |
| 2 | DART 추출 시작 | POST /api/extract → task_id |
| 2 | 추출 완료 | status 폴링 → completed (타임아웃 360초) |
| 3 | VCM 데이터 | vcm, vcm_display 존재 확인 |
| 3 | 재무상태표/손익계산서/현금흐름표 | BS, IS, CF 데이터 존재 확인 |
| 3 | 자산 등식 | 자산총계 = 유동자산 + 비유동자산 (±2) |
| 3 | 대차 균형 | 부채와자본총계 = 부채총계 + 자본총계 (±2) |
| 4 | AI 분석 (--full) | POST /api/analyze → 보고서 생성 |
| 4 | 엑셀 AI시트 추가 | POST /api/add-insight → 엑셀에 시트 추가 |
| 5 | 기업 리서치 (--full) | POST /api/super-research → 리서치 완료 |
| 6 | 엑셀 다운로드 | GET /api/download → 파일 수신 |
| 6 | 엑셀 시트 검증 | Financials, Frontdata 시트 존재 + 데이터 확인 |
| 7 | Heartbeat 정상 | 존재하는 task → success=true |
| 7 | Heartbeat 만료 감지 | 없는 task → expired=true |

#### 수정 후 테스트 의무
- **server.py 수정 시**: 서버 재시작 후 기본 테스트 필수
- **index.html 수정 시**: 서버 재시작 후 기본 테스트 필수
- **추출 로직 수정 시**: 기본 테스트 + 기존 검증 기업(삼성전자, E1) 추가 확인
- **AI 분석/리서치 수정 시**: `--full` 전체 테스트 필수
- **프로덕션 배포 시**: `--url http://localhost:8000` 프로덕션 테스트 권장

#### 테스트 실패 시
1. 실패 항목 확인 (로그에 [FAIL] 표시)
2. 원인 분석 후 수정
3. 재테스트하여 전체 통과 확인
4. **테스트 통과 전까지 프로덕션 배포 금지**

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

## 툴팁 데이터 동기화 규칙 (★최우선★)

### 핵심 원칙
**VCM 항목을 수정할 때는 반드시 "본체 값 계산"과 "Frontdata 하위항목 추가" 로직을 함께 수정해야 한다!**

### 시스템 구조 이해

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    server.py create_vcm_format()                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [로직 1] 본체 값 계산 (find_in_section, find_bs_val)                    │
│  ────────────────────────────────────────────────────────────────────── │
│  예: 매출채권및기타채권 = 매출채권 + 미수금 + 미수수익 + 선급금 + ...      │
│  → 이 값이 Financials 시트에 표시됨 → 웹 UI 테이블에 표시                │
│                                                                         │
│  [로직 2] Frontdata 하위항목 추가 (group_items_by_category)              │
│  ────────────────────────────────────────────────────────────────────── │
│  예: bs_items.append((item['name'], '매출채권및기타채권', item['value'])) │
│  → 이 값이 Frontdata 시트에 저장됨 → 웹 UI 툴팁에 표시                   │
│                                                                         │
│  ⚠️ 두 로직이 다른 소스에서 데이터를 가져오면 불일치 발생!               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 프론트엔드 데이터 흐름

```
┌─────────────────┐     ┌─────────────────┐
│  Financials     │     │   Frontdata     │
│  (vcm_display)  │     │   (vcm)         │
├─────────────────┤     ├─────────────────┤
│ 항목 | FY값     │     │ 항목 | 부모 | FY값│
├─────────────────┤     ├─────────────────┤
│매출채권및기타채권│     │ 미수금   │매출채│
│   5,042 (백만)  │     │   43 (원단위)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
   ┌──────────┐           ┌──────────┐
   │ 테이블 셀 │           │  툴팁    │
   │  5,042   │           │ 미수금 43 │
   │          │           │ 선급금 50 │
   │          │           │ 합계 ???  │
   └──────────┘           └──────────┘
         ↓                       ↓
    본체 값과 툴팁 합계가 다르면 ❌ 버그!
```

### 자주 발생하는 버그 패턴

**패턴 1: 본체 계산에 포함됐지만 Frontdata 하위에 없는 경우**
```python
# ❌ 잘못된 코드
매출채권및기타채권 = 매출채권 + 미수금 + 선급금  # 매출채권 포함해서 계산
# 하지만 parse_bs_sections에서 "매출채권" 단독 항목을 파싱 못함
# → Frontdata에 매출채권 하위항목 없음
# → 결과: 본체 5,042 vs 툴팁 합계 179 (불일치!)
```

**패턴 2: Frontdata에는 있지만 본체 계산에서 누락된 경우**
```python
# ❌ 잘못된 코드
기타자본구성요소 = 자본잉여금 + 기타포괄손익  # 자본조정 누락!
# 하지만 자본_items에 자본조정이 있으면 Frontdata 하위로 추가됨
# → 결과: 본체 100 vs 툴팁 합계 -1,142 (불일치!)
```

### 올바른 수정 방법

**원칙: 본체 값 계산에 사용된 각 항목을 명시적으로 Frontdata 하위로도 추가하라!**

**예시: 매출채권및기타채권 올바른 처리**
```python
# [로직 1] 본체 값 계산
매출채권 = find_in_section(유동자산_items, ['매출채권']) or 0
미수금 = find_in_section(유동자산_items, ['미수금']) or 0
미수수익 = find_in_section(유동자산_items, ['미수수익']) or 0
매출채권및기타채권 = 매출채권 + 미수금 + 미수수익  # 합산

bs_items.append(('매출채권및기타채권', '유동자산', 매출채권및기타채권))

# [로직 2] Frontdata 하위항목 추가 - 본체 계산에 사용된 항목들을 명시적으로 추가!
if 매출채권:
    bs_items.append(('매출채권', '매출채권및기타채권', 매출채권))
if 미수금:
    bs_items.append(('미수금', '매출채권및기타채권', 미수금))
if 미수수익:
    bs_items.append(('미수수익', '매출채권및기타채권', 미수수익))
```

### VCM 수정 체크리스트 (★매번 확인★)

항목(카테고리) 추가/수정 시:
1. ☐ **본체 값 계산 로직** 확인 (find_in_section, find_bs_val 등)
2. ☐ **Frontdata 하위항목 추가 로직** 확인 (bs_items.append)
3. ☐ 본체 계산에 포함된 모든 세부항목이 Frontdata 하위로 추가되는지 확인
4. ☐ group_items_by_category 사용 시, items 소스가 본체 계산 소스와 일치하는지 확인
5. ☐ **서버 재시작 후 웹에서 툴팁 검증** (본체 = 툴팁 합계)

### 관련 코드 위치 (server.py)

| 카테고리 | 본체 값 계산 | Frontdata 하위 추가 |
|----------|-------------|-------------------|
| 매출채권및기타채권 | 3911-3919줄 | 4238-4250줄 (group_items_by_category) |
| 기타자본구성요소 | 4040-4044줄 | 4392-4396줄 |
| 기타유동자산 | 4252-4263줄 | 4257-4263줄 |
| 기타비유동자산 | 4295-4304줄 | 4299-4304줄 |
| 유동차입부채 | 3955-3967줄 | 4309-4320줄 |
| NWC | 4057줄 | 4402-4404줄 |
| Net Debt | 4060줄 | 4413-4417줄 |

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

### 재무제표 추출 버그 수정 원칙 (★필수★)

**기존에 잘 작동하는 기업들이 있다고 가정하고, 새로운 예외 조건을 추가하는 방식으로 수정하라.**

#### 잘못된 수정 (❌)
```python
# 기존 코드
excludes = ['유동성']

# 잘못된 수정 - 기존 조건 변경
excludes = ['유동']  # 기존 '유동성' 제거하고 '유동'으로 변경
```

#### 올바른 수정 (✓)
```python
# 기존 코드
excludes = ['유동성']

# 올바른 수정 - 조건 추가
excludes = ['유동성', '유동', '전환']  # 기존 유지 + 새 조건 추가
```

#### 원칙
1. **기존 로직 유지**: 이미 작동하는 케이스를 깨뜨리지 않음
2. **조건 추가 방식**: exclude 리스트에 항목 추가, if 조건에 `and` 추가
3. **테스트**: 새 케이스와 기존 케이스 모두 검증

#### 예시 (2026-01 수정)
- 문제: '유동전환사채'가 '비유동차입부채'로 잘못 분류
- 원인: exclude에 '유동성'만 있고 '유동'이 없음
- 수정: `['유동성']` → `['유동성', '유동', '전환']` (추가)
- 결과: 기존 '사채', '유동성사채' 케이스 유지 + '유동전환사채' 제외

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
├── pe_chatbot.py                # PE 전문 AI 챗봇 모듈
├── test_companies.json          # 테스트 기업 리스트 (회귀 테스트용)
├── output/                      # 생성된 Excel 파일
└── CLAUDE.md                    # 이 파일
```

---

## 주석(Footnotes) 데이터

### 개요
DART 재무제표 주석 데이터를 백엔드에서 처리. **프론트엔드 탭은 미구현 (데이터가 너무 방대하여 롤백).**
엑셀 저장과 API 응답에는 포함되어 있음.

### 데이터 흐름
1. **추출**: `fs_data['notes']` → `{'is_notes': [], 'bs_notes': [], 'cf_notes': []}` (각 항목: `{name, df, consolidated}`)
2. **서버**: `preview_data['notes']` → 각 주석을 `{title, type, consolidated, source, data}` 형태로 변환 (FY + 계정과목 컬럼만)
3. **프론트**: 미사용 (프론트엔드 탭 롤백됨)
4. **엑셀**: 기존대로 `손익주석1`, `재무주석1`, `현금주석1` 시트로 저장 (변경 없음)

### VCM 서브탭 순서
| 인덱스 | 탭 | data-vcm-box | 기본 상태 |
|--------|-----|-------------|-----------|
| 0 | 재무상태표 | bs | active |
| 1 | 손익계산서 | is | active |
| 2 | 재무분석 AI | ai | inactive |

### 관련 코드
- `server.py` line ~1173: `preview_data['notes']` 생성 (API 응답에 포함)

### 기업 리서치 공시 제거
- `super_research_pipeline.py`: `step2_parallel_search()`에서 `_search_disclosure()` 호출 제거
- Step5 프롬프트에서 "최근 공시" 섹션 제거
- 뉴스 JSON에서 `type` 필드 제거 (공시/뉴스 구분 불필요)
- 프론트엔드: "주요 뉴스/공시" → "주요 뉴스"로 변경

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

## PE 전용 AI 챗봇

### 개요
DART 재무제표 추출 완료 후 기업개황정보 우측에 활성화되는 PE 전문 AI 챗봇.
기업개황 + VCM/IS/BS/CF 재무데이터를 컨텍스트로 주입하여 할루시네이션 없는 정확한 답변 제공.

### 아키텍처
```
[질문 입력]
     ↓
[LLM 분류] (Flash) → simple / complex / search
     ↓
[simple] Flash 단일 에이전트 → 즉시 응답
[complex] 3-에이전트 병렬:
  ├── Financial Analyst (Flash): 재무 정량 분석
  ├── Market Researcher (Flash + Firecrawl): 웹 검색 + 시장 조사
  └── PE Advisor (Pro): 종합 → 최종 답변
[search] Firecrawl 검색 → Flash 응답
     ↓
[SSE 스트리밍] → 프론트엔드 타이핑 효과
```

### 관련 파일
- `pe_chatbot.py`: PEChatbot 클래스 (시스템 프롬프트, 재무 컨텍스트, 검색, 멀티에이전트)
- `server.py`: 챗봇 API 엔드포인트 4개
- `index.html`: 챗봇 UI (CSS + HTML + JS)

### API 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/chat/init/{task_id}` | 챗봇 세션 초기화 |
| POST | `/api/chat/message/{task_id}` | 메시지 전송 (SSE 스트리밍) |
| GET | `/api/chat/history/{task_id}` | 대화 내역 조회 |
| POST | `/api/chat/update-context/{task_id}` | AI분석/리서치 완료 후 컨텍스트 업데이트 |

### UI 레이아웃
- 추출 완료 시 `#companyInfoWrapper`가 flex 레이아웃으로 전환
- 좌측: 기업개황정보 (420px 고정)
- 우측: AI 챗봇 (나머지 공간, min 320px)
- 900px 이하: column 방향 (아래로 이동)

### 메모리 관리
- `task['chatbot']`에 PEChatbot 인스턴스 저장
- 챗봇 활성화 시 `preview_data` cleanup 방지 (`'chatbot' not in task` 조건)
- 대화 최대 50턴 → 오래된 대화 자동 제거

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

---

## 검증 완료 기업 목록 (★필수 관리★)

### 핵심 원칙
**새로운 기업을 테스트할 때마다 `test_companies.json`에 추가하라. 버그 수정 후 회귀 테스트에 사용.**

### 테스트 기업 리스트 파일
- **`/home/servermanager/pers-dev/test_companies.json`** — 전체 기업 리스트 (JSON)
- 새 기업 테스트 시 해당 파일에 추가
- 카테고리: `listed.standard`, `listed.edge_cases`, `listed.bottom50_small_cap`, `unlisted.verified`, `unlisted.no_data`

### 현재 현황 (2026-02-26)

| 카테고리 | 개수 | 설명 |
|----------|------|------|
| 상장사 - 표준 | 3 | 삼성전자, E1, 아이티엠반도체 |
| 상장사 - 엣지케이스 | 7 | 소계중복, IS구조특성, 판관비음수, 별도재무제표 등 |
| 상장사 - 소형주(시총하위50) | 30 | 별도 19개 + 연결 11개, 자본잠식 3개 포함 |
| 상장사 - 소형주배치2(시총64~184억) | 46 | KOSDAQ/KOSPI 시총 하위, 매각예정자산 구조 10건 포함 |
| 비상장사 - 검증완료 | 136 | 대기업 포함 (교보생명, 포스코, 비바리퍼블리카 등) |
| 비상장사 - 데이터없음 | 69 | DART 공시 의무 없는 기업 (농업회사법인 다수) |
| **합계** | **291** | |

### 최근 배치 테스트 (2026-02-26)
- 시총 하위 소형주 50개 테스트 (KOSDAQ/KOSPI, 시총 64~184억)
- **결과: 46/48 추출 성공 (95.8%)**
- Balance check: 46/46 전원 PASS (자산총계 = 부채총계 + 자본총계)
- Asset equation FAIL 10건: 매각예정자산이 유동자산에 포함된 구조 (추출 로직 문제 아님)
- NO_DATA 2건: 세니젠(188260), 판타지오(032800) — VCM 데이터 없음
- RESOLVE_FAIL 2건: 한국IT전문학교(226440), 다이얼로그(064260) — DART corp_code 조회 실패

### 이전 배치 테스트 (2026-02-23)
- 시가총액 하위 50 종목 중 보통주 31개 테스트
- **결과: 30/31 성공 (96.8%)**
- 실패 1건: 아이엠(101390) — 동명 사모펀드로 잘못 매칭 (추출 로직 문제 아님)
- 별도재무제표 fallback 19개 기업 모두 정상 추출

### 검증 항목 체크리스트 (★필수★)

**새 기업 테스트 시 반드시 웹 UI에서 다음 항목을 직접 확인하라. 특히 툴팁이 있는 셀은 마우스 호버하여 반드시 검증!**

**수치 검증:**
1. ☐ 자산총계 = 유동자산 + 비유동자산
2. ☐ 부채와자본총계 = 부채총계 + 자본총계
3. ☐ 영업이익 ≈ 매출총이익 - 판관비 (IS 구조에 따라 다를 수 있음)
4. ☐ 음수 값 정상 표시 (영업손실, 자본잠식 등)

**툴팁 검증 (★매우 중요★):**
모든 툴팁이 있는 셀(calc-cell)에서 **본체 값 = 툴팁 하위 항목 합계** 확인!

5. ☐ 기타유동자산: 본체 값 = 툴팁 합계 (다른 카테고리 items 혼입 금지)
6. ☐ 기타비유동자산: 본체 값 = 툴팁 합계
7. ☐ 기타유동부채: 본체 값 = 툴팁 합계
8. ☐ 기타비유동부채: 본체 값 = 툴팁 합계
9. ☐ 기타자본구성요소: 본체 값 = 툴팁 합계 (자본잉여금, 기타포괄손익누계액 등)
10. ☐ 이익잉여금: 본체 값 = 툴팁 합계 (하위 항목이 있는 경우)
11. ☐ NWC: 툴팁 계산식 검증 (유동자산 - 유동부채)
12. ☐ Net Debt: 툴팁 계산식 검증 (차입금 - 현금성자산)
13. ☐ EBITDA: 툴팁 계산식 검증 (영업이익 + 감가상각비 + 무형자산상각비)
14. ☐ 각 카테고리 툴팁의 하위 항목이 해당 카테고리에 실제로 속하는지 확인

**툴팁 검증 방법:**
1. 웹 UI에서 해당 기업 조회
2. Financials 탭에서 calc-cell (밑줄 있는 셀) 확인
3. 마우스를 셀 위에 올려서 툴팁 확인
4. 툴팁의 "합계" 값과 셀에 표시된 값이 일치하는지 비교
5. 불일치 발견 시: Frontdata 시트와 Financials 시트의 해당 값 비교

**불일치 발생 시 확인사항:**
- Frontdata의 '부모' 컬럼이 올바르게 설정되어 있는지
- 하위 항목의 원 단위 → 백만원 변환이 올바른지 (반올림 오차)
- 중복 항목이 있는지

### 3단계 필수 검증 프로세스 (★버그 수정 후 반드시 수행★)

**수정 후 테스트 시 아래 3단계를 반드시 순서대로 검증하라. 하나라도 누락하면 버그 재발!**

#### 1단계: 툴팁 합계 검증

**목적:** 본체 값과 툴팁에 표시되는 하위 항목 합계가 일치하는지 확인

```
예시: 기타자본구성요소 = -1,142 (본체)
      툴팁: 자본잉여금 132 + 자본조정 -1,274 = -1,142 ✓

❌ 틀린 예: 본체 -1,142인데 툴팁에 자본잉여금 132만 표시 (자본조정 누락!)
```

**검증 방법:**
1. 웹 UI에서 툴팁이 있는 모든 셀(calc-cell)에 마우스 호버
2. 툴팁의 "합계" 값과 셀에 표시된 본체 값 비교
3. 불일치 시 → Frontdata 시트에서 해당 항목의 하위가 모두 있는지 확인

**주요 검증 대상:**
- 기타자본구성요소 (자본잉여금 + 자본조정 + 기타포괄손익누계액)
- 매출채권및기타채권 (매출채권 + 미수금 + 미수수익 + 선급금 + 선급비용)
- 기타유동자산, 기타비유동자산, 기타유동부채, 기타비유동부채

#### 2단계: 엑셀-프론트 수치 일치 검증

**목적:** 엑셀 Financials 시트의 수치와 웹 UI에 표시되는 수치가 동일한지 확인

```
엑셀 Financials 시트        웹 UI Financials 탭
─────────────────────────────────────────────
매출채권및기타채권  5,042  →  매출채권및기타채권  5,042 ✓
기타자본구성요소   -1,142  →  기타자본구성요소   -1,142 ✓
```

**검증 방법:**
1. 생성된 Excel 파일의 `Financials` 시트 열기
2. 웹 UI의 Financials 탭과 항목별 수치 비교
3. 특히 음수 값, 큰 숫자가 올바르게 표시되는지 확인

**확인 포인트:**
- 숫자가 일치하는가? (백만원 단위, 정수)
- 음수가 올바르게 표시되는가? (영업손실, 자본잠식 등)
- 천 단위 콤마가 정상인가?

#### 3단계: 항목 존재 여부 검증

**목적:** 본체 계산에 포함된 세부항목이 엑셀과 프론트에 모두 존재하는지 확인

```
본체 계산식: 기타자본 = 자본잉여금 + 자본조정 + 기타포괄손익누계액

확인 대상:
☐ 자본잉여금이 Frontdata에 있는가?
☐ 자본조정이 Frontdata에 있는가?
☐ 기타포괄손익누계액이 Frontdata에 있는가?

❌ 틀린 예: 본체 계산엔 자본조정이 포함되는데 Frontdata에 없음 → 툴팁 불일치!
```

**검증 방법:**
1. server.py에서 본체 값 계산 로직 확인 (어떤 항목들을 합산하는지)
2. 엑셀 `Frontdata` 시트에서 해당 항목들이 하위로 존재하는지 확인
3. 웹 UI 툴팁에서 해당 항목들이 표시되는지 확인

**주요 확인 항목:**

| 본체 항목 | 포함되어야 할 하위 항목 |
|----------|----------------------|
| 기타자본구성요소 | 자본잉여금, 자본조정, 기타포괄손익누계액, 기타자본항목 |
| 매출채권및기타채권 | 매출채권, 미수금, 미수수익, 선급금, 선급비용, 계약자산, 기타금융자산 |
| NWC | 유동자산, 유동부채 |
| Net Debt | 유동차입부채, 비유동차입부채, 현금및현금성자산, 단기투자자산 |
| EBITDA | 영업이익, 감가상각비, 무형자산상각비 |

### 검증 실패 시 디버깅 가이드

**문제: 툴팁 합계 ≠ 본체 값**
1. Frontdata 시트에서 해당 항목의 하위 확인 (`부모` 컬럼)
2. 누락된 하위항목이 있으면 → server.py에서 bs_items.append 로직 확인
3. 본체 계산에 사용된 값이 명시적으로 추가되는지 확인

**문제: 엑셀 수치 ≠ 프론트 수치**
1. API 응답의 `vcm_display` 확인
2. Financials 시트 생성 로직 확인 (백만원 변환, 반올림)
3. 프론트엔드 렌더링 로직 확인 (추가 변환 없어야 함)

**문제: 특정 항목이 엑셀/프론트에 없음**
1. 원본 재무상태표에 해당 항목 존재 여부 확인
2. parse_bs_sections에서 해당 항목이 파싱되는지 확인
3. 본체 계산에 사용되는 항목이면 명시적 추가 로직 필요

### 주요 버그 수정 이력

| 날짜 | 문제 | 원인 | 수정 |
|------|------|------|------|
| 2026-01-29 | 영업손실이 양수로 표시 | find_val에서 괄호 제거만 함 | 괄호 → 음수 변환 로직 추가 |
| 2026-01-29 | 자본잠식 기업 자본총계 오류 | find_bs_val에서 괄호 미처리 + 부채와자본총계 매칭 | 괄호 처리 + excludes 추가 |
| 2026-01-29 | 비유동자산 소계 중복 | items 합계가 소계 포함 | find_bs_val 값 우선 사용 |
| 2026-01-29 | 기타유동자산 툴팁에 다른 카테고리 items 표시 | MAX_ITEMS 초과 카테고리 items가 기타유동자산 하위로 추가됨 | 기타유동자산_원래_items 사용 (line 4211-4221, 4256-4264, 4289-4297, 4332-4340) |
| 2026-01-29 | 손익계산서 항목(판관비 상세 등) UI에서 안 보임 | Financials 시트가 좌우 병렬 구조인데 API에서 그대로 반환 → 프론트가 '항목' 컬럼 못 찾음 | server.py:1077-1120 좌우→상하 변환 로직 추가 |
| 2026-01-29 | 자본조정, 매출채권(주N) 등 항목 인식 못함 | normalize 함수가 "(주10)" 형태와 반각 로마숫자(III.) 처리 안함 | normalize에 `\(주[석\d,\s]*\)` 패턴 및 `^[IVX]+\.` 패턴 추가 (line 2802-2808) |
| 2026-01-29 | 기타자본구성요소 툴팁에 자본조정 누락 | 자본 세부항목이 자본_items에서만 추출되어 parse_bs_sections에서 파싱 안 된 항목 누락 | 본체 계산에 사용된 값(자본잉여금, 자본조정, 기타포괄손익누계액, 기타자본항목)을 명시적으로 추가 (line 4411-4419) |
| 2026-01-29 | 매출채권및기타채권 툴팁에 매출채권 누락 | group_items_by_category가 parse_bs_sections에서 파싱된 것만 사용, 소계 행만 있는 경우 세부항목 없음 | 본체 계산에 사용된 값(매출채권, 미수금, 미수수익 등)을 명시적으로 추가 (line 4242-4258) |
