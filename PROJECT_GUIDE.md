# PERS/ValuLens 프로젝트 총괄 가이드

> **이 문서는 프로젝트의 모든 규칙과 절차를 총괄합니다.**
> 코드 수정, 검증, 배포 전에 반드시 이 문서와 관련 문서를 참고하세요.

---

## 1. 문서 체계

| 문서 | 위치 | 역할 |
|------|------|------|
| **PROJECT_GUIDE.md** (이 파일) | 프로젝트 루트 | 전체 규칙/절차 총괄 |
| **CLAUDE.md** | 프로젝트 루트 | 코드 규칙, API 문서, 검증 체크리스트 (48항목), 버그 수정 이력 |
| **MODIFICATION_HISTORY.md** | 프로젝트 루트 | 수정 히스토리 (날짜별 변경/원인/부작용/교훈) |
| **MEMORY.md** | `~/.claude/projects/.../memory/` | Claude 메모리 (프로젝트 컨텍스트, 아키텍처 결정) |

---

## 2. 코드 수정 절차 (★최우선★)

### 수정 전 (반드시)
1. **MODIFICATION_HISTORY.md 확인** — 과거 유사 수정 및 부작용 파악
2. **CLAUDE.md 검증 체크리스트 확인** — 수정 영향 범위의 검증 항목 파악
3. **수정 방향 결정** — 기존 동작 보호 원칙 적용:
   - 기존 로직 유지, 조건 추가 방식으로 수정
   - 기존 419개 기업 검증 결과를 깨뜨리면 안 됨
   - 동일계정명 등 edge case 고려

### 수정 중
4. **dev2에서만 수정** — 프로덕션 직접 수정 금지
5. **문법 검사** — `python3 -c "import ast; ast.parse(open('server.py').read())"`
6. **서버 재시작** — `sudo systemctl restart pers-dev` 또는 수동

### 수정 후 (반드시)
7. **MODIFICATION_HISTORY.md 업데이트** — 아래 템플릿으로 기록:
   ```markdown
   ### [FIX/FEAT/DOC] 제목
   - **문제**:
   - **원인**:
   - **수정**:
   - **영향 범위**: 파일명, 함수명, 라인 범위
   - **검증**: 테스트 결과
   - **회귀**: 회귀 테스트 결과
   - **부작용**: 있으면 기록, 없으면 "없음"
   - **교훈**: 향후 참고할 점
   ```
8. **검증 체크리스트 수행** — CLAUDE.md의 해당 항목 체크
9. **회귀 테스트** — 삼성전자, E1, 패스트파이브, 한화에어로스페이스 필수

---

## 3. 검증 절차

### 단계별 검증
| 단계 | 방법 | 필수 여부 |
|------|------|----------|
| 1. 단위 테스트 | `test_operational.py` | server.py 수정 시 |
| 2. 통합 테스트 | `test_full_pipeline.py` | 추출 로직 수정 시 |
| 3. 프론트엔드 검증 | `test_frontend_verify.py` | 모든 수정의 최종 단계 |
| 4. 브라우저 검증 | `test_frontend_browser.py` | UI 변경 시 |
| 5. 회귀 테스트 | 4개 기업 + CJ대한통운 | 필수 |

### 검증 체크리스트 (48항목)
→ **CLAUDE.md** "검증 항목 체크리스트" 섹션 참조

주요 카테고리:
- #1-14: 수치/툴팁 검증
- #15-22: 섹션합계/EBITDA/Net Debt 합리성
- #23-26: BS/IS 탭 분리
- #27-42: **V1→V2 회귀 검증** (BS/IS 필수 항목, EBITDA 구성요소, Net Debt)
- #43-48: HTML IS 폴백

### 합격 기준
- FAIL=0 필수
- WARN은 INFO 레벨(판관비 없음 등)만 허용
- ERROR=0 필수

---

## 4. 배포 절차

```
[1] dev2에서 수정 + 테스트
    cd /home/servermanager/pers-dev2
    # 수정 → 재시작 → 테스트

[2] 커밋 + 푸시
    git add . && git commit -m "fix: ..." && git push

[3] 프로덕션 배포 (사용자 승인 후)
    cd /home/servermanager/pers
    git fetch origin
    git merge origin/feature/llm-classification --no-edit
    git push origin main
    sudo systemctl restart pers

[4] 프로덕션 검증
    curl -s http://localhost:8000/ → 200 OK
```

---

## 5. 프로젝트 아키텍처 요약

### 환경
| 환경 | 디렉토리 | 포트 | 브랜치 | URL |
|------|---------|------|--------|-----|
| Prod | `/home/servermanager/pers` | 8000 | main | https://pers.moatai.app |
| Dev | `/home/servermanager/pers-dev` | 8001 | develop | 백업용, 개발 금지 |
| Dev2 | `/home/servermanager/pers-dev2` | 8002 | feature/llm-classification | 모든 개발 |

### 핵심 파일
| 파일 | 라인 수 | 역할 |
|------|---------|------|
| `server.py` | ~10,000 | FastAPI 백엔드, VCM 생성, API 엔드포인트 |
| `index.html` | ~6,000 | 프론트엔드 UI (SPA) |
| `dart_financial_extractor.py` | ~900 | DART 재무제표 추출 |
| `financial_insight_analyzer.py` | ~800 | AI 재무분석 (Gemini) |
| `pe_chatbot.py` | ~600 | PE 전문 AI 챗봇 |
| `super_research_pipeline.py` | ~700 | 기업 리서치 파이프라인 |
| `account_classifier.py` | ~400 | LLM 계정 분류 (V2) |
| `database.py` | ~300 | SQLite DB (사용자/캐시) |

### 데이터 흐름
```
[사용자] → 기업 검색 → DART API 추출
    ↓
[XBRL/HTML] → BS/IS/CF/Notes 파싱
    ↓
[VCM v2] → LLM 분류 → 그룹핑 → Financials/Frontdata 생성
    ↓
[Excel] → 시트 저장 + [API] → preview_data 반환
    ↓
[프론트엔드] → vcm_display 테이블 + vcm 툴팁 렌더링
    ↓
[AI 분석] → Gemini 이상감지 → 웹검색 → 보고서 생성
    ↓
[챗봇] → 재무 컨텍스트 + 실시간 Q&A
```

### 외부 API
| API | 용도 | 키 환경변수 |
|-----|------|-----------|
| DART OpenAPI | 재무제표 추출 | `DART_API_KEY` |
| Gemini (Pro/Flash) | AI 분석, 챗봇, 분류 | `GEMINI_API_KEY`, `GEMINI_CHATBOT_API_KEY` |
| Firecrawl | 웹 검색/스크래핑 | `FIRECRAWL_API_KEY` |
| Naver Search | 뉴스 검색 | `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` |

---

## 6. 알려진 이슈 및 주의사항

### 현재 미해결
| 이슈 | 심각도 | 설명 |
|------|--------|------|
| EBITDA Notes FY2023+ | HIGH | DART HTML 주석의 FY 커버리지 한계 |
| 주석 단위 불일치 | - | XBRL(원) vs HTML(천원) — 새 소스 추가 시 반드시 확인 |

### 과거 교훈 (수정 시 참고)
1. **substring 매칭 금지** — `'매출' in text`는 '매출채권' 등과 오매칭 → exact match 사용
2. **본체-Frontdata 동기화** — VCM 항목 수정 시 본체 값 계산과 Frontdata 하위항목 추가를 반드시 함께
3. **overflow 로직** — 전체 연도를 동시에 처리해야 연도 간 항목 불일치 방지
4. **동일계정명** — 유동/비유동 구분 필요 (차입금, 리스부채 등)
5. **Notes 단위** — 소스마다 원/천원 다름, 새 소스 추가 시 단위 검증 필수
6. **LLM 프롬프트 수정** — 캐시 무효화 범위 최소화

---

## 7. 연락처 및 참고

- **프로젝트**: ValuLens (PERS)
- **서버**: Azure VM (vm-docker-01), 2코어/3.8GB RAM
- **도메인**: pers.moatai.app
- **관리자**: admin@valulens.com
