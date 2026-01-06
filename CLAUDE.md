# PERS 프로젝트 개발 규칙

## 백엔드/프론트엔드 동기화 규칙

### 핵심 원칙
**백엔드(server.py)를 수정할 때 반드시 프론트엔드(index.html)도 함께 확인하라.**

### VCM 데이터 흐름

1. **백엔드 (server.py)**
   - `create_vcm_format()` 함수에서 VCM 데이터 생성
   - Excel 파일의 'VCM전용포맷' 시트에 저장
   - `preview_data['vcm']`으로 API 응답에 포함

2. **프론트엔드 (index.html)**
   - `previewData.vcm`에서 VCM 데이터를 직접 읽음
   - `bsItems` 배열을 VCM 데이터에서 **동적으로 생성**
   - 세부항목(부모가 있는 항목)은 합산하여 **툴팁**에 표시
   - 빈 행은 백엔드에서 필터링되어 프론트에 전달되지 않음

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
