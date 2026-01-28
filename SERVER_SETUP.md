# 서버 설정 문서

## ⚠️ 중요 주의사항

**server.py 또는 관련 백엔드 파일을 수정한 후에는 반드시 서버를 재시작해야 변경사항이 적용됩니다.**

```bash
pkill -9 -f "uvicorn server:app"
cd /home/servermanager/pers
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

---

## 🔧 서버 구성

### 백엔드 서버
- **프레임워크**: FastAPI
- **ASGI 서버**: Uvicorn
- **포트**: 8000
- **호스트**: 0.0.0.0
- **작업 디렉토리**: `/home/servermanager/pers`

**서버 시작 명령어:**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

**백그라운드 실행:**
```bash
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

**서버 재시작:**
```bash
pkill -9 -f "uvicorn server:app"
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

---

## 🌐 Nginx 리버스 프록시

### 설정 파일 위치
- **메인 설정**: `/etc/nginx/sites-available/pers.moatai.app`
- **심볼릭 링크**: `/etc/nginx/sites-enabled/pers.moatai.app`

### 포트 매핑
```
외부 요청 (pers.moatai.app:80)
    ↓
Nginx (포트 80)
    ↓
Uvicorn (localhost:8000)
```

### 설정 내용
```nginx
server {
    listen 80;
    server_name pers.moatai.app;

    location / {
        proxy_pass http://localhost:8000;  # ⚠️ 중요: 포트 8000
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_buffering off;  # SSE 지원
        proxy_read_timeout 86400;
    }
}
```

### Nginx 관리 명령어
```bash
# 설정 테스트
sudo nginx -t

# 설정 적용 (reload)
sudo systemctl reload nginx

# Nginx 재시작
sudo systemctl restart nginx

# Nginx 상태 확인
sudo systemctl status nginx
```

---

## ⚠️ 문제 해결

### 502 Bad Gateway 에러
**원인**: Nginx가 잘못된 포트로 프록시 중

**확인 방법:**
```bash
# 1. Uvicorn 서버 포트 확인
lsof -i :8000

# 2. Nginx 설정 확인
sudo cat /etc/nginx/sites-available/pers.moatai.app | grep proxy_pass

# 3. 포트 불일치 시 수정
sudo sed -i 's/8080/8000/g' /etc/nginx/sites-available/pers.moatai.app
sudo nginx -t
sudo systemctl reload nginx
```

### 서버가 응답하지 않을 때
```bash
# 1. 프로세스 확인
ps aux | grep uvicorn

# 2. 포트 확인
lsof -i :8000

# 3. 로그 확인
tail -50 /home/servermanager/pers/server.log

# 4. 서버 재시작
pkill -9 -f "uvicorn"
cd /home/servermanager/pers
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

### 중복 서버 프로세스
```bash
# 모든 Python 서버 확인
ps aux | grep -E "python.*server|uvicorn" | grep -v grep

# 중복 프로세스 종료
pkill -9 -f "uvicorn"
pkill -9 -f "python3 server.py"
```

---

## 📋 체크리스트

서버 배포/재시작 시 반드시 확인:

- [ ] Uvicorn이 **포트 8000**에서 실행 중인가?
- [ ] Nginx가 **localhost:8000**로 프록시하는가?
- [ ] 중복 서버 프로세스가 없는가?
- [ ] 로그에 에러가 없는가?
- [ ] 브라우저에서 정상 접속되는가?

---

## 🔍 빠른 진단

```bash
# 한 번에 모든 상태 확인
echo "=== Uvicorn 서버 ===" && \
ps aux | grep uvicorn | grep -v grep && \
echo "=== 포트 8000 ===" && \
lsof -i :8000 && \
echo "=== Nginx 설정 ===" && \
sudo grep -r "proxy_pass" /etc/nginx/sites-enabled/ && \
echo "=== 로컬 테스트 ===" && \
curl -s http://localhost:8000 | head -5
```

---

**마지막 업데이트**: 2026-01-04
**작성자**: Claude Code
