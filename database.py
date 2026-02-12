"""
데이터베이스 모듈 - SQLite 기반 사용자 및 사용량 관리
"""

import sqlite3
import hashlib
import secrets
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
from contextlib import contextmanager

DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'financial_data.db')


@contextmanager
def get_db():
    """데이터베이스 연결 컨텍스트 매니저"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # dict-like 접근 가능
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_db():
    """데이터베이스 초기화 - 테이블 생성"""
    with get_db() as conn:
        cursor = conn.cursor()

        # 사용자 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',           -- 'admin' / 'user'
                tier TEXT DEFAULT 'free',           -- 'free' / 'basic' / 'pro'
                search_limit INTEGER DEFAULT 10,    -- 무료 검색 한도
                search_used INTEGER DEFAULT 0,      -- 사용한 검색 수
                extract_limit INTEGER DEFAULT 5,    -- 무료 추출 한도
                extract_used INTEGER DEFAULT 0,     -- 사용한 추출 수
                ai_limit INTEGER DEFAULT 3,         -- 무료 AI 분석 한도
                ai_used INTEGER DEFAULT 0,          -- 사용한 AI 분석 수
                subscription_start TIMESTAMP,       -- 유료 시작일
                expires_at TIMESTAMP,               -- 유료 만료일 (NULL이면 무료)
                tokens INTEGER DEFAULT 5,           -- 검색 토큰 (Free 기본 5개)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 기존 테이블에 subscription_start 컬럼 추가 (없으면)
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN subscription_start TIMESTAMP')
        except:
            pass  # 이미 존재하면 무시

        # 기존 테이블에 tokens 컬럼 추가 (없으면)
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN tokens INTEGER DEFAULT 5')
        except:
            pass  # 이미 존재하면 무시

        # 로그인 기록 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS login_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                login_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # 검색 기록 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                corp_code TEXT,
                corp_name TEXT,
                market TEXT,                        -- 코스피/코스닥/비상장
                searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # 추출 기록 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extraction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                corp_code TEXT,
                corp_name TEXT,
                start_year INTEGER,
                end_year INTEGER,
                file_path TEXT,                     -- 생성된 Excel 경로
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # AI 사용량 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                corp_code TEXT,
                corp_name TEXT,
                model_name TEXT,                    -- gemini-2.5-pro 등
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost REAL DEFAULT 0,                -- 비용 (USD)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # 세션 테이블 (토큰 기반 인증)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # 인덱스 생성
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_user ON search_history(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_extraction_user ON extraction_history(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_llm_user ON llm_usage(user_id)')

        print("[DB] 데이터베이스 초기화 완료")


def hash_password(password: str) -> str:
    """비밀번호 해시 생성 (SHA-256 + salt)"""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, password_hash: str) -> bool:
    """비밀번호 검증"""
    try:
        salt, hashed = password_hash.split(':')
        return hashlib.sha256((password + salt).encode()).hexdigest() == hashed
    except:
        return False


def generate_session_token() -> str:
    """세션 토큰 생성"""
    return secrets.token_urlsafe(32)


# ==================== 사용자 관리 ====================

def create_user(email: str, password: str, role: str = 'user', tier: str = 'free') -> Optional[int]:
    """사용자 생성"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            password_hash = hash_password(password)
            cursor.execute('''
                INSERT INTO users (email, password_hash, role, tier)
                VALUES (?, ?, ?, ?)
            ''', (email, password_hash, role, tier))
            user_id = cursor.lastrowid
            print(f"[DB] 사용자 생성: {email} (ID: {user_id})")
            return user_id
        except sqlite3.IntegrityError:
            print(f"[DB] 사용자 생성 실패 - 이메일 중복: {email}")
            return None


def get_user_by_email(email: str) -> Optional[Dict]:
    """이메일로 사용자 조회"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict]:
    """ID로 사용자 조회"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """사용자 인증"""
    user = get_user_by_email(email)
    if user and verify_password(password, user['password_hash']):
        return user
    return None


def update_user_tier(user_id: int, tier: str, expires_at: Optional[datetime] = None):
    """사용자 등급 업데이트"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users
            SET tier = ?, expires_at = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (tier, expires_at, user_id))


def reset_user_usage(user_id: int):
    """사용자 사용량 초기화 (월간 리셋 등)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users
            SET search_used = 0, extract_used = 0, ai_used = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (user_id,))


# ==================== 세션 관리 ====================

def create_session(user_id: int, ip_address: str = None, user_agent: str = None) -> str:
    """세션 생성 및 로그인 기록"""
    token = generate_session_token()
    with get_db() as conn:
        cursor = conn.cursor()

        # 세션 생성
        cursor.execute('''
            INSERT INTO sessions (user_id, token)
            VALUES (?, ?)
        ''', (user_id, token))

        # 로그인 기록
        cursor.execute('''
            INSERT INTO login_history (user_id, ip_address, user_agent)
            VALUES (?, ?, ?)
        ''', (user_id, ip_address, user_agent))

    return token


def check_and_expire_subscription(user_id: int, expires_at: str) -> bool:
    """구독 만료 체크 및 Free 전환. 만료되었으면 True 반환"""
    if not expires_at:
        return False

    from datetime import datetime
    try:
        # 만료일 파싱 (날짜만 비교)
        expiry_date = datetime.strptime(expires_at.split('T')[0], '%Y-%m-%d').date()
        today = datetime.now().date()

        if today > expiry_date:
            # 만료됨 - Free로 전환하고 토큰 0으로
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE users SET tier = 'free', tokens = 0,
                           search_limit = 10, extract_limit = 5, ai_limit = 3,
                           updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (user_id,))
            print(f"[DB] 구독 만료: user_id={user_id}, 만료일={expires_at}")
            return True
    except Exception as e:
        print(f"[DB] 만료 체크 오류: {e}")
    return False


def get_session(token: str) -> Optional[Dict]:
    """세션 조회 및 활동 시간 업데이트"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT s.*, u.email, u.role, u.tier, u.search_limit, u.search_used,
                   u.extract_limit, u.extract_used, u.ai_limit, u.ai_used, u.expires_at, u.tokens
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
        ''', (token,))
        row = cursor.fetchone()

        if row:
            result = dict(row)

            # 만료 체크 (유료 사용자만)
            if result.get('tier') in ['basic', 'pro'] and result.get('expires_at'):
                if check_and_expire_subscription(result['user_id'], result['expires_at']):
                    # 만료되었으면 다시 조회
                    cursor.execute('''
                        SELECT s.*, u.email, u.role, u.tier, u.search_limit, u.search_used,
                               u.extract_limit, u.extract_used, u.ai_limit, u.ai_used, u.expires_at, u.tokens
                        FROM sessions s
                        JOIN users u ON s.user_id = u.id
                        WHERE s.token = ?
                    ''', (token,))
                    row = cursor.fetchone()
                    result = dict(row) if row else result

            # 활동 시간 업데이트
            cursor.execute('''
                UPDATE sessions SET last_activity = CURRENT_TIMESTAMP
                WHERE token = ?
            ''', (token,))
            return result
        return None


def delete_session(token: str):
    """세션 삭제 (로그아웃)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sessions WHERE token = ?', (token,))


def delete_user_sessions(user_id: int):
    """사용자의 모든 세션 삭제"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))


# ==================== 사용량 기록 ====================

def log_search(user_id: int, corp_code: str, corp_name: str, market: str = None):
    """검색 기록 저장 및 사용량 증가"""
    with get_db() as conn:
        cursor = conn.cursor()

        # 검색 기록 저장
        cursor.execute('''
            INSERT INTO search_history (user_id, corp_code, corp_name, market)
            VALUES (?, ?, ?, ?)
        ''', (user_id, corp_code, corp_name, market))

        # 사용량 증가
        cursor.execute('''
            UPDATE users SET search_used = search_used + 1, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (user_id,))


def log_extraction(user_id: int, corp_code: str, corp_name: str,
                   start_year: int, end_year: int, file_path: str):
    """추출 기록 저장 및 사용량 증가"""
    with get_db() as conn:
        cursor = conn.cursor()

        # 추출 기록 저장
        cursor.execute('''
            INSERT INTO extraction_history (user_id, corp_code, corp_name, start_year, end_year, file_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, corp_code, corp_name, start_year, end_year, file_path))

        # 사용량 증가
        cursor.execute('''
            UPDATE users SET extract_used = extract_used + 1, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (user_id,))


def log_llm_usage(user_id: int, corp_code: str, corp_name: str, model_name: str,
                  input_tokens: int, output_tokens: int, cost: float = 0):
    """LLM 사용량 기록 및 AI 사용 횟수 증가"""
    with get_db() as conn:
        cursor = conn.cursor()

        # LLM 사용량 저장
        cursor.execute('''
            INSERT INTO llm_usage (user_id, corp_code, corp_name, model_name,
                                   input_tokens, output_tokens, total_tokens, cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, corp_code, corp_name, model_name,
              input_tokens, output_tokens, input_tokens + output_tokens, cost))

        # AI 사용 횟수 증가
        cursor.execute('''
            UPDATE users SET ai_used = ai_used + 1, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (user_id,))


# ==================== 통계 조회 ====================

def get_user_stats(user_id: int) -> Dict:
    """사용자 통계 조회"""
    with get_db() as conn:
        cursor = conn.cursor()

        # 검색 횟수
        cursor.execute('SELECT COUNT(*) FROM search_history WHERE user_id = ?', (user_id,))
        search_count = cursor.fetchone()[0]

        # 추출 횟수
        cursor.execute('SELECT COUNT(*) FROM extraction_history WHERE user_id = ?', (user_id,))
        extract_count = cursor.fetchone()[0]

        # AI 사용 횟수
        cursor.execute('SELECT COUNT(*) FROM llm_usage WHERE user_id = ?', (user_id,))
        ai_count = cursor.fetchone()[0]

        # 총 토큰 사용량
        cursor.execute('SELECT SUM(total_tokens) FROM llm_usage WHERE user_id = ?', (user_id,))
        total_tokens = cursor.fetchone()[0] or 0

        # 사용자 정보
        user = get_user_by_id(user_id)

        return {
            'search_count': search_count,
            'extract_count': extract_count,
            'ai_count': ai_count,
            'total_tokens': total_tokens,
            'search_limit': user['search_limit'] if user else 0,
            'search_used': user['search_used'] if user else 0,
            'extract_limit': user['extract_limit'] if user else 0,
            'extract_used': user['extract_used'] if user else 0,
            'ai_limit': user['ai_limit'] if user else 0,
            'ai_used': user['ai_used'] if user else 0,
            'tier': user['tier'] if user else 'free'
        }


def get_user_activity_history(user_id: int, limit: int = 200) -> Optional[Dict]:
    """회원 활동 이력 조회 (관리자용)"""
    user = get_user_by_id(user_id)
    if not user:
        return None

    with get_db() as conn:
        cursor = conn.cursor()

        # 검색 기록
        cursor.execute('''
            SELECT corp_name, market, searched_at
            FROM search_history
            WHERE user_id = ?
            ORDER BY searched_at DESC
            LIMIT ?
        ''', (user_id, limit))
        searches = [dict(row) for row in cursor.fetchall()]

        # 추출 기록
        cursor.execute('''
            SELECT corp_name, start_year, end_year, created_at
            FROM extraction_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        extractions = [dict(row) for row in cursor.fetchall()]

        # AI 분석 기록
        cursor.execute('''
            SELECT corp_name, model_name, total_tokens, cost, created_at
            FROM llm_usage
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        analyses = [dict(row) for row in cursor.fetchall()]

    return {
        'user': {
            'id': user['id'],
            'email': user['email'],
            'role': user['role'],
            'tier': user['tier'],
            'created_at': user['created_at'],
        },
        'searches': searches,
        'extractions': extractions,
        'analyses': analyses,
    }


def get_popular_companies(limit: int = 10) -> List[Dict]:
    """인기 검색 종목"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT corp_code, corp_name, COUNT(*) as count
            FROM search_history
            GROUP BY corp_code
            ORDER BY count DESC
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]


def get_all_users() -> List[Dict]:
    """모든 사용자 조회 (관리자용) - 활동 통계 포함"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, email, role, tier, search_used, extract_used, ai_used,
                   subscription_start, expires_at, tokens, created_at
            FROM users
            ORDER BY created_at DESC
        ''')
        users = [dict(row) for row in cursor.fetchall()]

        # 각 사용자별 활동 통계 추가
        for user in users:
            user_id = user['id']

            # 마지막 로그인 시간
            cursor.execute('''
                SELECT login_at FROM login_history
                WHERE user_id = ?
                ORDER BY login_at DESC LIMIT 1
            ''', (user_id,))
            row = cursor.fetchone()
            user['last_login'] = row[0] if row else None

            # 총 토큰 사용량 (llm_usage에서 total_tokens 합계)
            cursor.execute('''
                SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage
                WHERE user_id = ?
            ''', (user_id,))
            user['total_tokens_used'] = cursor.fetchone()[0]

            # 최근 7일간 토큰 사용량
            cursor.execute('''
                SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage
                WHERE user_id = ? AND created_at >= datetime('now', '-7 days')
            ''', (user_id,))
            weekly_tokens = cursor.fetchone()[0]
            user['weekly_tokens_used'] = weekly_tokens
            user['daily_avg_tokens'] = round(weekly_tokens / 7, 1) if weekly_tokens else 0

            # 첫 사용일부터 현재까지 일수 (평균 계산용)
            cursor.execute('''
                SELECT MIN(created_at) FROM llm_usage WHERE user_id = ?
            ''', (user_id,))
            first_usage = cursor.fetchone()[0]
            if first_usage and user['total_tokens_used'] > 0:
                from datetime import datetime
                try:
                    first_date = datetime.fromisoformat(first_usage.replace('Z', '+00:00'))
                    days_since_first = max(1, (datetime.now() - first_date.replace(tzinfo=None)).days)
                    user['overall_daily_avg'] = round(user['total_tokens_used'] / days_since_first, 1)
                except:
                    user['overall_daily_avg'] = 0
            else:
                user['overall_daily_avg'] = 0

        return users


def use_token(user_id: int) -> bool:
    """토큰 1개 사용 (차감). 성공하면 True, 토큰 부족하면 False"""
    with get_db() as conn:
        cursor = conn.cursor()
        # 현재 토큰 확인
        cursor.execute('SELECT tokens FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        if not row or (row[0] is not None and row[0] <= 0):
            return False
        # 토큰 차감
        cursor.execute('''
            UPDATE users SET tokens = tokens - 1, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND (tokens IS NULL OR tokens > 0)
        ''', (user_id,))
        return cursor.rowcount > 0


def update_user_tokens(user_id: int, tokens: int):
    """사용자 토큰 수정 (관리자용)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET tokens = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (tokens, user_id))


def update_user_password(user_id: int, new_password_hash: str):
    """사용자 비밀번호 변경"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (new_password_hash, user_id))


def get_admin_analytics() -> Dict:
    """관리자용 종합 분석 데이터"""
    with get_db() as conn:
        cursor = conn.cursor()
        analytics = {}

        # 1. 사용자 통계
        cursor.execute('SELECT COUNT(*) FROM users')
        analytics['total_users'] = cursor.fetchone()[0]

        # 2. DAU (Daily Active Users) - 오늘 로그인한 유니크 사용자
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) FROM login_history
            WHERE date(login_at) = date('now')
        ''')
        analytics['dau'] = cursor.fetchone()[0]

        # 3. WAU (Weekly Active Users) - 최근 7일 로그인한 유니크 사용자
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) FROM login_history
            WHERE login_at >= datetime('now', '-7 days')
        ''')
        analytics['wau'] = cursor.fetchone()[0]

        # 4. MAU (Monthly Active Users) - 최근 30일 로그인한 유니크 사용자
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) FROM login_history
            WHERE login_at >= datetime('now', '-30 days')
        ''')
        analytics['mau'] = cursor.fetchone()[0]

        # 5. 리텐션율 (7일) - 7일 전에 가입한 사용자 중 최근 7일 내 재접속 비율
        cursor.execute('''
            SELECT COUNT(*) FROM users
            WHERE created_at <= datetime('now', '-7 days')
        ''')
        users_7days_ago = cursor.fetchone()[0]

        cursor.execute('''
            SELECT COUNT(DISTINCT u.id) FROM users u
            INNER JOIN login_history lh ON u.id = lh.user_id
            WHERE u.created_at <= datetime('now', '-7 days')
            AND lh.login_at >= datetime('now', '-7 days')
        ''')
        retained_7days = cursor.fetchone()[0]
        analytics['retention_7d'] = round(retained_7days / users_7days_ago * 100, 1) if users_7days_ago > 0 else 0

        # 6. 리텐션율 (30일)
        cursor.execute('''
            SELECT COUNT(*) FROM users
            WHERE created_at <= datetime('now', '-30 days')
        ''')
        users_30days_ago = cursor.fetchone()[0]

        cursor.execute('''
            SELECT COUNT(DISTINCT u.id) FROM users u
            INNER JOIN login_history lh ON u.id = lh.user_id
            WHERE u.created_at <= datetime('now', '-30 days')
            AND lh.login_at >= datetime('now', '-30 days')
        ''')
        retained_30days = cursor.fetchone()[0]
        analytics['retention_30d'] = round(retained_30days / users_30days_ago * 100, 1) if users_30days_ago > 0 else 0

        # 7. 토큰 사용량
        cursor.execute('SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage')
        analytics['tokens_total'] = cursor.fetchone()[0]

        cursor.execute('''
            SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage
            WHERE created_at >= datetime('now', '-7 days')
        ''')
        analytics['tokens_week'] = cursor.fetchone()[0]

        cursor.execute('''
            SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage
            WHERE date(created_at) = date('now')
        ''')
        analytics['tokens_today'] = cursor.fetchone()[0]

        # 8. 추출/AI 분석 횟수
        cursor.execute('SELECT COUNT(*) FROM extraction_history')
        analytics['extractions_total'] = cursor.fetchone()[0]

        cursor.execute('''
            SELECT COUNT(*) FROM extraction_history
            WHERE date(created_at) = date('now')
        ''')
        analytics['extractions_today'] = cursor.fetchone()[0]

        cursor.execute('''
            SELECT COUNT(*) FROM extraction_history
            WHERE created_at >= datetime('now', '-7 days')
        ''')
        analytics['extractions_week'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM llm_usage')
        analytics['ai_analyses_total'] = cursor.fetchone()[0]

        cursor.execute('''
            SELECT COUNT(*) FROM llm_usage
            WHERE date(created_at) = date('now')
        ''')
        analytics['ai_analyses_today'] = cursor.fetchone()[0]

        # 9. 검색 횟수
        cursor.execute('SELECT COUNT(*) FROM search_history')
        analytics['searches_total'] = cursor.fetchone()[0]

        cursor.execute('''
            SELECT COUNT(*) FROM search_history
            WHERE date(searched_at) = date('now')
        ''')
        analytics['searches_today'] = cursor.fetchone()[0]

        cursor.execute('''
            SELECT COUNT(*) FROM search_history
            WHERE searched_at >= datetime('now', '-7 days')
        ''')
        analytics['searches_week'] = cursor.fetchone()[0]

        # 10. 인기 검색 종목 (최근 30일)
        cursor.execute('''
            SELECT corp_name, corp_code, COUNT(*) as cnt
            FROM search_history
            WHERE searched_at >= datetime('now', '-30 days')
            GROUP BY corp_code
            ORDER BY cnt DESC
            LIMIT 10
        ''')
        analytics['top_searched'] = [
            {'corp_name': row[0], 'corp_code': row[1], 'count': row[2]}
            for row in cursor.fetchall()
        ]

        # 11. 최근 7일 일별 활동 (차트용)
        cursor.execute('''
            SELECT date(login_at) as day, COUNT(DISTINCT user_id) as users
            FROM login_history
            WHERE login_at >= datetime('now', '-7 days')
            GROUP BY date(login_at)
            ORDER BY day
        ''')
        analytics['daily_active_users'] = [
            {'date': row[0], 'users': row[1]}
            for row in cursor.fetchall()
        ]

        cursor.execute('''
            SELECT date(created_at) as day, COUNT(*) as cnt
            FROM extraction_history
            WHERE created_at >= datetime('now', '-7 days')
            GROUP BY date(created_at)
            ORDER BY day
        ''')
        analytics['daily_extractions'] = [
            {'date': row[0], 'count': row[1]}
            for row in cursor.fetchall()
        ]

        return analytics


# 초기화
if __name__ == '__main__':
    init_db()

    # 테스트용 관리자 계정 생성
    admin_id = create_user('admin@example.com', 'admin123', role='admin', tier='pro')
    if admin_id:
        print(f"관리자 계정 생성됨: admin@example.com / admin123")
