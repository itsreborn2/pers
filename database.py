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
                name TEXT,                          -- 이름
                company TEXT,                       -- 회사명
                phone TEXT,                         -- 연락처
                role TEXT DEFAULT 'user',           -- 'admin' / 'user'
                tier TEXT DEFAULT 'free',           -- 'free' / 'basic' / 'pro'
                search_limit INTEGER DEFAULT 9999,  -- 검색 한도 (무제한)
                search_used INTEGER DEFAULT 0,      -- 사용한 검색 수
                extract_limit INTEGER DEFAULT 9999, -- 추출 한도 (무제한)
                extract_used INTEGER DEFAULT 0,     -- 사용한 추출 수
                ai_limit INTEGER DEFAULT 9999,      -- AI 분석 한도 (무제한)
                ai_used INTEGER DEFAULT 0,          -- 사용한 AI 분석 수
                subscription_start TIMESTAMP,       -- 유료 시작일
                expires_at TIMESTAMP,               -- 유료 만료일 (NULL이면 무료)
                search_count INTEGER DEFAULT 30,     -- 통합검색 횟수 (Free 기본 30회)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 기존 테이블에 컬럼 추가 (없으면 무시)
        for col_def in [
            'ALTER TABLE users ADD COLUMN subscription_start TIMESTAMP',
            'ALTER TABLE users ADD COLUMN search_count INTEGER DEFAULT 5',
            'ALTER TABLE users ADD COLUMN name TEXT',
            'ALTER TABLE users ADD COLUMN company TEXT',
            'ALTER TABLE users ADD COLUMN phone TEXT',
            'ALTER TABLE users ADD COLUMN chat_used INTEGER DEFAULT 0',
            'ALTER TABLE llm_usage ADD COLUMN ip_address TEXT',
        ]:
            try:
                cursor.execute(col_def)
            except:
                pass  # 이미 존재하면 무시

        # 레거시 tokens 컬럼 제거 (search_count로 통합됨)
        try:
            # tokens 컬럼이 존재하는지 확인
            cursor.execute("PRAGMA table_info(users)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'tokens' in columns:
                # tokens 값을 search_count에 반영 (tokens가 더 작으면 사용된 것이므로 반영)
                cursor.execute('''
                    UPDATE users SET search_count = tokens
                    WHERE tokens < search_count
                ''')
                # SQLite는 DROP COLUMN을 3.35.0+에서만 지원
                try:
                    cursor.execute('ALTER TABLE users DROP COLUMN tokens')
                    print("[DB] 레거시 tokens 컬럼 제거 완료")
                except:
                    print("[DB] tokens 컬럼 제거 불가 (SQLite 버전), 무시됨")
        except Exception as e:
            print(f"[DB] tokens 마이그레이션 참고: {e}")

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

        # 세션 테이블 (세션 기반 인증)
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

        # 결제 기록 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS payment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                amount INTEGER NOT NULL,             -- 결제 금액 (원)
                payment_method TEXT DEFAULT '계좌이체', -- 결제 수단
                tier_granted TEXT,                    -- 부여된 등급 (basic/pro)
                duration_days INTEGER,                -- 부여 기간 (일)
                memo TEXT,                            -- 관리자 메모
                admin_id INTEGER,                     -- 처리한 관리자 ID
                paid_at TIMESTAMP,                    -- 실제 입금일
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (admin_id) REFERENCES users(id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_payment_user ON payment_history(user_id)')

        # 챗봇 최근 질문 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_recent_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_recent_q_user ON chat_recent_questions(user_id)')

        # 챗봇 대화 이력 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                corp_code TEXT,
                corp_name TEXT,
                question TEXT NOT NULL,
                response TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id)')

        # 앱 설정 테이블 (key-value 방식)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 게스트 사용량 테이블 (비로그인 무료 체험)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS guest_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address TEXT NOT NULL,
                extract_used INTEGER DEFAULT 0,
                chat_used INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_guest_ip ON guest_usage(ip_address)')

        # 기본 설정값 삽입 (없는 경우에만)
        default_settings = {
            'free_search_limit': '9999',
            'free_extract_limit': '9999',
            'free_ai_limit': '9999',
            'free_search_count': '30',
            'guest_extract_limit': '1',
            'guest_chat_limit': '5',
        }
        for key, value in default_settings.items():
            cursor.execute('''
                INSERT OR IGNORE INTO app_settings (key, value) VALUES (?, ?)
            ''', (key, value))

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


# ==================== 앱 설정 관리 ====================

def get_setting(key: str, default: str = None) -> Optional[str]:
    """설정값 조회"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT value FROM app_settings WHERE key = ?', (key,))
        row = cursor.fetchone()
        return row[0] if row else default


def get_free_tier_settings() -> Dict:
    """Free tier 기본 한도 설정값 일괄 조회"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM app_settings WHERE key LIKE 'free_%'")
        settings = {row[0]: int(row[1]) for row in cursor.fetchall()}
    return {
        'search_limit': settings.get('free_search_limit', 10),
        'extract_limit': settings.get('free_extract_limit', 30),
        'ai_limit': settings.get('free_ai_limit', 3),
        'search_count': settings.get('free_search_count', 5),
    }


def update_setting(key: str, value: str):
    """설정값 업데이트 (없으면 생성)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP
        ''', (key, value, value))


# ==================== 사용자 관리 ====================

def create_user(email: str, password: str, role: str = 'user', tier: str = 'free',
                name: str = None, company: str = None, phone: str = None) -> Optional[int]:
    """사용자 생성 - free tier는 app_settings에서 기본 한도 조회"""
    # free tier 기본 한도를 settings에서 가져오기
    if tier == 'free':
        free_settings = get_free_tier_settings()
        search_limit = free_settings['search_limit']
        extract_limit = free_settings['extract_limit']
        ai_limit = free_settings['ai_limit']
        search_count = free_settings['search_count']
    else:
        search_limit = 10
        extract_limit = 5
        ai_limit = 3
        search_count = 5

    with get_db() as conn:
        cursor = conn.cursor()
        try:
            password_hash = hash_password(password)
            cursor.execute('''
                INSERT INTO users (email, password_hash, role, tier, name, company, phone,
                                   search_limit, extract_limit, ai_limit, search_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (email, password_hash, role, tier, name, company, phone,
                  search_limit, extract_limit, ai_limit, search_count))
            user_id = cursor.lastrowid
            print(f"[DB] 사용자 생성: {email} / {name} / {company} (ID: {user_id})")
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
    """사용자 사용량 초기화 (월간 리셋 등) - search_count도 tier에 맞게 복원"""
    free_settings = get_free_tier_settings()
    tier_search_count = {'free': free_settings['search_count'], 'basic': 300, 'pro': 4000}
    with get_db() as conn:
        cursor = conn.cursor()
        # 현재 tier 조회
        cursor.execute('SELECT tier FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        restore_count = tier_search_count.get(row[0], free_settings['search_count']) if row else free_settings['search_count']
        cursor.execute('''
            UPDATE users
            SET search_used = 0, extract_used = 0, ai_used = 0, chat_used = 0,
                search_count = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (restore_count, user_id))


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
            # 만료됨 - Free로 전환 (한도는 settings에서 조회)
            free_settings = get_free_tier_settings()
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE users SET tier = 'free', search_count = 0,
                           search_limit = ?, extract_limit = ?, ai_limit = ?,
                           updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (free_settings['search_limit'], free_settings['extract_limit'],
                      free_settings['ai_limit'], user_id))
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
                   u.extract_limit, u.extract_used, u.ai_limit, u.ai_used, u.expires_at, u.search_count
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
                               u.extract_limit, u.extract_used, u.ai_limit, u.ai_used, u.expires_at, u.search_count
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


def log_llm_usage(user_id: Optional[int], corp_code: str, corp_name: str, model_name: str,
                  input_tokens: int, output_tokens: int, cost: float = 0,
                  increment_used: bool = True, usage_type: str = 'ai',
                  ip_address: str = None):
    """LLM 사용량 기록. increment_used=False이면 토큰만 기록하고 횟수는 안 올림.
    usage_type: 'ai' (재무분석/리서치) 또는 'chat' (챗봇)
    ip_address: 게스트 사용자의 경우 IP로 추적"""
    with get_db() as conn:
        cursor = conn.cursor()

        # LLM 사용량 저장 (user_id가 None이면 게스트 → 0으로 저장)
        effective_user_id = user_id if user_id else 0
        cursor.execute('''
            INSERT INTO llm_usage (user_id, corp_code, corp_name, model_name,
                                   input_tokens, output_tokens, total_tokens, cost, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (effective_user_id, corp_code, corp_name, model_name,
              input_tokens, output_tokens, input_tokens + output_tokens, cost, ip_address))

        # 사용 횟수 증가 (로그인 사용자만)
        if increment_used and user_id:
            if usage_type == 'chat':
                cursor.execute('''
                    UPDATE users SET chat_used = chat_used + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (user_id,))
            else:
                cursor.execute('''
                    UPDATE users SET ai_used = ai_used + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (user_id,))


# ==================== 통계 조회 ====================

def get_user_stats(user_id: int) -> Dict:
    """사용자 통계 조회"""
    with get_db() as conn:
        cursor = conn.cursor()

        # 총 검색 횟수 (이력 기반)
        cursor.execute('SELECT COUNT(*) FROM search_history WHERE user_id = ?', (user_id,))
        total_searches = cursor.fetchone()[0]

        # 총 추출 횟수
        cursor.execute('SELECT COUNT(*) FROM extraction_history WHERE user_id = ?', (user_id,))
        extract_count = cursor.fetchone()[0]

        # AI 사용 횟수
        cursor.execute('SELECT COUNT(*) FROM llm_usage WHERE user_id = ?', (user_id,))
        ai_count = cursor.fetchone()[0]

        # 총 AI API 사용량
        cursor.execute('SELECT SUM(total_tokens) FROM llm_usage WHERE user_id = ?', (user_id,))
        total_tokens = cursor.fetchone()[0] or 0

        # 사용자 정보
        user = get_user_by_id(user_id)

        return {
            'total_searches': total_searches,
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


def get_user_activity_history(user_id: int) -> Optional[Dict]:
    """회원 활동 이력 조회 (관리자용) - 전체 이력"""
    user = get_user_by_id(user_id)
    if not user:
        return None

    with get_db() as conn:
        cursor = conn.cursor()

        # 검색 기록 (전체)
        cursor.execute('''
            SELECT corp_name, market, searched_at
            FROM search_history
            WHERE user_id = ?
            ORDER BY searched_at DESC
        ''', (user_id,))
        searches = [dict(row) for row in cursor.fetchall()]

        # 추출 기록 (전체)
        cursor.execute('''
            SELECT corp_name, start_year, end_year, created_at
            FROM extraction_history
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        extractions = [dict(row) for row in cursor.fetchall()]

        # AI 분석 기록 (챗봇 제외)
        cursor.execute('''
            SELECT corp_name, model_name, total_tokens, cost, created_at
            FROM llm_usage
            WHERE user_id = ? AND model_name != 'gemini-chat'
            ORDER BY created_at DESC
        ''', (user_id,))
        analyses = [dict(row) for row in cursor.fetchall()]

        # 챗봇 대화 이력
        cursor.execute('''
            SELECT corp_name, question, response, input_tokens, output_tokens, created_at
            FROM chat_history
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        chats = [dict(row) for row in cursor.fetchall()]

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
        'chats': chats,
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
    """모든 사용자 조회 (관리자용) - 활동 통계 포함 (단일 쿼리 최적화)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                u.id, u.email, u.name, u.company, u.phone, u.role, u.tier,
                u.search_used, u.extract_used, u.ai_used, u.chat_used,
                u.subscription_start, u.expires_at, u.search_count, u.created_at,
                -- 마지막 로그인
                (SELECT MAX(login_at) FROM login_history WHERE user_id = u.id) as last_login,
                -- 마지막 검색
                (SELECT MAX(searched_at) FROM search_history WHERE user_id = u.id) as last_search,
                -- 검색한 고유 기업 수
                (SELECT COUNT(DISTINCT corp_code) FROM search_history WHERE user_id = u.id) as unique_companies,
                -- 총 AI API 토큰
                (SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage WHERE user_id = u.id) as total_tokens_used,
                -- 최근 7일 AI API 토큰
                (SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage
                 WHERE user_id = u.id AND created_at >= datetime('now', '-7 days')) as weekly_tokens_used
            FROM users u
            ORDER BY u.created_at DESC
        ''')
        users = [dict(row) for row in cursor.fetchall()]

        for user in users:
            weekly = user.get('weekly_tokens_used', 0) or 0
            user['daily_avg_tokens'] = round(weekly / 7, 1) if weekly else 0

        return users


def use_search(user_id: int) -> bool:
    """검색횟수 1회 차감. 성공하면 True, 부족하면 False"""
    with get_db() as conn:
        cursor = conn.cursor()
        # 현재 검색횟수 확인
        cursor.execute('SELECT search_count FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        if not row or (row[0] is not None and row[0] <= 0):
            return False
        # 검색횟수 차감
        cursor.execute('''
            UPDATE users SET search_count = search_count - 1, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND (search_count IS NULL OR search_count > 0)
        ''', (user_id,))
        return cursor.rowcount > 0


def update_user_search_count(user_id: int, count: int):
    """사용자 검색횟수 수정 (관리자용)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET search_count = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (count, user_id))


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

        # 7. AI API 사용량
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


# ==================== 결제 기록 관리 ====================

def record_payment(user_id: int, amount: int, tier_granted: str, duration_days: int,
                   payment_method: str = '계좌이체', memo: str = None,
                   admin_id: int = None, paid_at: str = None) -> Optional[int]:
    """결제 기록 저장"""
    with get_db() as conn:
        cursor = conn.cursor()
        if not paid_at:
            paid_at = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''
            INSERT INTO payment_history (user_id, amount, payment_method, tier_granted,
                                         duration_days, memo, admin_id, paid_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, amount, payment_method, tier_granted, duration_days, memo, admin_id, paid_at))
        payment_id = cursor.lastrowid
        print(f"[DB] 결제 기록: user_id={user_id}, amount={amount}, tier={tier_granted}, days={duration_days}")
        return payment_id


def get_payment_history(user_id: int) -> List[Dict]:
    """특정 사용자의 결제 이력 조회"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ph.*, u.email as admin_email
            FROM payment_history ph
            LEFT JOIN users u ON ph.admin_id = u.id
            WHERE ph.user_id = ?
            ORDER BY ph.created_at DESC
        ''', (user_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_all_payments(limit: int = 100) -> List[Dict]:
    """전체 결제 이력 조회 (관리자용)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ph.*, u.email as user_email, u.name as user_name, u.company as user_company,
                   a.email as admin_email
            FROM payment_history ph
            JOIN users u ON ph.user_id = u.id
            LEFT JOIN users a ON ph.admin_id = a.id
            ORDER BY ph.created_at DESC
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]


def delete_payment(payment_id: int) -> bool:
    """결제 기록 삭제"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM payment_history WHERE id = ?', (payment_id,))
        return cursor.rowcount > 0


# ==================== 챗봇 최근 질문 관리 ====================

def log_chat_history(user_id: int, corp_code: str, corp_name: str,
                     question: str, response: str,
                     input_tokens: int = 0, output_tokens: int = 0):
    """챗봇 대화 이력 저장 (질문 + 응답)"""
    with get_db() as conn:
        cursor = conn.cursor()
        # 응답이 너무 길면 앞부분만 저장 (10000자)
        resp_trimmed = response[:10000] if response else ''
        cursor.execute('''
            INSERT INTO chat_history (user_id, corp_code, corp_name, question, response, input_tokens, output_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, corp_code, corp_name, question, resp_trimmed, input_tokens, output_tokens))


def save_recent_question(user_id, question, max_count=4):
    """최근 질문 저장 (중복 시 최신으로, 최대 max_count개)"""
    with get_db() as conn:
        cursor = conn.cursor()
        # 동일 질문 있으면 삭제 (최신으로 올림)
        cursor.execute('DELETE FROM chat_recent_questions WHERE user_id=? AND question=?', (user_id, question))
        # 새로 삽입
        cursor.execute('INSERT INTO chat_recent_questions (user_id, question) VALUES (?, ?)', (user_id, question))
        # max_count 초과 시 오래된 것 삭제
        cursor.execute('''
            DELETE FROM chat_recent_questions WHERE user_id=? AND id NOT IN (
                SELECT id FROM chat_recent_questions WHERE user_id=? ORDER BY id DESC LIMIT ?
            )
        ''', (user_id, user_id, max_count))


def get_recent_questions(user_id, limit=4):
    """최근 질문 조회 (최신순, [{id, question}])"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, question FROM chat_recent_questions WHERE user_id=? ORDER BY id DESC LIMIT ?', (user_id, limit))
        return [{'id': row[0], 'question': row[1]} for row in cursor.fetchall()]


def delete_recent_question(user_id, question_id):
    """특정 질문 삭제"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_recent_questions WHERE id=? AND user_id=?', (question_id, user_id))
        return cursor.rowcount > 0


# ==================== 게스트(비로그인) 사용량 관리 ====================

def get_guest_usage(ip_address: str) -> Dict:
    """게스트 사용량 조회 (없으면 생성)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM guest_usage WHERE ip_address = ?', (ip_address,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        # 없으면 생성
        cursor.execute('INSERT INTO guest_usage (ip_address) VALUES (?)', (ip_address,))
        return {'ip_address': ip_address, 'extract_used': 0, 'chat_used': 0}


def use_guest_extract(ip_address: str) -> bool:
    """게스트 추출 1회 사용. 성공하면 True, 한도 초과면 False (원자적)"""
    limit = int(get_setting('guest_extract_limit', '1'))
    with get_db() as conn:
        cursor = conn.cursor()
        # UPSERT + WHERE 조건으로 원자적 체크
        cursor.execute('''
            INSERT INTO guest_usage (ip_address, extract_used)
            VALUES (?, 1)
            ON CONFLICT(ip_address) DO UPDATE SET
                extract_used = extract_used + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE extract_used < ?
        ''', (ip_address, limit))
        return cursor.rowcount > 0


def use_guest_chat(ip_address: str) -> bool:
    """게스트 챗 1회 사용. 성공하면 True, 한도 초과면 False (원자적)"""
    limit = int(get_setting('guest_chat_limit', '5'))
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO guest_usage (ip_address, chat_used)
            VALUES (?, 1)
            ON CONFLICT(ip_address) DO UPDATE SET
                chat_used = chat_used + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE chat_used < ?
        ''', (ip_address, limit))
        return cursor.rowcount > 0


def get_guest_limits() -> Dict:
    """게스트 한도 설정값 조회"""
    return {
        'extract_limit': int(get_setting('guest_extract_limit', '1')),
        'chat_limit': int(get_setting('guest_chat_limit', '5')),
    }


# 초기화
if __name__ == '__main__':
    init_db()

    # 테스트용 관리자 계정 생성
    admin_id = create_user('admin@example.com', 'admin123', role='admin', tier='pro')
    if admin_id:
        print(f"관리자 계정 생성됨: admin@example.com / admin123")
