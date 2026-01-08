"""
DART 재무제표 추출 웹 서버

FastAPI를 사용한 백엔드 서버
- 기업 검색 API
- 재무제표 추출 API (진행상태 SSE)
- 파일 다운로드 API
- 작업 취소 API
"""

import os
import asyncio
import uuid
import threading
import math
from datetime import datetime
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import dart_fss as dart
import pandas as pd


# ============================================================
# 전역 변수: 작업 상태 관리
# ============================================================
# 작업 ID별 상태 저장: {task_id: {"status": "...", "progress": 0, "message": "...", "file_path": "...", "cancelled": False}}
TASKS: Dict[str, Dict[str, Any]] = {}

# 기업 리스트 캐시 (한 번만 로드)
CORP_LIST = None
CORP_LIST_LOCK = threading.Lock()


# ============================================================
# Pydantic 모델
# ============================================================
class SearchRequest(BaseModel):
    """기업 검색 요청"""
    company_name: str
    market: Optional[str] = None  # 'Y': 코스피, 'K': 코스닥, 'N': 코넥스, 'E': 기타


class ExtractRequest(BaseModel):
    """재무제표 추출 요청"""
    corp_code: str  # DART 고유번호
    corp_name: str  # 회사명 (파일명용)
    start_year: int = 2020
    end_year: Optional[int] = None


# ============================================================
# 유틸리티 함수
# ============================================================
def get_corp_list():
    """기업 리스트를 싱글톤으로 로드"""
    global CORP_LIST
    with CORP_LIST_LOCK:
        if CORP_LIST is None:
            api_key = os.environ.get('DART_API_KEY')
            if not api_key:
                raise ValueError("DART_API_KEY 환경변수가 설정되지 않았습니다.")
            dart.set_api_key(api_key)
            CORP_LIST = dart.get_corp_list()
        return CORP_LIST


def cleanup_task(task_id: str):
    """작업 정리 및 메모리 회수"""
    if task_id in TASKS:
        task = TASKS[task_id]
        # 임시 파일 삭제 (취소된 경우)
        if task.get('cancelled') and task.get('file_path'):
            try:
                if os.path.exists(task['file_path']):
                    os.remove(task['file_path'])
            except Exception:
                pass
        # 작업 정보는 다운로드 후 삭제하도록 유지
        if task.get('cancelled'):
            del TASKS[task_id]


# ============================================================
# FastAPI 앱 설정
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    # 시작 시: output 폴더 생성
    os.makedirs("output", exist_ok=True)
    yield
    # 종료 시: 모든 작업 정리
    for task_id in list(TASKS.keys()):
        TASKS[task_id]['cancelled'] = True
        cleanup_task(task_id)


app = FastAPI(title="DART 재무제표 추출기", lifespan=lifespan)


# ============================================================
# API 엔드포인트
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    """메인 페이지"""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/search")
async def search_company(request: SearchRequest):
    """기업 검색 API"""
    import traceback
    
    try:
        print(f"[DEBUG] 검색 요청: company_name='{request.company_name}', market='{request.market}'")
        
        corp_list = get_corp_list()
        print(f"[DEBUG] corp_list 로드 완료, 타입: {type(corp_list)}")
        
        # market이 None이거나 'None' 문자열이면 전체 시장 검색 ('YKNE')
        # Y: 코스피, K: 코스닥, N: 코넥스, E: 기타(비상장)
        market_filter = request.market
        if market_filter is None or market_filter == 'None' or market_filter == '':
            market_filter = 'YKNE'  # 전체 시장 (상장+비상장 모두)
        print(f"[DEBUG] market_filter 변환 후: {market_filter}")
        
        # 회사명으로 검색 (market=None이면 상장/비상장 모두 검색)
        companies = corp_list.find_by_corp_name(
            request.company_name, 
            exactly=False, 
            market=market_filter  # None이면 전체 검색 (기타법인 포함)
        )
        print(f"[DEBUG] 검색 결과 타입: {type(companies)}, 값: {companies}")
        
        # None 체크 및 리스트로 변환
        if companies is None:
            print("[DEBUG] companies가 None입니다")
            companies = []
        elif not isinstance(companies, list):
            print(f"[DEBUG] companies를 리스트로 변환: {type(companies)}")
            companies = [companies]
        
        print(f"[DEBUG] 처리할 기업 수: {len(companies)}")
        
        # 결과 포맷팅
        results = []
        for i, corp in enumerate(companies[:50]):  # 최대 50개
            if corp is None:
                print(f"[DEBUG] companies[{i}]가 None, 스킵")
                continue
            print(f"[DEBUG] 처리 중: {corp.corp_name}, corp_code={corp.corp_code}")
            results.append({
                "corp_code": corp.corp_code,
                "corp_name": corp.corp_name,
                "stock_code": corp.stock_code or "",
                "market": _get_market_name(corp)
            })
        
        print(f"[DEBUG] 최종 결과 수: {len(results)}")
        return {"success": True, "data": results, "count": len(results)}
    
    except ValueError as e:
        print(f"[ERROR] ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")


def _get_market_name(corp) -> str:
    """시장 구분명 반환"""
    if hasattr(corp, 'stock_code') and corp.stock_code:
        # 상장사 - 시장 구분 확인
        try:
            if hasattr(corp, 'corp_cls'):
                cls = corp.corp_cls
                if cls == 'Y':
                    return "코스피"
                elif cls == 'K':
                    return "코스닥"
                elif cls == 'N':
                    return "코넥스"
        except:
            pass
        return "상장"
    return "비상장"


@app.post("/api/extract")
async def start_extraction(request: ExtractRequest, background_tasks: BackgroundTasks):
    """재무제표 추출 시작 API"""
    # 작업 ID 생성
    task_id = str(uuid.uuid4())
    
    # 작업 상태 초기화
    TASKS[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "작업 대기 중...",
        "file_path": None,
        "cancelled": False,
        "corp_name": request.corp_name
    }
    
    # 백그라운드에서 추출 작업 실행
    background_tasks.add_task(
        extract_financial_data,
        task_id,
        request.corp_code,
        request.corp_name,
        request.start_year,
        request.end_year
    )
    
    return {"success": True, "task_id": task_id}


async def extract_financial_data(
    task_id: str,
    corp_code: str,
    corp_name: str,
    start_year: int,
    end_year: Optional[int]
):
    """재무제표 추출 백그라운드 작업"""
    import traceback
    
    try:
        print(f"[EXTRACT] 추출 시작: task_id={task_id}, corp_code={corp_code}, corp_name={corp_name}")
        
        task = TASKS.get(task_id)
        if not task:
            print(f"[EXTRACT] task를 찾을 수 없음: {task_id}")
            return
        
        # 취소 확인
        if task['cancelled']:
            print(f"[EXTRACT] 작업 취소됨: {task_id}")
            cleanup_task(task_id)
            return
        
        # 진행상태 업데이트: 시작
        task['status'] = 'running'
        task['progress'] = 10
        task['message'] = '재무제표 데이터 요청 중...'
        print(f"[EXTRACT] 상태 업데이트: running, 10%")
        
        # 종료 연도 설정
        if end_year is None:
            end_year = datetime.now().year
        
        start_date = f"{start_year}0101"
        end_date = f"{end_year}1231"
        print(f"[EXTRACT] 기간: {start_date} ~ {end_date}")
        
        # 취소 확인
        if task['cancelled']:
            cleanup_task(task_id)
            return
        
        # 진행상태 업데이트: 재무제표 추출
        task['progress'] = 30
        task['message'] = '재무상태표 추출 중...'
        print(f"[EXTRACT] DART API 호출 시작...")
        
        # 재무제표 추출 (동기 함수를 별도 스레드에서 실행)
        loop = asyncio.get_event_loop()
        
        # 진행률 콜백 함수
        def progress_callback(progress: int, message: str):
            task['progress'] = progress
            task['message'] = message
        
        # 회사 객체에서 직접 extract_fs() 호출 방식으로 변경
        fs_data = None
        try:
            fs_data = await loop.run_in_executor(
                None,
                lambda: extract_fs_from_corp(corp_code, start_date, end_date, progress_callback)
            )
            print(f"[EXTRACT] 재무제표 추출 성공")
        except Exception as e:
            print(f"[EXTRACT] 추출 실패: {e}")
            raise RuntimeError(f"재무제표를 찾을 수 없습니다. {str(e)}")
        
        print(f"[EXTRACT] DART API 호출 완료, fs_data 타입: {type(fs_data)}")
        
        # 취소 확인
        if task['cancelled']:
            cleanup_task(task_id)
            return
        
        task['progress'] = 70
        task['message'] = '엑셀 파일 생성 중...'
        
        # 엑셀 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{corp_name}_재무제표_{timestamp}.xlsx"
        filepath = os.path.join("output", filename)
        
        # 엑셀 저장
        await loop.run_in_executor(
            None,
            lambda: save_to_excel(fs_data, filepath)
        )
        
        # 취소 확인
        if task['cancelled']:
            cleanup_task(task_id)
            return
        
        # 완료
        task['status'] = 'completed'
        task['progress'] = 100
        task['message'] = '완료!'
        task['file_path'] = filepath
        task['filename'] = filename
        # VCM 포맷용 전체 데이터 저장 (XBRL 컬럼 정규화 적용)
        task['preview_data'] = {}
        print(f"[EXTRACT] fs_data 키: {list(fs_data.keys())}")
        
        # IS와 CIS 병합: 최대한 많은 데이터 수집
        is_df = fs_data.get('is')
        cis_df = fs_data.get('cis')
        is_empty = is_df is None or (hasattr(is_df, 'empty') and is_df.empty)
        cis_valid = cis_df is not None and hasattr(cis_df, 'empty') and not cis_df.empty
        print(f"[EXTRACT] is_empty={is_empty}, cis_valid={cis_valid}")

        # IS와 CIS 병합 로직
        if is_empty and cis_valid:
            # IS가 완전히 비어있으면 CIS로 시작
            fs_data['is'] = fs_data['cis']
            print(f"[EXTRACT] IS 데이터 없음, CIS를 IS로 사용")
        elif not is_empty and cis_valid:
            # IS와 CIS 모두 있으면 병합 (중복 제거)
            try:
                import pandas as pd
                # 컬럼 구조가 같은지 확인 후 병합
                is_accounts = set(is_df.index) if hasattr(is_df, 'index') else set()
                cis_accounts = set(cis_df.index) if hasattr(cis_df, 'index') else set()
                # CIS에만 있는 계정과목 추가
                unique_cis = cis_accounts - is_accounts
                if unique_cis:
                    cis_unique_df = cis_df.loc[list(unique_cis)]
                    fs_data['is'] = pd.concat([is_df, cis_unique_df], axis=0)
                    print(f"[EXTRACT] IS와 CIS 병합: CIS에서 {len(unique_cis)}개 계정과목 추가")
                else:
                    print(f"[EXTRACT] IS와 CIS 병합: 추가할 CIS 계정과목 없음")
            except Exception as e:
                print(f"[EXTRACT] IS/CIS 병합 실패: {e}, IS만 사용")
                # 병합 실패 시 IS만 사용
        
        for key in ['bs', 'is', 'cis', 'cf']:
            df = fs_data.get(key)
            print(f"[EXTRACT] fs_data[{key}]: 존재={df is not None}, empty={df.empty if df is not None else 'N/A'}, 행수={len(df) if df is not None and not df.empty else 0}")
            if df is not None and not df.empty:
                # XBRL 형식인 경우 컬럼 정규화 후 JSON 변환
                if key == 'cis':
                    print(f"[EXTRACT] CIS 원본 컬럼: {list(df.columns)[:10]}")
                df_normalized = normalize_xbrl_columns(df)
                if key == 'cis':
                    print(f"[EXTRACT] CIS 정규화 후 컬럼: {list(df_normalized.columns)}")
                task['preview_data'][key] = safe_dataframe_to_json(df_normalized)
                print(f"[EXTRACT] preview_data[{key}] 생성: {len(task['preview_data'][key])}개 행")

                # CIS 디버깅: 급여 관련 계정과목 확인
                if key == 'cis':
                    print(f"[EXTRACT] CIS 계정과목 샘플:")
                    for i, row in enumerate(task['preview_data']['cis'][:10]):
                        account = row.get('계정과목', 'N/A')
                        print(f"  [{i}] {account}")
                    print(f"[EXTRACT] CIS 급여 관련 계정과목:")
                    for row in task['preview_data']['cis']:
                        account = row.get('계정과목', '')
                        if '급여' in account or '수수료' in account or '임차' in account:
                            fy2024 = row.get('FY2024', 'N/A')
                            print(f"  - {account}: FY2024={fy2024}")
        
        # 주석 테이블들도 preview_data에 추가 (프론트엔드에서 세부항목 표시용)
        # 항상 Excel에서 주석 데이터 읽기 (fs_data['notes']가 없을 수 있으므로)
        print(f"[EXTRACT] Excel에서 주석 데이터 읽기...")
        import sys
        sys.stdout.flush()
        notes = {'is_notes': [], 'bs_notes': [], 'cf_notes': []}
        try:
            import pandas as pd
            xl = pd.ExcelFile(filepath)
            # 손익계산서 관련 주석 패턴 확장
            is_patterns = ['손익주석', '포괄손익주석', '손익계산서주석', 'IS주석', '손익', '포괄손익']
            # 재무상태표 관련 주석 패턴 확장
            bs_patterns = ['재무주석', '재무상태표주석', 'BS주석', '재무상태']
            # 현금흐름표 관련 주석 패턴
            cf_patterns = ['현금흐름주석', '현금흐름표주석', 'CF주석', '현금흐름']

            for sheet in xl.sheet_names:
                sheet_lower = sheet.lower()
                # IS 주석 매칭 (더 유연한 패턴)
                if any(pattern in sheet for pattern in is_patterns) or 'income' in sheet_lower or 'pl' in sheet_lower:
                    # 재무상태표나 현금흐름표 주석은 제외
                    if not any(pattern in sheet for pattern in bs_patterns + cf_patterns):
                        df = pd.read_excel(filepath, sheet_name=sheet)
                        if df is not None and not df.empty:
                            notes['is_notes'].append({'df': df, 'name': sheet})
                            print(f"[EXTRACT] IS 주석 발견: {sheet}")
                # BS 주석 매칭
                elif any(pattern in sheet for pattern in bs_patterns) or 'balance' in sheet_lower or 'bs' in sheet_lower:
                    df = pd.read_excel(filepath, sheet_name=sheet)
                    if df is not None and not df.empty:
                        notes['bs_notes'].append({'df': df, 'name': sheet})
                        print(f"[EXTRACT] BS 주석 발견: {sheet}")
                # CF 주석 매칭
                elif any(pattern in sheet for pattern in cf_patterns) or 'cash' in sheet_lower or 'cf' in sheet_lower:
                    df = pd.read_excel(filepath, sheet_name=sheet)
                    if df is not None and not df.empty:
                        notes['cf_notes'].append({'df': df, 'name': sheet})
                        print(f"[EXTRACT] CF 주석 발견: {sheet}")
            print(f"[EXTRACT] Excel 주석 로드 완료: IS={len(notes['is_notes'])}개, BS={len(notes['bs_notes'])}개, CF={len(notes['cf_notes'])}개")
            sys.stdout.flush()
        except Exception as e:
            print(f"[EXTRACT] Excel 주석 로드 실패: {e}")
            import traceback
            print(f"[EXTRACT] 상세: {traceback.format_exc()}")
            notes = None
        
        if notes and isinstance(notes, dict):
            # 손익계산서 주석들의 데이터를 is/cis DataFrame에 병합
            is_notes = notes.get('is_notes', [])
            print(f"[EXTRACT] is_notes 수: {len(is_notes)}")
            if is_notes:
                # fs_data의 is 또는 cis DataFrame에 병합
                target_key = 'cis' if fs_data.get('cis') is not None else 'is'
                if fs_data.get(target_key) is not None:
                    print(f"[EXTRACT] {target_key}에 주석 병합 시도...")
                    import pandas as pd
                    base_df = fs_data[target_key]

                    for note in is_notes:
                        try:
                            note_df = note['df']
                            if note_df is not None and not note_df.empty:
                                print(f"[EXTRACT] IS 주석 병합: shape={note_df.shape}")
                                # 계정과목 컬럼이 있는지 확인
                                if '계정과목' in note_df.columns:
                                    # 기존 계정과목 목록
                                    existing_accounts = set(base_df['계정과목'].values) if '계정과목' in base_df.columns else set()
                                    # 새로운 계정과목만 추가
                                    new_rows = note_df[~note_df['계정과목'].isin(existing_accounts)]
                                    if len(new_rows) > 0:
                                        base_df = pd.concat([base_df, new_rows], ignore_index=True)
                                        print(f"[EXTRACT] IS 주석 {len(new_rows)}개 행 추가")
                                        for acc in new_rows['계정과목'].head(5):
                                            print(f"[EXTRACT]   - {acc}")
                        except Exception as e:
                            print(f"[EXTRACT] IS 주석 병합 실패: {e}")

                    fs_data[target_key] = base_df
                    print(f"[EXTRACT] {target_key} 병합 완료: {len(base_df)}개 행")

                # preview_data도 업데이트
                if not task['preview_data'].get('is'):
                    task['preview_data']['is'] = []
                merged_is_data = list(task['preview_data']['is'])
                for note in is_notes:
                    try:
                        note_df = note['df']
                        if note_df is not None and not note_df.empty:
                            note_json = safe_dataframe_to_json(note_df)
                            existing_accounts = {row.get('계정과목', '') for row in merged_is_data}
                            for row in note_json:
                                account = row.get('계정과목', '')
                                if account and account not in existing_accounts:
                                    merged_is_data.append(row)
                                    existing_accounts.add(account)
                    except Exception as e:
                        pass
                task['preview_data']['is'] = merged_is_data
            
            # 재무상태표 주석들도 병합
            bs_notes = notes.get('bs_notes', [])
            if bs_notes and fs_data.get('bs') is not None:
                print(f"[EXTRACT] bs에 주석 병합 시도...")
                import pandas as pd
                base_df = fs_data['bs']

                for note in bs_notes:
                    try:
                        note_df = note['df']
                        if note_df is not None and not note_df.empty:
                            note_df_normalized = normalize_xbrl_columns(note_df)
                            if '계정과목' in note_df_normalized.columns:
                                existing_accounts = set(base_df['계정과목'].values) if '계정과목' in base_df.columns else set()
                                new_rows = note_df_normalized[~note_df_normalized['계정과목'].isin(existing_accounts)]
                                if len(new_rows) > 0:
                                    base_df = pd.concat([base_df, new_rows], ignore_index=True)
                                    print(f"[EXTRACT] BS 주석 {len(new_rows)}개 행 추가")
                    except Exception as e:
                        print(f"[EXTRACT] BS 주석 병합 실패: {e}")

                fs_data['bs'] = base_df
                print(f"[EXTRACT] BS 병합 완료: {len(base_df)}개 행")

                # preview_data도 업데이트
                if task['preview_data'].get('bs'):
                    merged_bs_data = list(task['preview_data']['bs'])
                    for note in bs_notes:
                        try:
                            note_df = note['df']
                            if note_df is not None and not note_df.empty:
                                note_df_normalized = normalize_xbrl_columns(note_df)
                                note_json = safe_dataframe_to_json(note_df_normalized)
                                existing_accounts = {row.get('계정과목', '') for row in merged_bs_data}
                                for row in note_json:
                                    account = row.get('계정과목', '')
                                    if account and account not in existing_accounts:
                                        merged_bs_data.append(row)
                                        existing_accounts.add(account)
                        except Exception as e:
                            pass
                    task['preview_data']['bs'] = merged_bs_data

        # ★ Excel 파일에서 모든 데이터를 다시 읽어서 preview_data 업데이트
        # (Excel에 저장된 데이터가 fs_data보다 더 완전함)
        try:
            print(f"[EXTRACT] Excel에서 preview_data 재생성...")
            xl = pd.ExcelFile(filepath)

            # BS, IS, CIS, CF 시트 매핑
            sheet_mapping = {
                'bs': '재무상태표',
                'is': '손익계산서',
                'cis': '포괄손익계산서',
                'cf': '현금흐름표'
            }

            for key, sheet_name in sheet_mapping.items():
                if sheet_name in xl.sheet_names:
                    try:
                        df = pd.read_excel(filepath, sheet_name=sheet_name)
                        if df is not None and not df.empty:
                            task['preview_data'][key] = safe_dataframe_to_json(df)
                            print(f"[EXTRACT] preview_data[{key}] Excel에서 재생성: {len(task['preview_data'][key])}개 행")

                            # CIS 디버깅
                            if key == 'cis':
                                print(f"[EXTRACT] CIS Excel 컬럼: {list(df.columns)}")
                                for row in task['preview_data']['cis']:
                                    account = row.get('계정과목', '')
                                    if '급여' in account and account == '급여':
                                        fy2024 = row.get('FY2024', 'N/A')
                                        print(f"[EXTRACT] 급여 FY2024 (Excel에서): {fy2024}")
                                        break
                    except Exception as e:
                        print(f"[EXTRACT] {key} Excel 읽기 실패: {e}")

            # VCM 포맷 데이터도 preview_data에 추가
            if 'VCM전용포맷' in xl.sheet_names:
                vcm_df = pd.read_excel(filepath, sheet_name='VCM전용포맷', engine='openpyxl')
                if vcm_df is not None and not vcm_df.empty:
                    task['preview_data']['vcm'] = safe_dataframe_to_json(vcm_df)
                    print(f"[EXTRACT] preview_data['vcm'] 생성: {len(task['preview_data']['vcm'])}개 행")

            # 복사용테이블 데이터도 preview_data에 추가
            if '복사용테이블' in xl.sheet_names:
                display_df = pd.read_excel(filepath, sheet_name='복사용테이블', engine='openpyxl')
                if display_df is not None and not display_df.empty:
                    task['preview_data']['vcm_display'] = safe_dataframe_to_json(display_df)
                    print(f"[EXTRACT] preview_data['vcm_display'] 생성: {len(task['preview_data']['vcm_display'])}개 행")
        except Exception as e:
            print(f"[EXTRACT] Excel preview_data 재생성 실패: {e}")

        print(f"[EXTRACT] 추출 완료: {filename}")
        
    except Exception as e:
        print(f"[EXTRACT ERROR] 추출 실패: {e}")
        print(f"[EXTRACT ERROR] Traceback:\n{traceback.format_exc()}")
        if task_id in TASKS:
            TASKS[task_id]['status'] = 'error'
            TASKS[task_id]['message'] = f'오류: {str(e)}'
            TASKS[task_id]['progress'] = 0


def extract_fs_from_corp(corp_code: str, start_date: str, end_date: str, progress_callback=None):
    """
    회사 객체에서 재무제표 추출
    
    1차: 사업보고서에서 추출 시도 (상장사)
    2차: 실패 시 감사보고서에서 추출 (비상장사)
    
    Args:
        progress_callback: 진행률 콜백 함수 (progress: int, message: str)
    """
    from dart_fss.filings import search as search_filings
    
    def update_progress(progress: int, message: str):
        if progress_callback:
            progress_callback(progress, message)
    
    print(f"[FS] 회사 검색: corp_code={corp_code}")
    update_progress(15, '회사 정보 조회 중...')
    
    corp_list = get_corp_list()
    corp = corp_list.find_by_corp_code(corp_code)
    
    if not corp:
        raise RuntimeError(f"회사를 찾을 수 없습니다: {corp_code}")
    
    print(f"[FS] 회사 찾음: {corp.corp_name}")
    print(f"[FS] 재무제표 추출 시작: {start_date} ~ {end_date}")

    # datetime import (함수 내에서 사용)
    import datetime

    fs_data = {'bs': None, 'is': None, 'cis': None, 'cf': None, 'notes': None}

    # 현재 연도 확인
    current_year = datetime.datetime.now().year
    
    # 1차: 사업보고서에서 추출 시도 (XBRL 데이터)
    xbrl_data = None
    has_current_year_annual = False  # 당해년도 사업보고서 존재 여부
    annual_report = None  # 사업보고서 객체 저장

    try:
        print(f"[FS] 사업보고서에서 추출 시도...")
        update_progress(20, '사업보고서 확인 중...')
        fs = corp.extract_fs(bgn_de=start_date, end_de=end_date, separate=False)

        if fs is not None:
            for key in ['bs', 'is', 'cis', 'cf']:
                try:
                    fs_data[key] = fs[key]
                except:
                    pass

            has_data = any(fs_data[key] is not None for key in fs_data)
            # 사업보고서 추출 직후 상태 로깅
            for key in ['bs', 'is', 'cis', 'cf']:
                df = fs_data.get(key)
                print(f"[FS] 사업보고서 fs_data[{key}]: {df is not None and not df.empty if df is not None else False}")
            if has_data:
                print(f"[FS] 사업보고서에서 추출 성공")
                xbrl_data = fs_data.copy()

                # fs 객체에서 주석 테이블 추출 시도 (비용의 성격별 분류 등)
                try:
                    notes_tables = {'is_notes': [], 'bs_notes': [], 'cf_notes': []}
                    # fs.tables 또는 fs.info 등에서 추가 테이블 검색
                    tables_to_check = []
                    if hasattr(fs, 'tables'):
                        tables_to_check = fs.tables if fs.tables else []
                    elif hasattr(fs, '_tables'):
                        tables_to_check = fs._tables if fs._tables else []

                    print(f"[FS] fs 객체 테이블 수: {len(tables_to_check)}")

                    for table in tables_to_check:
                        try:
                            if hasattr(table, 'to_DataFrame'):
                                temp_df = table.to_DataFrame()
                            elif hasattr(table, 'dataframe'):
                                temp_df = table.dataframe
                            else:
                                continue

                            if temp_df is None or temp_df.empty:
                                continue

                            col_str = str(temp_df.columns[0]) if len(temp_df.columns) > 0 else ""
                            is_consolidated = '연결' in col_str or 'Consolidated' in col_str

                            # 비용/수익 관련 주석 테이블 찾기
                            expense_keywords = ['비용', '수익', '매출', '판매비', '관리비', '원가', '급여', '감가상각', '성격별']
                            if any(kw in col_str for kw in expense_keywords):
                                if not any(kw in col_str for kw in ['재무상태표', '손익계산서', '현금흐름표', '포괄손익']):
                                    notes_tables['is_notes'].append({
                                        'name': col_str[:80],
                                        'df': temp_df,
                                        'consolidated': is_consolidated
                                    })
                                    print(f"[FS] 주석 테이블 발견: {col_str[:60]}... shape={temp_df.shape}")
                        except Exception as e:
                            pass

                    if any(notes_tables[key] for key in notes_tables):
                        fs_data['notes'] = notes_tables
                        print(f"[FS] 주석 테이블 추출 완료: IS={len(notes_tables['is_notes'])}개")
                except Exception as e:
                    print(f"[FS] fs 객체 주석 추출 실패: {e}")

                # 사업보고서 객체 찾기 (HTML 주석 추출용)
                try:
                    from dart_fss.filings import search as search_filings
                    annual_filings = search_filings(
                        corp_code=corp_code,
                        bgn_de=start_date,
                        end_de=end_date,
                        pblntf_ty='A'
                    )
                    if annual_filings:
                        annual_list = [f for f in annual_filings if '사업보고서' in str(f) and '반기' not in str(f) and '분기' not in str(f)]
                        if annual_list:
                            annual_report = annual_list[0]
                            print(f"[FS] 사업보고서 객체 찾음: {annual_report.report_nm}")
                except Exception as e:
                    print(f"[FS] 사업보고서 객체 검색 실패: {e}")

                # 사업보고서를 못 찾은 경우 감사보고서에서 주석 추출 시도 (비상장사용)
                if annual_report is None:
                    try:
                        print(f"[FS] 사업보고서 없음, 감사보고서에서 주석 추출 시도...")
                        audit_filings = search_filings(
                            corp_code=corp_code,
                            bgn_de=start_date,
                            end_de=end_date,
                            pblntf_ty=None
                        )
                        if audit_filings:
                            for af in audit_filings:
                                if '감사보고서' in str(af.report_nm) and '연결' in str(af.report_nm):
                                    annual_report = af
                                    print(f"[FS] 연결감사보고서 발견: {af.report_nm}")
                                    break
                            if annual_report is None:
                                for af in audit_filings:
                                    if '감사보고서' in str(af.report_nm):
                                        annual_report = af
                                        print(f"[FS] 감사보고서 발견: {af.report_nm}")
                                        break
                    except Exception as e:
                        print(f"[FS] 감사보고서 검색 실패: {e}")

                # 당해년도 사업보고서가 있는지 확인 (컬럼에서 FY{current_year} 존재 여부)
                for key in ['bs', 'is']:
                    df = fs_data.get(key)
                    if df is not None and not df.empty:
                        for col in df.columns:
                            col_str = str(col[0]) if isinstance(col, tuple) else str(col)
                            if str(current_year) in col_str:
                                has_current_year_annual = True
                                print(f"[FS] 당해년도({current_year}) 사업보고서 데이터 존재")
                                break
                    if has_current_year_annual:
                        break

                # 사업보고서 HTML 주석 페이지 추출
                if annual_report and hasattr(annual_report, 'pages'):
                    print(f"[FS] 사업보고서 HTML 주석 페이지 추출 시도...")
                    try:
                        # 보고서 연도 추출 (예: "사업보고서 (2023.12)" -> 2023)
                        import re
                        year_match = re.search(r'\((\d{4})', annual_report.report_nm)
                        report_year = int(year_match.group(1)) if year_match else current_year

                        html_notes = extract_fs_from_pages(annual_report, report_year)
                        if html_notes and html_notes.get('notes'):
                            # HTML 주석을 XBRL 주석과 병합
                            if fs_data.get('notes'):
                                # 기존 XBRL 주석이 있으면 병합
                                for note_type in ['is_notes', 'bs_notes', 'cf_notes']:
                                    if html_notes['notes'].get(note_type):
                                        fs_data['notes'][note_type].extend(html_notes['notes'][note_type])
                            else:
                                # 주석이 없으면 그대로 사용
                                fs_data['notes'] = html_notes['notes']
                            print(f"[FS] HTML 주석 병합 완료: IS={len(fs_data['notes'].get('is_notes', []))}개")

                            # ★ 핵심: 주석 데이터를 실제 DataFrame에 병합 (Excel 저장 전에 반영)
                            import pandas as pd

                            # 손익계산서 주석을 cis 또는 is DataFrame에 병합
                            is_notes = fs_data['notes'].get('is_notes', [])
                            if is_notes:
                                # cis 우선, 없으면 is 사용
                                target_key = 'cis' if fs_data.get('cis') is not None else 'is'
                                if fs_data.get(target_key) is not None:
                                    print(f"[FS] {target_key}에 HTML 주석 병합 시도...")
                                    base_df = fs_data[target_key]

                                    # ★ 핵심 수정: base_df를 먼저 정규화하여 컬럼 구조 통일
                                    base_df_normalized = normalize_xbrl_columns(base_df)

                                    for note in is_notes:
                                        try:
                                            note_df = note['df']
                                            if note_df is not None and not note_df.empty:
                                                print(f"[FS] IS 주석 병합: shape={note_df.shape}")
                                                # 계정과목 컬럼이 있는지 확인
                                                if '계정과목' in note_df.columns:
                                                    # 기존 계정과목 목록
                                                    existing_accounts = set(base_df_normalized['계정과목'].values) if '계정과목' in base_df_normalized.columns else set()
                                                    # 새로운 계정과목만 추가
                                                    new_rows = note_df[~note_df['계정과목'].isin(existing_accounts)]
                                                    if len(new_rows) > 0:
                                                        # 정규화된 DataFrame에 병합
                                                        base_df_normalized = pd.concat([base_df_normalized, new_rows], ignore_index=True)
                                                        print(f"[FS] IS 주석 {len(new_rows)}개 행 추가")
                                                        for acc in new_rows['계정과목'].head(5):
                                                            print(f"[FS]   - {acc}")
                                        except Exception as e:
                                            print(f"[FS] IS 주석 병합 실패: {e}")

                                    fs_data[target_key] = base_df_normalized
                                    print(f"[FS] {target_key} 병합 완료: {len(base_df_normalized)}개 행")

                            # 재무상태표 주석을 bs DataFrame에 병합
                            bs_notes = fs_data['notes'].get('bs_notes', [])
                            if bs_notes and fs_data.get('bs') is not None:
                                print(f"[FS] bs에 HTML 주석 병합 시도...")
                                base_df = fs_data['bs']

                                # ★ 핵심 수정: base_df를 먼저 정규화하여 컬럼 구조 통일
                                base_df_normalized = normalize_xbrl_columns(base_df)

                                for note in bs_notes:
                                    try:
                                        note_df = note['df']
                                        if note_df is not None and not note_df.empty:
                                            note_df_normalized = normalize_xbrl_columns(note_df)
                                            if '계정과목' in note_df_normalized.columns:
                                                existing_accounts = set(base_df_normalized['계정과목'].values) if '계정과목' in base_df_normalized.columns else set()
                                                new_rows = note_df_normalized[~note_df_normalized['계정과목'].isin(existing_accounts)]
                                                if len(new_rows) > 0:
                                                    base_df_normalized = pd.concat([base_df_normalized, new_rows], ignore_index=True)
                                                    print(f"[FS] BS 주석 {len(new_rows)}개 행 추가")
                                    except Exception as e:
                                        print(f"[FS] BS 주석 병합 실패: {e}")

                                fs_data['bs'] = base_df_normalized
                                print(f"[FS] BS 병합 완료: {len(base_df_normalized)}개 행")

                    except Exception as e:
                        print(f"[FS] HTML 주석 추출 실패: {e}")
                        import traceback
                        print(f"[FS] 상세: {traceback.format_exc()}")

                # 당해년도 사업보고서가 있으면 바로 반환
                if has_current_year_annual:
                    update_progress(80, '데이터 처리 중...')
                    return fs_data

                # 당해년도 사업보고서가 없으면 분기/반기 데이터 추가 조회
                print(f"[FS] 당해년도({current_year}) 사업보고서 없음, 분기/반기 보고서 추가 검색...")
    except Exception as e:
        print(f"[FS] 사업보고서 추출 실패: {e}")
    
    # 2차: 당해년도 분기/반기 보고서 추가 검색 (상장사: XBRL에 병합 / 비상장사: 감사보고서+분기)
    # XBRL 데이터가 있으면 당해년도 분기/반기만 검색, 없으면 감사보고서도 검색
    if xbrl_data:
        print(f"[FS] 상장사: 당해년도 분기/반기 보고서만 추가 검색...")
    else:
        print(f"[FS] 비상장사: 공시보고서에서 추출 시도...")
    update_progress(25, '공시보고서 검색 중...')
    
    # 연도별 데이터 저장용 딕셔너리 (키: 연도 또는 "2025 3Q" 형식)
    yearly_data = {'bs': {}, 'is': {}, 'cis': {}, 'cf': {}}
    
    try:
        # 1. 감사보고서 검색 (비상장사용, XBRL 데이터가 없을 때만)
        audit_filings = []
        if not xbrl_data:
            filings = search_filings(
                corp_code=corp_code,
                bgn_de=start_date,
                end_de=end_date,
                pblntf_ty='F',  # F: 감사보고서
            )
            audit_filings = list(filings) if filings else []
            print(f"[FS] 검색된 감사보고서 수: {len(audit_filings)}")
        
        # 2. 당해년도 분기/반기 보고서 검색 (가장 최신 것만 사용)
        latest_periodic = None
        try:
            # 분기 보고서 검색 기간: end_year 다음 연도부터 오늘까지 검색
            # 예: 2024년까지 요청 시 2025년 1월 ~ 오늘까지 검색하여 2025 3Q 등을 찾음
            today_str = datetime.datetime.now().strftime('%Y%m%d')
            end_year = int(end_date[:4])
            next_year = end_year + 1

            # 다음 연도가 현재 연도를 초과하면 검색하지 않음
            if next_year > current_year:
                print(f"[FS] 분기/반기 보고서 검색 건너뜀: 다음 연도({next_year})가 현재 연도({current_year})를 초과")
            else:
                periodic_filings = search_filings(
                    corp_code=corp_code,
                    bgn_de=f"{next_year}0101",
                    end_de=today_str,  # 항상 오늘까지 검색
                    pblntf_ty='A',  # A: 정기공시 (사업보고서, 반기, 분기)
                )
            if periodic_filings:
                # 분기/반기 보고서만 필터링하고 가장 최신 것 선택
                periodic_list = [f for f in periodic_filings if '반기' in str(f) or '분기' in str(f)]
                if periodic_list:
                    # 가장 최신 보고서 선택 (리스트의 첫 번째가 최신)
                    latest_periodic = periodic_list[0]
                    print(f"[FS] 당해년도 최신 보고서: {latest_periodic}")
        except Exception as e:
            print(f"[FS] 분기/반기 보고서 검색 실패: {e}")
        
        # 감사보고서 + 당해년도 최신 분기/반기 보고서 합치기
        filings = audit_filings
        if latest_periodic:
            filings = filings + [latest_periodic]
        
        total_filings = len(filings) if filings else 0
        print(f"[FS] 처리할 총 보고서 수: {total_filings}")
        
        if filings and total_filings > 0:
            import re
            for idx, filing in enumerate(filings):
                # 진행률 계산: 30% ~ 75% 구간에서 감사보고서 처리
                progress = 30 + int((idx / total_filings) * 45)
                update_progress(progress, f'감사보고서 처리 중... ({idx + 1}/{total_filings})')
                try:
                    # Report 객체 정보 출력
                    report_name = str(filing)
                    print(f"[FS] 보고서 처리: {report_name}")
                    
                    # 보고서에서 연도/기간 추출
                    # 감사보고서: "감사보고서 (2024.12)" -> 2024
                    # 반기보고서: "반기보고서 (2025.06)" -> "2025 반기"
                    # 분기보고서: "분기보고서 (2025.09)" -> "2025 3Q"
                    report_period = None
                    period_match = re.search(r'\((\d{4})\.(\d{2})\)', report_name)
                    if period_match:
                        year = int(period_match.group(1))
                        month = int(period_match.group(2))
                        
                        if '감사보고서' in report_name or month == 12:
                            # 연간 결산
                            report_period = year
                        elif '반기' in report_name:
                            # 반기보고서: "2025 반기"
                            report_period = f"{year} 반기"
                        elif '분기' in report_name:
                            # 분기보고서: 월에 따라 1Q, 2Q, 3Q 결정
                            quarter = (month - 1) // 3 + 1
                            if quarter == 4:
                                quarter = 3  # 4Q는 연간결산이므로 3Q로
                            report_period = f"{year} {quarter}Q"
                        else:
                            # 기타: 월 표시
                            report_period = f"{year} {month}월"
                        print(f"[FS] 보고서 기간: {report_period}")
                    
                    # Report 객체에서 XBRL 데이터 추출 시도
                    xbrl = filing.xbrl
                    if xbrl is not None:
                        print(f"[FS] XBRL 데이터 발견")
                        extracted = extract_fs_from_xbrl(xbrl)
                        print(f"[FS] extracted 결과: {[(k, v is not None) for k, v in extracted.items()]}")
                        print(f"[FS] report_period={report_period}, current_year={current_year}")
                        if extracted:
                            for key in ['bs', 'is', 'cis', 'cf']:
                                if extracted.get(key) is not None:
                                    print(f"[FS] {key} 데이터 있음, 분기 체크: {report_period}, {str(current_year) in str(report_period) if report_period else False}")
                                    # 분기 데이터인 경우 yearly_data에 저장 (상장사 XBRL 병합용)
                                    if report_period and str(current_year) in str(report_period):
                                        yearly_data[key][report_period] = extracted[key]
                                        print(f"[FS] 분기 XBRL 데이터 저장: {key} -> {report_period}")
                                    elif fs_data[key] is None:
                                        fs_data[key] = extracted[key]
                                    else:
                                        fs_data[key] = pd.concat([fs_data[key], extracted[key]], ignore_index=True).drop_duplicates()
                            # 주석 테이블도 fs_data에 추가
                            if extracted.get('notes'):
                                fs_data['notes'] = extracted['notes']
                                print(f"[FS] 주석 데이터 저장: IS={len(extracted['notes'].get('is_notes', []))}개, BS={len(extracted['notes'].get('bs_notes', []))}개")
                    else:
                        print(f"[FS] XBRL 없음, 페이지에서 추출 시도...")
                        # 페이지에서 재무제표 테이블 추출 시도
                        extracted = extract_fs_from_pages(filing, report_period)
                        if extracted:
                            for key in ['bs', 'is', 'cis', 'cf']:
                                if extracted.get(key) is not None:
                                    df = extracted[key]
                                    if report_period and len(df.columns) >= 2:
                                        # 기간별로 데이터 저장
                                        yearly_data[key][report_period] = df
                            
                except Exception as e:
                    print(f"[FS] 감사보고서 처리 실패: {e}")
                    import traceback
                    print(f"[FS] 상세 오류: {traceback.format_exc()}")
                    continue
    except Exception as e:
        print(f"[FS] 감사보고서 검색 실패: {e}")
    
    # 연도별 데이터를 합쳐서 정규화된 형태로 변환
    update_progress(80, '데이터 정규화 중...')
    
    # XBRL 데이터가 있는 상장사: 분기 데이터를 XBRL에 병합
    if xbrl_data and any(yearly_data[key] for key in yearly_data):
        print(f"[FS] 상장사: XBRL 데이터에 분기 데이터 병합...")
        print(f"[FS] yearly_data 키: {[(k, list(v.keys())) for k, v in yearly_data.items() if v]}")
        
        for key in ['bs', 'is', 'cis', 'cf']:
            # xbrl_data[key]가 None이고 yearly_data에 데이터가 있으면 분기 데이터를 기본으로 사용
            if yearly_data[key] and xbrl_data.get(key) is None:
                first_period = list(yearly_data[key].keys())[0]
                xbrl_data[key] = yearly_data[key][first_period]
                print(f"[FS] {key}: XBRL 데이터 없음, 분기 데이터({first_period})를 기본으로 사용")
            
            if yearly_data[key] and xbrl_data.get(key) is not None:
                xbrl_df = xbrl_data[key]
                print(f"[FS] {key}: XBRL 컬럼 = {list(xbrl_df.columns)[:5]}...")
                
                # 분기 XBRL DataFrame에서 당해년도 컬럼 추출하여 XBRL에 추가
                for period_key, quarterly_df in yearly_data[key].items():
                    # period_key 예: "2025 3Q" -> new_col_name: "FY2025 3Q"
                    new_col_name = f"FY{period_key}"
                    print(f"[FS] {key}: 분기 데이터 {period_key} 처리")
                    print(f"[FS] {key}: 분기 컬럼 전체 = {list(quarterly_df.columns)}")
                    
                    # 분기 DataFrame에서 연결재무제표의 당해년도 컬럼 찾기
                    # 우선순위: 1) 당해년도 2) 연결재무제표 3) 누적기간(01월 시작)
                    target_col = None
                    target_col_cumulative = None  # 누적기간 컬럼
                    quarterly_label_col = None
                    for col in quarterly_df.columns:
                        # 튜플 컬럼인 경우 두 번째 요소가 label_ko인지 확인
                        if isinstance(col, tuple):
                            if len(col) >= 2 and 'label_ko' in str(col[1]):
                                quarterly_label_col = col
                            # 첫 번째 요소에 당해년도가 있는지 확인
                            col_str = str(col[0])
                            if str(current_year) in col_str:
                                is_consolidated = len(col) > 1 and '연결' in str(col[1])
                                # 누적기간 확인 (20250101-20250930 형식에서 01월 시작)
                                is_cumulative = '-' in col_str and col_str[4:6] == '01'
                                # 재무상태표는 시점 데이터 (20250930)
                                is_point_data = '-' not in col_str and len(col_str) == 8
                                
                                if is_consolidated:
                                    if is_cumulative:
                                        target_col_cumulative = col
                                    elif is_point_data:
                                        # 재무상태표용 시점 데이터
                                        if target_col is None:
                                            target_col = col
                                    elif target_col is None:
                                        target_col = col
                        else:
                            col_str = str(col)
                            if 'label_ko' in col_str:
                                quarterly_label_col = col
                            if str(current_year) in col_str:
                                target_col = col
                    
                    # 누적기간 컬럼이 있으면 우선 사용 (손익계산서, 현금흐름표용)
                    if target_col_cumulative is not None:
                        target_col = target_col_cumulative
                    
                    print(f"[FS] {key}: quarterly_label_col = {quarterly_label_col}")
                    print(f"[FS] {key}: target_col = {target_col}")
                    
                    if target_col is not None and quarterly_label_col is not None:
                        print(f"[FS] 당해년도 컬럼 발견: {target_col}")
                        
                        # XBRL DataFrame의 label_ko 컬럼 찾기
                        xbrl_label_col = None
                        for col in xbrl_df.columns:
                            col_str = str(col) if not isinstance(col, tuple) else str(col)
                            if 'label_ko' in col_str:
                                xbrl_label_col = col
                                break
                        
                        if xbrl_label_col is not None:
                            # 계정과목 기준으로 분기 값 매핑
                            # 분기 데이터의 계정과목-값 딕셔너리 생성
                            quarterly_values = dict(zip(
                                quarterly_df[quarterly_label_col].astype(str),
                                quarterly_df[target_col]
                            ))
                            
                            # XBRL DataFrame에 새 컬럼 추가 (계정과목 매칭)
                            if new_col_name not in xbrl_df.columns:
                                xbrl_df[new_col_name] = xbrl_df[xbrl_label_col].apply(
                                    lambda x: quarterly_values.get(str(x), None)
                                )
                                print(f"[FS] 새 컬럼 추가 완료: {new_col_name}, 매칭된 값: {xbrl_df[new_col_name].notna().sum()}개")
                            else:
                                print(f"[FS] 컬럼 이미 존재: {new_col_name}")
                        else:
                            print(f"[FS] XBRL label_ko 컬럼을 찾을 수 없음")
                    else:
                        print(f"[FS] 분기 데이터에서 필요한 컬럼을 찾을 수 없음")
                
                xbrl_data[key] = xbrl_df
        
        # notes 데이터 유지 (분기 보고서에서 추출한 주석)
        # ★ 중요: HTML 주석으로 병합된 DataFrame도 유지해야 함!
        notes_backup = fs_data.get('notes')

        # HTML 주석으로 확장된 DataFrame이 있으면 xbrl_data 대신 사용
        import pandas as pd
        for key in ['bs', 'is', 'cis', 'cf']:
            fs_df = fs_data.get(key)
            xbrl_df = xbrl_data.get(key)

            # fs_data에 데이터가 있고, xbrl_data보다 행이 더 많으면 (HTML 주석 병합됨)
            if fs_df is not None and xbrl_df is not None:
                fs_row_count = len(fs_df) if hasattr(fs_df, '__len__') else 0
                xbrl_row_count = len(xbrl_df) if hasattr(xbrl_df, '__len__') else 0
                print(f"[FS] {key}: fs_row_count={fs_row_count}, xbrl_row_count={xbrl_row_count}")

                if fs_row_count >= xbrl_row_count:
                    print(f"[FS] {key}: HTML 주석 병합 데이터 유지 ({fs_row_count}행 > {xbrl_row_count}행)")
                    # xbrl_df를 정규화하여 fs_df와 컬럼 구조 통일
                    xbrl_df_normalized = normalize_xbrl_columns(xbrl_df)
                    fs_df_normalized = normalize_xbrl_columns(fs_df)  # fs_df도 정규화 (이미 정규화되어있으면 그대로)

                    # xbrl_df의 새 컬럼들을 fs_df에 추가
                    if '계정과목' in fs_df_normalized.columns and '계정과목' in xbrl_df_normalized.columns:
                        for col in xbrl_df_normalized.columns:
                            # 계정과목 컬럼 자체는 스킵
                            if col == '계정과목':
                                continue
                            # 이미 있는 컬럼은 스킵 (중복 방지)
                            if col in fs_df_normalized.columns:
                                continue

                            # 계정과목 기준 매칭
                            try:
                                # xbrl_df_normalized에서 해당 컬럼이 중복되어 있으면 첫 번째만 사용
                                if isinstance(xbrl_df_normalized[col], pd.DataFrame):
                                    # 중복 컬럼인 경우 첫 번째만 사용
                                    xbrl_col_data = xbrl_df_normalized[col].iloc[:, 0]
                                else:
                                    xbrl_col_data = xbrl_df_normalized[col]

                                col_values = dict(zip(xbrl_df_normalized['계정과목'], xbrl_col_data))
                                fs_df_normalized[col] = fs_df_normalized['계정과목'].map(col_values)
                                print(f"[FS] {key}: 분기 컬럼 추가 - {col}")
                            except Exception as e:
                                print(f"[FS] {key}: 컬럼 {col} 추가 실패 - {e}")
                    xbrl_data[key] = fs_df_normalized

        fs_data = xbrl_data
        if notes_backup:
            fs_data['notes'] = notes_backup
            print(f"[FS] 주석 데이터 유지: IS={len(notes_backup.get('is_notes', []))}개, BS={len(notes_backup.get('bs_notes', []))}개")
        print(f"[FS] 상장사: 병합 완료")
        return fs_data
    
    # 비상장사: 감사보고서 데이터 정규화
    for key in ['bs', 'is', 'cis', 'cf']:
        if yearly_data[key]:
            fs_data[key] = normalize_yearly_data(yearly_data[key], key)
    
    # 최소한 하나의 재무제표라도 있는지 확인
    has_data = any(fs_data[key] is not None for key in fs_data)
    if not has_data:
        raise RuntimeError("사업보고서와 감사보고서 모두에서 재무제표를 찾을 수 없습니다.")
    
    print(f"[FS] 감사보고서에서 추출 완료")
    return fs_data


def extract_fs_from_xbrl(xbrl):
    """XBRL 데이터에서 재무제표 추출 (모든 주석 테이블 포함)"""
    fs_data = {'bs': None, 'is': None, 'cis': None, 'cf': None}
    # 주석 테이블들 (재무제표별로 분류)
    notes_tables = {
        'bs_notes': [],   # 재무상태표 관련 주석
        'is_notes': [],   # 손익계산서 관련 주석
        'cf_notes': [],   # 현금흐름표 관련 주석
        'other_notes': [] # 기타 주석
    }
    
    try:
        print(f"[XBRL Extract] xbrl 타입: {type(xbrl)}")
        
        # 방법 0: xbrl.tables에서 모든 테이블을 가져와 컬럼명으로 분류 (분기보고서용)
        if hasattr(xbrl, 'tables') and xbrl.tables:
            print(f"[XBRL Extract] tables 속성 발견, 테이블 수: {len(xbrl.tables)}")
            
            for table in xbrl.tables:
                try:
                    if hasattr(table, 'to_DataFrame'):
                        temp_df = table.to_DataFrame()
                    elif hasattr(table, 'dataframe'):
                        temp_df = table.dataframe
                    else:
                        continue
                    
                    if temp_df is None or temp_df.empty:
                        continue
                    
                    # 컬럼명으로 재무제표 타입 식별
                    col_str = str(temp_df.columns[0]) if len(temp_df.columns) > 0 else ""
                    
                    # 연결재무제표 우선 (D21xxxx는 별도, D210000은 연결)
                    is_consolidated = '연결' in col_str or 'Consolidated' in col_str
                    
                    if '재무상태표' in col_str or 'financial position' in col_str.lower():
                        if fs_data['bs'] is None or is_consolidated:
                            fs_data['bs'] = temp_df
                            print(f"[XBRL Extract] bs 테이블 발견 (tables): {temp_df.shape}, 연결={is_consolidated}")
                    elif '포괄손익' in col_str or 'comprehensive income' in col_str.lower():
                        # 포괄손익계산서를 먼저 체크 (손익계산서보다 먼저)
                        if fs_data['cis'] is None or is_consolidated:
                            fs_data['cis'] = temp_df
                            print(f"[XBRL Extract] cis 테이블 발견 (tables): {temp_df.shape}, 연결={is_consolidated}")
                    elif ('손익계산서' in col_str or 'income statement' in col_str.lower()) and '포괄' not in col_str:
                        # 손익계산서 (포괄손익 제외)
                        if fs_data['is'] is None or is_consolidated:
                            fs_data['is'] = temp_df
                            print(f"[XBRL Extract] is 테이블 발견 (tables): {temp_df.shape}, 연결={is_consolidated}")
                    elif '현금흐름' in col_str or 'cash flow' in col_str.lower():
                        if fs_data['cf'] is None or is_consolidated:
                            fs_data['cf'] = temp_df
                            print(f"[XBRL Extract] cf 테이블 발견 (tables): {temp_df.shape}, 연결={is_consolidated}")
                    
                    # 주석 테이블 분류 (기본 재무제표가 아닌 모든 테이블)
                    # 손익계산서 관련 주석
                    is_note_table = False
                    if any(kw in col_str for kw in ['수익', '매출', '판매비', '관리비', '비용', '원가', '세분화', '영업']):
                        if not any(kw in col_str for kw in ['재무상태표', '손익계산서', '현금흐름표', '포괄손익']):
                            notes_tables['is_notes'].append({'name': col_str[:80], 'df': temp_df, 'consolidated': is_consolidated})
                            print(f"[XBRL Extract] 주석(IS관련) 발견: {col_str[:60]}... shape={temp_df.shape}")
                            is_note_table = True
                    # 재무상태표 관련 주석
                    if any(kw in col_str for kw in ['자산', '부채', '자본', '투자', '재고', '채권', '채무']):
                        if not any(kw in col_str for kw in ['재무상태표', '손익계산서', '현금흐름표', '포괄손익']):
                            if not is_note_table:
                                notes_tables['bs_notes'].append({'name': col_str[:80], 'df': temp_df, 'consolidated': is_consolidated})
                                print(f"[XBRL Extract] 주석(BS관련) 발견: {col_str[:60]}... shape={temp_df.shape}")
                                is_note_table = True
                    # 현금흐름표 관련 주석
                    if any(kw in col_str for kw in ['현금', '금융', '이자', '배당']):
                        if not any(kw in col_str for kw in ['재무상태표', '손익계산서', '현금흐름표', '포괄손익']):
                            if not is_note_table:
                                notes_tables['cf_notes'].append({'name': col_str[:80], 'df': temp_df, 'consolidated': is_consolidated})
                                print(f"[XBRL Extract] 주석(CF관련) 발견: {col_str[:60]}... shape={temp_df.shape}")
                            
                except Exception as e:
                    print(f"[XBRL Extract] tables 처리 중 오류: {e}")
        
        # 방법 1: get_financial_statement 메서드 사용 (tables에서 못 찾은 경우)
        if hasattr(xbrl, 'get_financial_statement'):
            for fs_type in ['bs', 'is', 'cis', 'cf']:
                if fs_data[fs_type] is not None:
                    continue  # 이미 tables에서 찾은 경우 스킵
                try:
                    result = xbrl.get_financial_statement(fs_type)
                    df = None
                    
                    # 결과가 리스트인 경우 (분기보고서 등)
                    if isinstance(result, list) and len(result) > 0:
                        print(f"[XBRL Extract] {fs_type}: 리스트 반환됨, 길이={len(result)}")
                        # 리스트에서 올바른 재무제표 찾기
                        for item in result:
                            if hasattr(item, 'to_DataFrame'):
                                temp_df = item.to_DataFrame()
                            elif hasattr(item, 'dataframe'):
                                temp_df = item.dataframe
                            elif isinstance(item, pd.DataFrame):
                                temp_df = item
                            else:
                                continue
                            
                            if temp_df is not None and not temp_df.empty:
                                # 컬럼명으로 재무제표 타입 확인
                                col_str = str(temp_df.columns[0]) if len(temp_df.columns) > 0 else ""
                                is_match = False
                                
                                if fs_type == 'bs' and ('재무상태표' in col_str or 'financial position' in col_str.lower()):
                                    is_match = True
                                elif fs_type == 'is' and ('손익계산서' in col_str or 'income statement' in col_str.lower()):
                                    is_match = True
                                elif fs_type == 'cis' and ('포괄손익' in col_str or 'comprehensive income' in col_str.lower()):
                                    is_match = True
                                elif fs_type == 'cf' and ('현금흐름' in col_str or 'cash flow' in col_str.lower()):
                                    is_match = True
                                
                                if is_match:
                                    df = temp_df
                                    print(f"[XBRL Extract] {fs_type}: 매칭된 테이블 발견")
                                    break
                        
                        if df is not None:
                            fs_data[fs_type] = df
                            print(f"[XBRL Extract] {fs_type} 추출 성공 (리스트): {df.shape}")
                    
                    elif isinstance(result, pd.DataFrame) and not result.empty:
                        fs_data[fs_type] = result
                        print(f"[XBRL Extract] {fs_type} 추출 성공: {result.shape}")
                        
                except Exception as e:
                    print(f"[XBRL Extract] {fs_type} get_financial_statement 실패: {e}")
        
        # 방법 2: 직접 속성 접근 (분기보고서용)
        if all(v is None for v in fs_data.values()):
            print(f"[XBRL Extract] get_financial_statement 실패, 직접 속성 접근 시도...")
            fs_mapping = {'bs': 'balance_sheet', 'is': 'income_statement', 'cis': 'comprehensive_income', 'cf': 'cash_flow'}
            for fs_type, attr_name in fs_mapping.items():
                if hasattr(xbrl, attr_name):
                    try:
                        df = getattr(xbrl, attr_name)
                        if df is not None and not df.empty:
                            fs_data[fs_type] = df
                            print(f"[XBRL Extract] {fs_type} ({attr_name}) 추출 성공: {df.shape}")
                    except Exception as e:
                        print(f"[XBRL Extract] {fs_type} ({attr_name}) 실패: {e}")
        
        # 방법 3: 인덱싱 접근
        if all(v is None for v in fs_data.values()):
            print(f"[XBRL Extract] 직접 속성 실패, 인덱싱 시도...")
            for fs_type in ['bs', 'is', 'cis', 'cf']:
                try:
                    df = xbrl[fs_type]
                    if df is not None and not df.empty:
                        fs_data[fs_type] = df
                        print(f"[XBRL Extract] {fs_type} 인덱싱 추출 성공: {df.shape}")
                except Exception as e:
                    print(f"[XBRL Extract] {fs_type} 인덱싱 실패: {e}")
                    
    except Exception as e:
        print(f"[XBRL Extract] 추출 실패: {e}")
        import traceback
        print(f"[XBRL Extract] 상세: {traceback.format_exc()}")
    
    # 주석 테이블들을 fs_data에 추가
    fs_data['notes'] = notes_tables
    print(f"[XBRL Extract] 주석 테이블 수: BS={len(notes_tables['bs_notes'])}, IS={len(notes_tables['is_notes'])}, CF={len(notes_tables['cf_notes'])}")
    return fs_data


def extract_fs_from_pages(report, report_year=None):
    """
    Report 객체의 페이지에서 재무제표 테이블 추출
    감사보고서 페이지 구조를 파싱하여 재무제표를 DataFrame으로 변환

    Args:
        report: Report 객체
        report_year: 보고서 연도 (예: 2024)
    """
    import re
    from bs4 import BeautifulSoup

    fs_data = {'bs': None, 'is': None, 'cis': None, 'cf': None}
    notes_tables = {'bs_notes': [], 'is_notes': [], 'cf_notes': [], 'other_notes': []}

    # 재무제표 관련 페이지 찾기
    fs_keywords = {
        'bs': ['재무상태표', '대차대조표', 'Statement of Financial Position', 'Balance Sheet'],
        'is': ['손익계산서', 'Income Statement', 'Statement of Income'],
        'cis': ['포괄손익계산서', 'Comprehensive Income'],
        'cf': ['현금흐름표', 'Cash Flow', 'Statement of Cash Flows']
    }

    try:
        pages = report.pages
        print(f"[PAGES] 총 페이지 수: {len(pages)}")

        for page in pages:
            page_title = page.title if hasattr(page, 'title') else str(page)
            # 공백 제거한 제목으로 매칭 (예: "재 무 상 태 표" -> "재무상태표")
            page_title_normalized = re.sub(r'\s+', '', page_title)
            print(f"[PAGES] 페이지: {page_title} -> {page_title_normalized}")

            # 어떤 재무제표 유형인지 확인
            for fs_type, keywords in fs_keywords.items():
                if fs_data[fs_type] is not None:
                    continue  # 이미 추출됨

                for keyword in keywords:
                    # 키워드도 공백 제거 후 비교
                    keyword_normalized = re.sub(r'\s+', '', keyword)
                    if keyword_normalized in page_title_normalized:
                        print(f"[PAGES] {fs_type} 페이지 발견: {page_title}")
                        try:
                            # 페이지 HTML 가져오기
                            html = page.html if hasattr(page, 'html') else None
                            print(f"[PAGES] HTML 길이: {len(html) if html else 0}")
                            if html:
                                df = parse_fs_table_from_html(html, fs_type)
                                if df is not None and not df.empty:
                                    fs_data[fs_type] = df
                                    print(f"[PAGES] {fs_type} 추출 성공, 행 수: {len(df)}")
                                else:
                                    print(f"[PAGES] {fs_type} 테이블 파싱 결과 없음")
                        except Exception as e:
                            print(f"[PAGES] {fs_type} 추출 실패: {e}")
                            import traceback
                            print(f"[PAGES] 상세: {traceback.format_exc()}")
                        break

            # 주석 페이지 찾기 (재무제표 주석, 주석사항 등)
            if any(kw in page_title_normalized for kw in ['주석', 'Notes', '주석사항']):
                print(f"[PAGES] 주석 페이지 발견: {page_title}")
                try:
                    html = page.html if hasattr(page, 'html') else None
                    if html:
                        # 비용의 성격별 분류 테이블 찾기
                        from io import StringIO
                        soup = BeautifulSoup(html, 'html.parser')
                        tables = pd.read_html(StringIO(str(soup)))

                        for idx, table in enumerate(tables):
                            # 급여, 복리후생비, 감가상각비 등이 포함된 테이블 찾기
                            table_str = table.to_string()
                            expense_keywords = ['급여', '복리후생', '감가상각', '판매수수료', '광고선전비', '지급수수료']

                            if any(kw in table_str for kw in expense_keywords):
                                # 비용 관련 주석 테이블
                                print(f"[PAGES] 비용 세부 테이블 발견: shape={table.shape}")

                                # 테이블 정제
                                if len(table) > 2:  # 최소 3행 이상
                                    # 첫 행 확인 (단위: 천원 등)
                                    cleaned_table = table.copy()

                                    # 첫 번째 행이 "단위" 정보면 제거
                                    if any('단위' in str(val) for val in cleaned_table.iloc[0].values):
                                        cleaned_table = cleaned_table.iloc[1:].reset_index(drop=True)

                                    # 컬럼명 정제: "구 분" -> "계정과목", "당기" -> "FY{year}", "전기" -> "FY{year-1}"
                                    if len(cleaned_table) > 0:
                                        new_columns = []
                                        # 먼저 컬럼명에서 패턴 확인
                                        cols_have_patterns = any('구' in str(c) or '당' in str(c) or '전' in str(c) for c in cleaned_table.columns)

                                        for i, col in enumerate(cleaned_table.columns):
                                            # 컬럼명에 패턴이 있으면 컬럼명 사용, 없으면 첫 행 값 사용
                                            if cols_have_patterns:
                                                col_str = str(col).strip()
                                            else:
                                                col_str = str(cleaned_table.iloc[0, i]).strip() if len(cleaned_table) > 0 else str(col)
                                            # 공백 제거 후 비교
                                            col_normalized = col_str.replace(' ', '')

                                            if '구분' in col_normalized:
                                                new_columns.append('계정과목')
                                            elif '당기' in col_normalized or '금기' in col_normalized:
                                                # 당기 = 보고서 연도
                                                year = report_year if report_year else 2023
                                                new_columns.append(f'FY{year}')
                                            elif '전기' in col_normalized and '전전기' not in col_normalized:
                                                # 전기 = 보고서 연도 - 1
                                                year = (report_year - 1) if report_year else 2022
                                                new_columns.append(f'FY{year}')
                                            elif '전전기' in col_normalized:
                                                # 전전기 = 보고서 연도 - 2
                                                year = (report_year - 2) if report_year else 2021
                                                new_columns.append(f'FY{year}')
                                            else:
                                                # 계정과목 컬럼이 첫 번째일 가능성 높음
                                                if i == 0:
                                                    new_columns.append('계정과목')
                                                else:
                                                    new_columns.append(col_str)

                                        # 컬럼명에 패턴이 없으면 첫 행 제거 (첫 행이 헤더였음)
                                        if not cols_have_patterns:
                                            cleaned_table = cleaned_table.iloc[1:].reset_index(drop=True)
                                        cleaned_table.columns = new_columns[:len(cleaned_table.columns)]

                                        notes_tables['is_notes'].append({
                                            'name': '비용의 성격별 분류',
                                            'df': cleaned_table,
                                            'consolidated': True
                                        })
                                        print(f"[PAGES] IS 주석 테이블 추가: {cleaned_table.shape}, 컬럼={list(cleaned_table.columns)}")
                except Exception as e:
                    print(f"[PAGES] 주석 페이지 처리 실패: {e}")
                    import traceback
                    print(f"[PAGES] 상세: {traceback.format_exc()}")

    except Exception as e:
        print(f"[PAGES] 페이지 처리 실패: {e}")

    # 주석이 있으면 fs_data에 추가
    if any(notes_tables[key] for key in notes_tables):
        fs_data['notes'] = notes_tables
        print(f"[PAGES] 주석 테이블 수: IS={len(notes_tables['is_notes'])}, BS={len(notes_tables['bs_notes'])}")

    return fs_data


def safe_dataframe_to_json(df):
    """
    DataFrame을 JSON 직렬화 가능한 형태로 안전하게 변환
    NaN, inf, -inf 값을 None으로 변환
    """
    records = df.to_dict('records')
    clean_records = []
    
    for record in records:
        clean_record = {}
        for k, v in record.items():
            # 키 이름도 문자열로 변환
            key_str = str(k) if k is not None else ''
            
            # 값 처리
            if v is None:
                clean_record[key_str] = None
            elif isinstance(v, bool):
                clean_record[key_str] = v
            elif isinstance(v, int):
                clean_record[key_str] = v
            elif isinstance(v, str):
                # 문자열이 숫자인 경우 float로 변환 시도
                v_stripped = v.strip()
                if v_stripped == '' or v_stripped == '-' or v_stripped.lower() == 'nan':
                    clean_record[key_str] = None
                else:
                    try:
                        # 괄호로 표시된 음수 처리: (1,000,000) -> -1000000
                        if v_stripped.startswith('(') and v_stripped.endswith(')'):
                            num_str = v_stripped[1:-1].replace(',', '')
                            clean_record[key_str] = -float(num_str)
                        else:
                            clean_record[key_str] = float(v_stripped.replace(',', ''))
                    except ValueError:
                        clean_record[key_str] = v  # 변환 실패 시 원본 유지
            else:
                # float, numpy 타입 등 처리
                try:
                    float_val = float(v)
                    if math.isnan(float_val) or math.isinf(float_val):
                        clean_record[key_str] = None
                    else:
                        clean_record[key_str] = float_val
                except (TypeError, ValueError):
                    # 변환 불가능한 경우 문자열로
                    clean_record[key_str] = str(v) if v is not None else None
        clean_records.append(clean_record)
    
    return clean_records


def normalize_yearly_data(yearly_data: dict, fs_type: str):
    """
    연도별로 수집된 재무제표 데이터를 정규화된 형태로 변환
    
    Args:
        yearly_data: {연도: DataFrame} 형태의 딕셔너리
        fs_type: 재무제표 유형 (bs, is, cf 등)
    
    Returns:
        계정과목을 행으로, 연도(FY20XX)를 열로 하는 DataFrame
    """
    import re
    
    if not yearly_data:
        return None
    
    # 기간을 오름차순으로 정렬 (과거 -> 현재)
    # 정수(연도)와 문자열("2025 반기", "2025 3Q")을 모두 처리
    def sort_key(period):
        if isinstance(period, int):
            return (period, 0, 0)  # 연간 결산
        else:
            parts = str(period).split()
            year = int(parts[0])
            suffix = parts[1] if len(parts) > 1 else ''
            
            if '반기' in suffix:
                return (year, 1, 6)  # 반기는 6월
            elif 'Q' in suffix:
                q = int(suffix.replace('Q', ''))
                return (year, 1, q * 3)  # 1Q=3월, 2Q=6월, 3Q=9월
            elif '월' in suffix:
                month = int(suffix.replace('월', ''))
                return (year, 1, month)
            else:
                return (year, 0, 0)
    
    sorted_periods = sorted(yearly_data.keys(), key=sort_key)
    print(f"[NORMALIZE] {fs_type}: 기간 목록 = {sorted_periods}")
    
    # 각 기간의 데이터에서 계정과목과 당기금액 추출
    all_accounts = {}  # {계정과목: {기간: 금액}}
    
    for period in sorted_periods:
        df = yearly_data[period]
        if df is None or df.empty:
            continue
        
        # 첫 번째 열을 계정과목으로 가정
        account_col = df.columns[0]
        
        # 당기 금액 열 찾기
        # HTML 테이블에서 당기 열이 두 개의 하위 열(소계/금액)로 나뉘어 있음
        # 일부 항목은 첫 번째 열에, 일부(법인세 등)는 두 번째 열(.1)에 값이 있음
        # 따라서 후보 열들을 모두 저장해두고 각 행에서 값이 있는 열 사용
        period_str = str(period)
        candidate_cols = []
        
        for col in df.columns[1:]:
            col_str = str(col).replace(' ', '')
            if '당' in col_str or period_str in col_str:
                candidate_cols.append(col)
        
        if not candidate_cols and len(df.columns) >= 2:
            candidate_cols = [df.columns[1]]  # 두 번째 열을 기본값으로
        
        if not candidate_cols:
            continue
        
        # 로그용 기본 열 (첫 번째 후보)
        value_col = candidate_cols[0]
        
        if value_col is None:
            continue
        
        print(f"[NORMALIZE] {period}: 계정과목열={account_col}, 금액열={value_col}")
        
        # 디버깅: 해당 기간 DataFrame의 첫 5행 출력
        if fs_type == 'bs':
            print(f"[NORMALIZE] {period} 샘플데이터: {df.head(3).to_dict()}")
        
        # 데이터 추출
        for idx, row in df.iterrows():
            account_raw = str(row[account_col]).strip() if pd.notna(row[account_col]) else ''
            if not account_raw or account_raw in ['nan', 'NaN', '']:
                continue
            
            # 공백, 특수문자 정리
            account_raw = account_raw.replace('\n', ' ').replace('\r', '').strip()
            
            # 계정과목 정규화: 주석번호 제거, 공백 제거
            # 예: "현금및현금성자산(주석3,19)" -> "현금및현금성자산"
            # 예: "자 산 총 계" -> "자산총계"
            account = re.sub(r'\(주석[0-9,]+\)', '', account_raw)  # 주석번호 제거
            account = re.sub(r'\s+', '', account)  # 공백 제거
            account = re.sub(r'\([0-9]+\)', '', account)  # (1), (2) 등 제거
            
            # 후보 열들 중에서 값이 있는 것을 찾아서 사용
            value = None
            for col in candidate_cols:
                val = row[col]
                if pd.notna(val) and str(val).strip() not in ['', '-', 'nan', 'NaN']:
                    value = val
                    break
            
            # 디버깅: 법인세 관련 행 출력
            if '법인세' in account and fs_type == 'is':
                print(f"[DEBUG 법인세] period={period}, account={account}, value={value}, cols_checked={candidate_cols}")
            
            if account not in all_accounts:
                all_accounts[account] = {}
            
            # 이미 값이 있으면 덮어쓰지 않음 (최신 데이터 우선 유지)
            if period not in all_accounts[account] or pd.isna(all_accounts[account][period]):
                all_accounts[account][period] = value
    
    if not all_accounts:
        return None
    
    # DataFrame 생성: 계정과목을 행으로, 기간을 열로
    # 컬럼 이름: 정수는 "FY2024", 문자열은 "FY2025 8월"
    def make_col_name(period):
        if isinstance(period, int):
            return f'FY{period}'
        else:
            return f'FY{period}'  # "FY2025 8월" 형식
    
    result_data = []
    for account, period_values in all_accounts.items():
        row = {'계정과목': account}
        for period in sorted_periods:
            col_name = make_col_name(period)
            row[col_name] = period_values.get(period, None)
        result_data.append(row)
    
    result_df = pd.DataFrame(result_data)
    
    # 열 순서 정렬: 계정과목, FY20XX (오름차순)
    cols = ['계정과목'] + [make_col_name(p) for p in sorted_periods]
    result_df = result_df[cols]
    
    print(f"[NORMALIZE] {fs_type}: 결과 shape = {result_df.shape}")
    
    # 디버깅: 계정과목 목록 출력 (처음 30개)
    if fs_type == 'is':
        accounts = result_df['계정과목'].tolist()[:30]
        print(f"[NORMALIZE] {fs_type} 계정과목 샘플: {accounts}")
    
    return result_df


def parse_fs_table_from_html(html, fs_type):
    """HTML에서 재무제표 테이블을 파싱하여 DataFrame으로 변환"""
    from bs4 import BeautifulSoup
    
    try:
        soup = BeautifulSoup(str(html), 'html.parser')
        tables = soup.find_all('table')
        print(f"[PARSE] {fs_type}: 테이블 수 = {len(tables)}")
        
        # 가장 큰 테이블 찾기 (행 수 기준)
        best_df = None
        best_rows = 0
        
        for idx, table in enumerate(tables):
            try:
                dfs = pd.read_html(str(table))
                if dfs and len(dfs) > 0:
                    df = dfs[0]
                    print(f"[PARSE] 테이블 {idx}: 행={len(df)}, 열={len(df.columns)}")
                    if len(df) > best_rows:
                        best_rows = len(df)
                        best_df = df
            except Exception as e:
                print(f"[PARSE] 테이블 {idx} 파싱 오류: {e}")
                continue
        
        if best_df is not None and best_rows > 3:
            print(f"[PARSE] 최적 테이블 선택: 행={best_rows}")
            return best_df
            
    except Exception as e:
        print(f"[PARSE] HTML 파싱 실패: {e}")
    
    return None


def create_vcm_format(fs_data, excel_filepath=None):
    """VCM 전용 포맷 DataFrame 생성 - 감사보고서(FY컬럼)와 사업보고서(XBRL) 모두 지원

    Args:
        fs_data: 재무제표 데이터 딕셔너리
        excel_filepath: 엑셀 파일 경로 (주석 시트 읽기용)
    """
    import re

    # 엑셀 파일에서 이미 정규화된 데이터 읽기 (재무상태표, 손익계산서 탭에서)
    bs_df = None
    is_df = None

    if excel_filepath and os.path.exists(excel_filepath):
        try:
            bs_df = pd.read_excel(excel_filepath, sheet_name='재무상태표', engine='openpyxl')
            print(f"[VCM] 엑셀에서 재무상태표 로드: {len(bs_df)}행, 컬럼: {list(bs_df.columns)}")

            # CIS(포괄손익계산서) 우선, 없으면 IS(손익계산서)
            try:
                is_df = pd.read_excel(excel_filepath, sheet_name='포괄손익계산서', engine='openpyxl')
                print(f"[VCM] 엑셀에서 포괄손익계산서 로드: {len(is_df)}행")
            except:
                is_df = pd.read_excel(excel_filepath, sheet_name='손익계산서', engine='openpyxl')
                print(f"[VCM] 엑셀에서 손익계산서 로드: {len(is_df)}행")
        except Exception as e:
            print(f"[VCM] 엑셀 파일 읽기 실패, 원본 데이터 사용: {e}")
            bs_df = fs_data.get('bs')
            is_df = fs_data.get('cis')
            if is_df is None or (isinstance(is_df, pd.DataFrame) and is_df.empty):
                is_df = fs_data.get('is')
    else:
        # 엑셀 파일 없으면 원본 사용
        bs_df = fs_data.get('bs')
        is_df = fs_data.get('cis')
        if is_df is None or (isinstance(is_df, pd.DataFrame) and is_df.empty):
            is_df = fs_data.get('is')

    if bs_df is None or is_df is None:
        print(f"[VCM] 필수 데이터 누락: bs={bs_df is not None}, is={is_df is not None}")
        return None

    # Excel 파일에서 주석 시트 읽기 (HTML에서 추출한 비용 항목 포함)
    notes_dfs = []
    if excel_filepath and os.path.exists(excel_filepath):
        try:
            xl = pd.ExcelFile(excel_filepath, engine='openpyxl')
            for sheet in xl.sheet_names:
                # 손익 관련 주석 시트 찾기
                if any(kw in sheet for kw in ['손익주석', '포괄손익주석', 'IS주석', '비용']):
                    note_df = pd.read_excel(excel_filepath, sheet_name=sheet, engine='openpyxl')
                    if not note_df.empty and '계정과목' in note_df.columns:
                        notes_dfs.append(note_df)
                        print(f"[VCM] 주석 시트 로드: {sheet}, {len(note_df)}개 행")
        except Exception as e:
            print(f"[VCM] 주석 시트 로드 실패: {e}")

    # fs_data에서 XBRL 주석 데이터 추가 (비용의 성격별 분류 등)
    if fs_data and 'notes' in fs_data and fs_data['notes'] is not None:
        is_notes = fs_data['notes'].get('is_notes', [])
        for note in is_notes:
            if isinstance(note, dict) and 'df' in note:
                note_df = note['df']
                if note_df is not None and not note_df.empty:
                    # 계정과목 컬럼이 없으면 첫 번째 컬럼을 계정과목으로 설정
                    if '계정과목' not in note_df.columns:
                        first_col = note_df.columns[0]
                        note_df = note_df.rename(columns={first_col: '계정과목'})
                    notes_dfs.append(note_df)
                    print(f"[VCM] XBRL 주석 추가: {note.get('name', 'unknown')}, {len(note_df)}개 행")

    # 주석 테이블 우선순위 정렬: 종합 비용 테이블 먼저 검색하도록
    # 급여, 감가상각비, 지급수수료 등 여러 키워드가 모두 있는 테이블을 우선
    def score_expense_table(df):
        """비용 테이블 우선순위 점수 계산 (높을수록 종합 테이블)"""
        if '계정과목' not in df.columns:
            return 0
        try:
            acc_str = df['계정과목'].astype(str).str.cat(sep=' ')
            score = 0
            # 종합 비용 테이블 특징 키워드
            priority_keywords = ['직원급여', '퇴직급여', '감가상각비', '지급수수료', '광고선전비', '수도광열비']
            for kw in priority_keywords:
                if kw in acc_str:
                    score += 10
            # 테이블 크기도 고려 (종합 테이블은 보통 10행 이상)
            if len(df) >= 10:
                score += 5
            return score
        except:
            return 0

    # 정렬 전 디버그 로깅
    print(f"[VCM] === 정렬 전 디버그 시작: {len(notes_dfs)}개 테이블 ===")
    for i, df in enumerate(notes_dfs):
        score = score_expense_table(df)
        if '계정과목' in df.columns:
            accounts = df['계정과목'].astype(str).tolist()[:5]
            print(f"[VCM] 주석 테이블 [{i}]: {len(df)}행, score={score}, accounts={accounts}...")
        else:
            print(f"[VCM] 주석 테이블 [{i}]: {len(df)}행, score={score}, 계정과목 컬럼 없음")

    notes_dfs.sort(key=score_expense_table, reverse=True)
    if notes_dfs:
        print(f"[VCM] 주석 테이블 정렬 완료: {len(notes_dfs)}개, 최우선 테이블 행수: {len(notes_dfs[0])}")
        # 최우선 테이블 상세 정보
        if '계정과목' in notes_dfs[0].columns:
            accounts = notes_dfs[0]['계정과목'].astype(str).tolist()
            print(f"[VCM] 최우선 테이블 계정과목: {accounts}")

    # 데이터 형식 감지: XBRL vs 감사보고서
    is_xbrl = False
    fy_col_map = {}  # {원본컬럼: 표시용문자열}
    account_col = '계정과목'  # 기본값: 감사보고서

    # 1) FY 컬럼 확인 (감사보고서 형식 또는 추가된 분기 컬럼)
    for c in is_df.columns:
        col_name = c[0] if isinstance(c, tuple) else c
        if isinstance(col_name, str) and col_name.startswith('FY'):
            fy_col_map[c] = col_name

    # 2) XBRL 형식 날짜 컬럼도 확인 (FY 컬럼과 병합)
    # XBRL 형식: ('20240101-20241231', ('연결재무제표',)) 같은 날짜 튜플 컬럼 찾기
    for c in is_df.columns:
        if isinstance(c, tuple) and len(c) >= 2:
            date_part = c[0]
            if isinstance(date_part, str) and '-' in date_part and len(date_part) == 17:
                # '20240101-20241231' 형식 - 연결재무제표만 사용
                if '연결' in str(c[1]):
                    year = date_part[:4]
                    fy_col_map[c] = f'FY{year}'
                    is_xbrl = True

    # 3) XBRL 데이터 감지 (정규화된 경우 또는 튜플 컬럼인 경우)
    # 정규화된 경우: '분류3', '개념ID', '계정과목(영문)' 컬럼
    # 튜플 컬럼인 경우: 'label_ko', 'concept_id', 'class2' 등 포함
    xbrl_indicator_cols = ['분류3', '개념ID', '계정과목(영문)']
    bs_cols_str = [str(c) for c in bs_df.columns]

    # 정규화된 XBRL 체크
    if any(col in bs_cols_str for col in xbrl_indicator_cols):
        is_xbrl = True
        print(f"[VCM] 정규화된 XBRL 데이터 감지 (분류 컬럼 존재)")
    # 튜플 컬럼 XBRL 체크 (label_ko, concept_id 등이 컬럼명에 포함)
    elif any('label_ko' in col_str or 'concept_id' in col_str or 'class2' in col_str for col_str in bs_cols_str):
        is_xbrl = True
        print(f"[VCM] 튜플 컬럼 XBRL 데이터 감지")
    
    # XBRL에서 계정과목 컬럼 찾기 (label_ko가 포함된 튜플 컬럼)
    bs_account_col = '계정과목'  # BS용 기본값
    bs_fy_col_map = {}  # BS용 연도 컬럼 매핑

    if is_xbrl:
        # IS/CIS용 계정과목 컬럼
        for c in is_df.columns:
            if isinstance(c, tuple):
                for part in c:
                    if isinstance(part, str) and 'label_ko' in part:
                        account_col = c
                        break
        # BS용 계정과목 컬럼 (별도로 찾기)
        for c in bs_df.columns:
            if isinstance(c, tuple):
                for part in c:
                    if isinstance(part, str) and 'label_ko' in part:
                        bs_account_col = c
                        print(f"[VCM] BS 계정과목 컬럼: {bs_account_col}")
                        break
        # BS용 연도 컬럼 찾기 (XBRL 형식)
        for c in bs_df.columns:
            if isinstance(c, tuple) and len(c) >= 2:
                date_part = c[0]
                # BS는 특정 시점 (20241231 형식) 또는 기간 (20240101-20241231 형식) 모두 가능
                if isinstance(date_part, str):
                    if '-' in date_part and len(date_part) == 17:
                        # '20240101-20241231' 형식 - 연결재무제표만 사용
                        if '연결' in str(c[1]):
                            year = date_part.split('-')[1][:4]  # 종료 연도 사용
                            bs_fy_col_map[c] = f'FY{year}'
                    elif len(date_part) == 8 and date_part.isdigit():
                        # '20241231' 형식 (BS 특정 시점)
                        if '연결' in str(c[1]):
                            year = date_part[:4]
                            bs_fy_col_map[c] = f'FY{year}'
        print(f"[VCM] BS 연도 컬럼: {list(bs_fy_col_map.values())}")
    else:
        bs_account_col = account_col  # 감사보고서 형식이면 동일
        bs_fy_col_map = fy_col_map  # 감사보고서 형식이면 동일

    if not fy_col_map:
        print(f"[VCM] 연도 컬럼을 찾을 수 없음")
        return None
    
    print(f"[VCM] 데이터 형식: {'XBRL' if is_xbrl else '감사보고서'}, 연도: {list(fy_col_map.values())}, 계정과목컬럼: {account_col}")
    
    # 정렬된 원본 컬럼 목록
    fy_cols = sorted(fy_col_map.keys(), key=lambda c: fy_col_map[c])
    
    # 계정과목 정규화 함수
    def normalize(s):
        if not s: return ''
        s = re.sub(r'\s', '', str(s))
        s = re.sub(r'^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.', '', s)
        s = re.sub(r'\(주석[0-9,]+\)', '', s)
        s = re.sub(r'\(손실\)', '', s)
        return s

    # 주석 참조 제거 함수 (매칭용) - 값은 유지하고 계정명만 정규화
    def strip_note_ref(s):
        """계정과목명에서 주석 참조 제거 (예: '유형자산(주2,4,8)' → '유형자산')"""
        if not s: return ''
        s = re.sub(r'\(주[\d,\s]+\)', '', str(s))
        s = re.sub(r'\(주석[\d,\s]+\)', '', s)
        return s.strip()

    # ========== 섹션 기반 재무상태표 파싱 ==========
    def parse_bs_sections(df, year_col, acc_col):
        """재무상태표를 섹션별로 파싱하여 모든 항목 추출

        Args:
            df: 재무상태표 DataFrame
            year_col: 연도 컬럼명
            acc_col: 계정과목 컬럼명 (BS용)

        Returns:
            dict: {
                '유동자산': [{'name': '현금및현금성자산', 'value': 1000000}, ...],
                '비유동자산': [...],
                '유동부채': [...],
                '비유동부채': [...],
                '자본': [...],
                '총계': {'자산총계': val, '부채총계': val, '자본총계': val}
            }
        """
        sections = {
            '유동자산': [],
            '비유동자산': [],
            '유동부채': [],
            '비유동부채': [],
            '자본': [],
            '총계': {}
        }

        # 섹션 마커 패턴 (로마숫자/숫자 prefix 허용)
        section_patterns = {
            '유동자산': r'^(Ⅰ|I|1)?\.?\s*유동자산$',
            '비유동자산': r'^(Ⅱ|II|2)?\.?\s*비유동자산$',
            '유동부채': r'^(Ⅰ|I|1)?\.?\s*유동부채$',
            '비유동부채': r'^(Ⅱ|II|2)?\.?\s*비유동부채$',
            '자본': r'^(Ⅰ|I|1)?\.?\s*자본$',  # '자본금' 제외 (정확히 '자본'만 매칭)
        }

        # 종료 마커 (다음 섹션 시작 또는 총계)
        end_markers = {
            '유동자산': ['비유동자산', '자산총계'],
            '비유동자산': ['자산총계', '부채', '유동부채'],
            '유동부채': ['비유동부채', '부채총계'],
            '비유동부채': ['부채총계', '자본'],
            '자본': ['자본총계', '부채와자본총계', '부채및자본총계', '자본과부채총계'],
        }

        # 총계 마커
        total_markers = ['자산총계', '부채총계', '자본총계', '부채와자본총계', '부채및자본총계', '자본과부채총계']

        current_section = None

        for idx, row in df.iterrows():
            raw_acc = str(row.get(acc_col, '')).strip()
            if not raw_acc or raw_acc == 'nan':
                continue

            # 정규화된 계정명 (주석 참조 제거)
            acc_normalized = strip_note_ref(raw_acc)
            acc_clean = normalize(acc_normalized)

            # 값 추출
            val = row.get(year_col)
            val_num = None
            if pd.notna(val):
                try:
                    val_num = float(str(val).replace(',', ''))
                except:
                    pass

            # 섹션 시작 감지
            section_found = False
            for section_name, pattern in section_patterns.items():
                if re.match(pattern, acc_clean, re.IGNORECASE):
                    current_section = section_name
                    section_found = True
                    print(f"[섹션파싱] {section_name} 시작: {raw_acc}")
                    break

            if section_found:
                continue

            # 총계 항목 처리
            is_total = False
            for marker in total_markers:
                if marker in acc_clean or acc_clean == normalize(marker):
                    sections['총계'][acc_clean] = val_num
                    is_total = True
                    # 자산총계, 부채총계 도달 시 해당 섹션 종료
                    if '자산총계' in acc_clean:
                        current_section = None
                    elif '부채총계' in acc_clean:
                        current_section = None
                    break

            if is_total:
                continue

            # 현재 섹션에 항목 추가
            if current_section:
                # 종료 마커 체크
                should_end = False
                for end_marker in end_markers.get(current_section, []):
                    if normalize(end_marker) in acc_clean:
                        should_end = True
                        break

                if should_end:
                    current_section = None
                elif val_num is not None and val_num != 0:
                    # 하위 섹션 헤더가 아닌 실제 항목만 추가
                    # (자산, 부채, 투자자산 같은 중간 헤더 제외)
                    if not re.match(r'^(자산|부채|자본|투자자산|당좌자산)$', acc_clean):
                        sections[current_section].append({
                            'name': acc_normalized,  # 원본 계정명 유지 (주석 참조만 제거)
                            'value': val_num
                        })

        return sections

    def parse_bs_sections_xbrl(df, year_col, acc_col):
        """XBRL 형식 재무상태표를 '분류3' 컬럼 기반으로 파싱

        XBRL 데이터는 '분류3' 컬럼에 섹션 정보가 있음:
        - '유동자산', '비유동자산', '유동부채', '비유동부채', '자본'
        """
        sections = {
            '유동자산': [],
            '비유동자산': [],
            '유동부채': [],
            '비유동부채': [],
            '자본': [],
            '총계': {}
        }

        # 분류3 컬럼이 없으면 빈 결과 반환
        if '분류3' not in df.columns:
            print(f"[VCM XBRL] 분류3 컬럼 없음, 빈 섹션 반환")
            return sections

        # 섹션 매핑
        section_mapping = {
            '유동자산': '유동자산',
            '비유동자산': '비유동자산',
            '유동부채': '유동부채',
            '비유동부채': '비유동부채',
            '자본': '자본',
        }

        # 총계 마커
        total_markers = ['자산총계', '부채총계', '자본총계', '부채와자본총계', '부채및자본총계']

        for idx, row in df.iterrows():
            raw_acc = str(row.get(acc_col, '')).strip()
            if not raw_acc or raw_acc == 'nan':
                continue

            section_name = str(row.get('분류3', '')).strip()
            acc_clean = normalize(raw_acc)

            # 값 추출
            val = row.get(year_col)
            val_num = None
            if pd.notna(val):
                try:
                    val_num = float(str(val).replace(',', ''))
                except:
                    pass

            # 총계 항목 처리
            is_total = False
            for marker in total_markers:
                if marker in acc_clean or acc_clean == normalize(marker):
                    sections['총계'][acc_clean] = val_num
                    is_total = True
                    break

            if is_total:
                continue

            # 섹션 헤더 자체는 스킵 (예: 계정='유동자산', 분류3='유동자산')
            if raw_acc == section_name:
                continue

            # 섹션에 항목 추가
            if section_name in section_mapping and val_num is not None and val_num != 0:
                target_section = section_mapping[section_name]
                # 중간 헤더 제외
                if not re.match(r'^(자산|부채|자본|투자자산|당좌자산)$', acc_clean):
                    sections[target_section].append({
                        'name': strip_note_ref(raw_acc),
                        'value': val_num
                    })

        return sections

    # 섹션 데이터에서 값 찾기 (parse_bs_sections 결과 사용)
    def find_in_section(section_items, keywords, excludes=[]):
        """섹션 내 항목에서 키워드로 값 찾기"""
        for item in section_items:
            name = normalize(item['name'])
            excluded = any(normalize(ex) in name for ex in excludes)
            if excluded:
                continue
            for kw in keywords:
                if normalize(kw) in name:
                    return item['value']
        return None

    def sum_in_section(section_items, keywords_list):
        """섹션 내 여러 항목의 합계"""
        total = 0
        for item in section_items:
            name = normalize(item['name'])
            for keywords in keywords_list:
                if isinstance(keywords, str):
                    keywords = [keywords]
                for kw in keywords:
                    if normalize(kw) in name:
                        total += item['value'] or 0
                        break
        return total

    def get_unmatched_items(section_items, matched_keywords_list):
        """매칭되지 않은 항목들 반환 (기타 항목용)"""
        unmatched = []
        for item in section_items:
            name = normalize(item['name'])
            is_matched = False
            for keywords in matched_keywords_list:
                if isinstance(keywords, str):
                    keywords = [keywords]
                for kw in keywords:
                    if normalize(kw) in name:
                        is_matched = True
                        break
                if is_matched:
                    break
            if not is_matched:
                unmatched.append(item)
        return unmatched

    # 값 찾기 함수 (is_df + notes_dfs에서 검색) - 원 단위 그대로 반환
    # 판관비 세부항목은 주석 테이블 먼저 검색 (is_df에 리스 관련 중복값이 있을 수 있음)
    expense_keywords_for_notes_priority = ['급여', '퇴직급여', '감가상각비', '지급수수료', '광고선전비',
                                           '수도광열비', '임차료', '복리후생비', '무형자산상각비', '대손상각비']

    def find_val(df, keywords, year, excludes=[], notes_first=False):
        """
        값을 찾는 함수
        notes_first=True면 주석 테이블을 먼저 검색
        """
        # 판관비 세부항목 키워드가 포함되어 있으면 주석 먼저 검색
        search_notes_first = notes_first or any(
            any(normalize(exp_kw) in normalize(kw) for exp_kw in expense_keywords_for_notes_priority)
            for kw in keywords
        )

        def search_is_df():
            """주 DataFrame(is_df)에서 검색"""
            for _, row in df.iterrows():
                acc = normalize(str(row.get(account_col, '')))
                excluded = any(normalize(ex) in acc for ex in excludes)
                if excluded: continue
                for kw in keywords:
                    if normalize(kw) in acc:
                        val = row.get(year)
                        if pd.notna(val):
                            try:
                                return float(str(val).replace(',', ''))
                            except:
                                pass
            return None

        def search_notes():
            """주석 시트에서 검색 (주석 테이블은 천원 단위이므로 1000배 변환)"""
            for note_idx, note_df in enumerate(notes_dfs):
                for _, row in note_df.iterrows():
                    acc = normalize(str(row.get('계정과목', '')))
                    excluded = any(normalize(ex) in acc for ex in excludes)
                    if excluded: continue
                    for kw in keywords:
                        if normalize(kw) in acc:
                            year_str = fy_col_map.get(year)
                            if year_str is None:
                                if isinstance(year, str) and year.startswith('FY'):
                                    year_str = year
                            if year_str and year_str in note_df.columns:
                                val = row.get(year_str)
                                if pd.notna(val):
                                    try:
                                        result = float(str(val).replace(',', ''))
                                        # 주석 테이블은 천원 단위이므로 원 단위로 변환
                                        result = result * 1000
                                        return result
                                    except:
                                        pass
            return None

        # 검색 순서 결정
        if search_notes_first:
            result = search_notes()
            if result is not None:
                return result
            return search_is_df()
        else:
            result = search_is_df()
            if result is not None:
                return result
            return search_notes()

    # VCM 항목 정의
    vcm_items = []
    
    # ========== 매출/매출원가 하위 항목 동적 추출 ==========
    # 손익계산서에서 매출 및 매출원가 하위 항목을 자동으로 추출
    revenue_sub_items = []  # 매출 하위 항목 [(항목명, 계정과목명), ...]
    cogs_sub_items = []     # 매출원가 하위 항목 [(항목명, 계정과목명), ...]

    in_revenue_section = False
    in_cogs_section = False

    for idx, row in is_df.iterrows():
        acc = normalize(str(row.get(account_col, '')))

        # 매출액 섹션 시작 (헤더만 인식: '매출액'으로 정확히 일치)
        if acc == '매출액':
            in_revenue_section = True
            in_cogs_section = False
            continue

        # 매출원가 섹션 시작 (헤더만 인식)
        if acc == '매출원가':
            in_revenue_section = False
            in_cogs_section = True
            continue

        # 매출총이익 섹션 시작 (매출원가 섹션 종료)
        if '매출총이익' in acc or '총이익' in acc:
            in_revenue_section = False
            in_cogs_section = False
            continue

        # 매출 하위 항목 수집
        if in_revenue_section and acc:
            # 원본 계정과목명 가져오기 (정규화되지 않은)
            original_acc = str(row.get(account_col, '')).strip()
            # Ⅰ, Ⅱ 등 제거
            clean_acc = re.sub(r'^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.', '', original_acc).strip()
            clean_acc = re.sub(r'\(주석[0-9,]+\)', '', clean_acc).strip()

            if clean_acc and clean_acc not in ['매출액', '매출', '수익']:
                revenue_sub_items.append((clean_acc, acc))

        # 매출원가 하위 항목 수집
        if in_cogs_section and acc:
            original_acc = str(row.get(account_col, '')).strip()
            clean_acc = re.sub(r'^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.', '', original_acc).strip()
            clean_acc = re.sub(r'\(주석[0-9,]+\)', '', clean_acc).strip()

            if clean_acc and clean_acc not in ['매출원가', '원가']:
                cogs_sub_items.append((clean_acc, acc))

    print(f"[VCM] 매출 하위 항목 추출: {[item[0] for item in revenue_sub_items]}")
    print(f"[VCM] 매출원가 하위 항목 추출: {[item[0] for item in cogs_sub_items]}")

    # 손익계산서 항목 (동적으로 생성)
    is_items = []

    # 매출 항목
    if revenue_sub_items:
        # 매출 합계: 모든 하위 항목의 합
        revenue_keywords = [(item[1], []) for item in revenue_sub_items]
        is_items.append(('매출', revenue_keywords, 'sum'))

        # 매출 하위 항목들
        for item_name, item_keyword in revenue_sub_items:
            is_items.append((f'  {item_name}', [(item_keyword, ['원가'])], 'find'))
    else:
        # 추출 실패 시 기본 항목 사용
        is_items.append(('매출', [('영업수익', []), ('매출액', [])], 'find'))

    # 매출원가 항목
    if cogs_sub_items:
        # 매출원가 합계: 모든 하위 항목의 합
        cogs_keywords = [(item[1], []) for item in cogs_sub_items]
        is_items.append(('매출원가', cogs_keywords, 'sum'))

        # 매출원가 하위 항목들
        for item_name, item_keyword in cogs_sub_items:
            is_items.append((f'  {item_name}', [(item_keyword, [])], 'find'))
    else:
        # 추출 실패 시 기본 항목 사용
        is_items.append(('매출원가', [('영업비용', []), ('매출원가', [])], 'find'))

    # 나머지 손익계산서 항목 (동료 의견 반영: 판관비는 동적으로 금액순 선택)
    is_items.extend([
        ('매출총이익', [], 'calc_gross'),
        ('판매비와관리비', [], 'calc_sga'),
        # 판관비 하위항목은 아래에서 동적으로 추가됨
        ('영업이익', [], 'calc_op'),
        ('영업외수익', [], 'find_direct'),  # 직접 찾기
        ('  금융수익', [('이자수익', []), ('기타금융수익', [])], 'sum'),
        ('영업외비용', [], 'find_direct'),  # 직접 찾기
        ('  금융비용', [('이자비용', [])], 'find'),
        ('법인세비용차감전이익', [], 'calc_ebt'),
        ('법인세비용', [('법인세비용', ['차감전']), ('법인세등', ['차감전'])], 'find'),
        ('당기순이익', [], 'calc_net'),
    ])

    # ========== 판관비 상위 6개 항목 미리 결정 (모든 연도 합계 기준) ==========
    def get_sga_items_for_year(is_df, year):
        """특정 연도의 모든 판관비 항목 값을 반환"""
        급여 = find_val(is_df, ['급여'], year, ['퇴직', '연차']) or 0
        퇴직급여 = find_val(is_df, ['퇴직급여'], year) or 0
        복리후생비 = find_val(is_df, ['복리후생비'], year) or 0
        기타장기종업원급여 = find_val(is_df, ['기타장기종업원급여'], year) or 0
        주식보상비용 = find_val(is_df, ['주식보상비용', '주식기준보상비용'], year) or 0
        인건비 = 급여 + 퇴직급여 + 복리후생비 + 기타장기종업원급여 + 주식보상비용
        감가상각비 = find_val(is_df, ['감가상각비'], year, ['무형']) or 0

        return {
            '인건비': 인건비,
            '감가상각비': 감가상각비,
            '수수료비용': find_val(is_df, ['지급수수료', '수수료비용', '판매수수료'], year) or 0,
            '광고선전비': find_val(is_df, ['광고선전비'], year) or 0,
            '수도광열비': find_val(is_df, ['수도광열비'], year) or 0,
            '임차료비용': find_val(is_df, ['지급임차료', '임차료비용', '임차료'], year) or 0,
            '무형자산상각비': find_val(is_df, ['무형자산상각비'], year) or 0,
            '대손상각비': find_val(is_df, ['대손상각비'], year) or 0,
            '연구비': (find_val(is_df, ['경상연구개발비'], year) or 0) + (find_val(is_df, ['경상시험연구비'], year) or 0),
            '보험료': find_val(is_df, ['보험료'], year) or 0,
            '여비교통비': find_val(is_df, ['여비교통비'], year) or 0,
            '접대비': find_val(is_df, ['접대비'], year) or 0,
            '세금과공과': find_val(is_df, ['세금과공과'], year) or 0,
            '차량유지비': find_val(is_df, ['차량유지비'], year) or 0,
            '운반비': find_val(is_df, ['운반비'], year) or 0,
            '교육훈련비': find_val(is_df, ['교육훈련비'], year) or 0,
            '통신비': find_val(is_df, ['통신비'], year) or 0,
            '도서인쇄비': find_val(is_df, ['도서인쇄비'], year) or 0,
            '사무용품비': find_val(is_df, ['사무용품비'], year) or 0,
            '소모품비': find_val(is_df, ['소모품비'], year) or 0,
            '보관료': find_val(is_df, ['보관료'], year) or 0,
            '건물관리비': find_val(is_df, ['건물관리비'], year) or 0,
            '협회비': find_val(is_df, ['협회비'], year) or 0,
            '폐기물처리비': find_val(is_df, ['폐기물처리비'], year) or 0,
        }

    # 모든 연도의 합계 계산
    sga_totals = {}
    for year in fy_cols:
        sga_year = get_sga_items_for_year(is_df, year)
        for item_name, val in sga_year.items():
            sga_totals[item_name] = sga_totals.get(item_name, 0) + (val or 0)

    # 합계 기준 상위 5개 항목명 선택 (순서 유지를 위해 리스트 사용)
    # 인건비, 감가상각비는 별도 계산용이므로 제외하고 선택
    sga_totals_nonzero = {k: v for k, v in sga_totals.items() if v and v > 0 and k not in ['인건비', '감가상각비']}
    sga_sorted_total = sorted(sga_totals_nonzero.items(), key=lambda x: abs(x[1]), reverse=True)
    sga_top5_names = [item[0] for item in sga_sorted_total[:5]]  # 상위 5개 항목명 (순서 유지)

    # 각 연도별 값 계산
    rows = []
    for year in fy_cols:
        # 표시용 컬럼명 (튜플이면 문자열로 변환)
        year_str = fy_col_map[year]
        row_data = {'항목': '', year_str: None}
        
        # 기본 값들 먼저 계산
        # 서비스업(영업수익/영업비용) vs 제조업(매출액/매출원가) 모두 지원
        영업수익 = find_val(is_df, ['영업수익', '매출액', '수익'], year, ['원가', '비용']) or 0
        영업비용 = find_val(is_df, ['영업비용', '매출원가'], year) or 0

        # 동적으로 추출된 매출 항목 값 계산
        revenue_values = {}
        for item_name, item_keyword in revenue_sub_items:
            revenue_values[item_name] = find_val(is_df, [item_keyword], year, ['원가']) or 0

        # 동적으로 추출된 매출원가 항목 값 계산
        cogs_values = {}
        for item_name, item_keyword in cogs_sub_items:
            cogs_values[item_name] = find_val(is_df, [item_keyword], year) or 0
        # ========== 판관비 하위항목 (미리 결정된 상위 6개 항목 사용) ==========
        sga_items_all = get_sga_items_for_year(is_df, year)
        인건비 = sga_items_all['인건비']
        감가상각비 = sga_items_all['감가상각비']

        # 미리 결정된 상위 5개 항목만 선택 (합계 기준 순서 유지)
        sga_top5 = [(name, sga_items_all.get(name, 0)) for name in sga_top5_names]
        sga_top5_set = set(sga_top5_names)
        sga_rest = [(k, v) for k, v in sga_items_all.items() if k not in sga_top5_set and k not in ['인건비', '감가상각비'] and v and v > 0]

        # 기타판매비와관리비 계산
        기타판관비 = sum(v for _, v in sga_rest)

        # 표시용 리스트: 인건비 + 상위 5개 (총 6개)
        sga_top6 = [('인건비', 인건비)] + sga_top5

        # 영업외 수익/비용 (직접 찾기)
        영업외수익_direct = find_val(is_df, ['영업외수익', '기타수익'], year) or 0
        영업외비용_direct = find_val(is_df, ['영업외비용', '기타비용'], year) or 0

        # 금융수익/비용
        이자수익 = find_val(is_df, ['이자수익'], year) or 0
        기타금융수익 = find_val(is_df, ['기타금융수익'], year) or 0
        금융수익 = 이자수익 + 기타금융수익
        이자비용 = find_val(is_df, ['이자비용'], year) or 0
        금융비용 = 이자비용
        법인세 = find_val(is_df, ['법인세비용', '법인세등'], year, ['차감전']) or 0

        # 계산 값
        # 서비스업: 영업수익이 있으면 우선 사용
        매출 = 영업수익 if 영업수익 > 0 else sum(revenue_values.values())
        원가 = 영업비용 if 영업비용 > 0 else sum(cogs_values.values())

        # 매출총이익: 손익계산서에서 직접 찾기 시도, 없으면 계산
        매출총이익_direct = find_val(is_df, ['매출총이익', '총이익'], year) or 0
        매출총이익 = 매출총이익_direct if 매출총이익_direct else (매출 - 원가)

        # 판매비와관리비: 손익계산서에서 직접 찾기 시도
        판관비_direct = find_val(is_df, ['판매비와관리비', '판매비및관리비', '판관비'], year) or 0

        # 서비스업 여부 판단: 판관비 계정이 없고 영업비용이 있으면 서비스업
        # 서비스업은 영업비용에 모든 비용이 포함되어 있으므로 판관비를 별도 계산하지 않음
        is_service_business = (판관비_direct == 0) and (영업비용 > 0)

        if is_service_business:
            # 서비스업: 판관비는 0 (영업비용에 이미 포함)
            판관비 = 0
        else:
            # 제조업 등: 판관비가 없으면 주석에서 계산 (인건비 + 상위5개 + 기타)
            판관비 = 판관비_direct if 판관비_direct else (인건비 + sum(v for _, v in sga_top5) + 기타판관비)

        # 영업이익: 손익계산서에서 직접 찾기 시도 (항상 원본값 우선)
        영업이익_direct = find_val(is_df, ['영업이익', '영업손익', '영업손실'], year) or 0
        영업이익 = 영업이익_direct if 영업이익_direct else (매출총이익 - 판관비)

        # 영업외수익/비용: 직접 찾은 값 또는 계산
        영업외수익 = 영업외수익_direct if 영업외수익_direct else 금융수익
        영업외비용 = 영업외비용_direct if 영업외비용_direct else 금융비용

        # 법인세비용차감전이익: 손익계산서에서 직접 찾기 시도
        세전이익_direct = find_val(is_df, ['법인세비용차감전순이익', '법인세비용차감전이익', '법인세차감전순이익', '세전이익'], year) or 0
        세전이익 = 세전이익_direct if 세전이익_direct else (영업이익 + 영업외수익 - 영업외비용)

        # 당기순이익: 손익계산서에서 직접 찾기 시도
        당기순이익_direct = find_val(is_df, ['당기순이익', '당기순손익', '총당기순이익'], year) or 0
        당기순이익 = 당기순이익_direct if 당기순이익_direct else (세전이익 - (법인세 or 0))
        
        values = {
            '매출': 매출,
        }

        # 매출 하위 항목 동적 추가
        for item_name in [item[0] for item in revenue_sub_items]:
            values[f'  {item_name}'] = revenue_values.get(item_name) if revenue_values.get(item_name) else None

        # 매출원가 추가
        values['매출원가'] = 원가

        # 매출원가 하위 항목 동적 추가
        for item_name in [item[0] for item in cogs_sub_items]:
            values[f'  {item_name}'] = cogs_values.get(item_name) if cogs_values.get(item_name) else None

        # 서비스업: 비용 세부항목을 매출원가(영업비용) 하위로 표시
        if is_service_business:
            # 금액순 상위 6개 동적 추가 (인건비, 감가상각비 포함)
            for item_name, item_val in sga_top6:
                values[f'  {item_name}'] = item_val if item_val else None
            # 기타영업비용
            values['  기타영업비용'] = 기타판관비 if 기타판관비 else None

        # EBITDA 계산 (영업이익 + 감가상각비 + 무형자산상각비) - 미리 계산
        무형자산상각비 = sga_items_all.get('무형자산상각비', 0) or 0
        EBITDA = 영업이익 + 감가상각비 + 무형자산상각비

        # % of Sales 계산 (소수점 형태로 저장, 예: 0.127 = 12.7%)
        매출총이익_pct = (매출총이익 / 매출) if 매출 and 매출 != 0 else None
        영업이익_pct = (영업이익 / 매출) if 매출 and 매출 != 0 else None
        당기순이익_pct = (당기순이익 / 매출) if 매출 and 매출 != 0 else None
        EBITDA_pct = (EBITDA / 매출) if 매출 and 매출 != 0 else None

        # 매출총이익 및 % of Sales
        values['매출총이익'] = 매출총이익
        values['% of Sales'] = 매출총이익_pct  # 매출총이익률

        # 판매비와관리비 (서비스업은 생략)
        if is_service_business:
            # 서비스업: 판관비 항목 자체를 생략 (영업비용에 이미 포함)
            pass
        else:
            # 제조업 등: 판관비 및 하위 항목 표시
            values['판매비와관리비'] = 판관비

            # 판관비 하위 항목: 금액순 상위 6개만 표시 (인건비, 감가상각비 포함)
            for item_name, item_val in sga_top6:
                values[f'  {item_name}'] = item_val if item_val else None

            # 기타판매비와관리비
            values['  기타판매비와관리비'] = 기타판관비 if 기타판관비 else None

        # 나머지 손익계산서 항목
        values.update({
            '영업이익': 영업이익,
            '% of Sales (영업이익)': 영업이익_pct,  # 영업이익률
            '영업외수익': 영업외수익,
            '  금융수익': 금융수익 if 금융수익 else None,
            '영업외비용': 영업외비용,
            '  금융비용': 금융비용 if 금융비용 else None,
            '법인세비용차감전이익': 세전이익,
            '법인세비용': 법인세 if 법인세 else None,
            '당기순이익': 당기순이익,
            '% of Sales (순이익)': 당기순이익_pct,  # 당기순이익률
            'EBITDA': EBITDA,
            '% of Sales (EBITDA)': EBITDA_pct,  # EBITDA 마진
        })

        # 손익계산서 세부항목 - 이미 values에서 추가됨, 여기서는 빈 배열
        is_items_detail = []

        # 첫 번째 연도에서 기존 항목 저장 (부모 설정)
        if year == fy_cols[0]:
            current_parent = ''  # 현재 섹션의 부모 항목
            for item_name in values:
                # 들여쓰기 없는 항목은 부모 항목 (섹션 헤더)
                if not item_name.startswith('  '):
                    current_parent = item_name
                    rows.append({'항목': item_name, '부모': ''})
                else:
                    # 들여쓰기 있는 항목은 현재 섹션의 하위 항목
                    rows.append({'항목': item_name, '부모': current_parent})

            # 세부항목 추가
            for item_name, parent, val in is_items_detail:
                # 이미 추가된 항목은 건너뜀
                if item_name not in [r['항목'] for r in rows]:
                    rows.append({'항목': item_name, '부모': parent})
        else:
            # 이후 연도에서 새로운 항목이 있으면 적절한 위치에 삽입
            existing_items = [r['항목'] for r in rows]
            for item_name in values:
                if item_name not in existing_items:
                    # 새로운 항목 발견 - 적절한 위치에 삽입
                    # 세부항목(들여쓰기)은 부모 항목 바로 뒤에 삽입
                    if item_name.startswith('  '):
                        # 부모 항목 찾기 (매출원가 또는 판관비)
                        target_parent = ''
                        insert_idx = len(rows)
                        for i in range(len(rows) - 1, -1, -1):
                            if rows[i]['항목'].startswith('  '):
                                continue
                            elif rows[i]['항목'] == '매출원가' or rows[i]['항목'] == '판매비와관리비':
                                # 매출원가 또는 판관비 세부항목
                                target_parent = rows[i]['항목']
                                insert_idx = i + 1
                                while insert_idx < len(rows) and rows[insert_idx]['항목'].startswith('  '):
                                    insert_idx += 1
                                break
                        # 들여쓰기 있는 항목은 부모 설정
                        rows.insert(insert_idx, {'항목': item_name, '부모': target_parent})
                    else:
                        rows.append({'항목': item_name, '부모': ''})

        # 각 연도별 값 저장
        for r in rows:
            item_name = r['항목']
            if item_name in values:
                val = values[item_name]
                # % of Sales는 소수점 값이므로 round() 적용하지 않음
                if val is not None and val != 0:
                    if '% of Sales' in item_name:
                        r[year_str] = val  # 소수점 그대로 저장
                    else:
                        r[year_str] = round(val)
                else:
                    r[year_str] = None
            else:
                # 세부항목에서 찾기
                for is_item_name, parent, is_val in is_items_detail:
                    if item_name == is_item_name:
                        r[year_str] = round(is_val) if is_val is not None and is_val != 0 else None
                        break

    # ========== 재무상태표 항목 추가 ==========
    # 재무상태표에서 값 찾기 (원 단위 그대로)
    def find_bs_val(keywords, year, excludes=[]):
        # BS용 컬럼 사용 (IS와 다를 수 있음)
        for _, row in bs_df.iterrows():
            acc = normalize(str(row.get(bs_account_col, '')))
            excluded = any(normalize(ex) in acc for ex in excludes)
            if excluded: continue
            for kw in keywords:
                if normalize(kw) in acc:
                    # BS 연도 컬럼 매핑 사용
                    bs_year = year
                    if bs_fy_col_map and fy_col_map:
                        year_str = fy_col_map.get(year, '')
                        for bs_col, bs_year_str in bs_fy_col_map.items():
                            if bs_year_str == year_str:
                                bs_year = bs_col
                                break
                    val = row.get(bs_year)
                    if pd.notna(val):
                        try:
                            return float(str(val).replace(',', ''))
                        except:
                            pass
        return None

    # 재무상태표 행 생성 (부모 컬럼 포함, 세부항목 별도 저장)
    all_bs_items_by_year = {}  # {year_str: bs_items list}

    # ========== 섹션 기반 파싱으로 모든 연도 데이터 수집 ==========
    all_sections = {}  # {year: parsed_sections}

    # BS용 연도 컬럼이 있으면 사용, 없으면 IS 연도 컬럼 사용
    bs_fy_cols = sorted(bs_fy_col_map.keys(), key=lambda c: bs_fy_col_map[c]) if bs_fy_col_map else fy_cols
    bs_col_display = bs_fy_col_map if bs_fy_col_map else fy_col_map

    for bs_year in bs_fy_cols:
        year_str = bs_col_display[bs_year]
        # XBRL 형식이면 분류3 컬럼 기반 파싱, 아니면 행 기반 파싱
        if is_xbrl and '분류3' in bs_df.columns:
            all_sections[year_str] = parse_bs_sections_xbrl(bs_df, bs_year, bs_account_col)
            print(f"[VCM XBRL] {year_str} 섹션 파싱 완료: 유동자산 {len(all_sections[year_str]['유동자산'])}개, 비유동자산 {len(all_sections[year_str]['비유동자산'])}개 항목")
        else:
            all_sections[year_str] = parse_bs_sections(bs_df, bs_year, bs_account_col)
            print(f"[VCM] {year_str} 섹션 파싱 완료: 유동자산 {len(all_sections[year_str]['유동자산'])}개, 비유동자산 {len(all_sections[year_str]['비유동자산'])}개 항목")

    # 항목 정의: (항목명, 부모, 값계산함수)
    # 부모가 있으면 세부항목, 없으면 합계 항목

    for year in fy_cols:
        year_str = fy_col_map[year]
        sections = all_sections.get(year_str, {'유동자산': [], '비유동자산': [], '유동부채': [], '비유동부채': [], '자본': [], '총계': {}})

        # ========== 자산 항목 (섹션 기반) ==========
        # 유동자산 섹션에서 항목 찾기
        유동자산_items = sections['유동자산']
        비유동자산_items = sections['비유동자산']
        유동부채_items = sections['유동부채']
        비유동부채_items = sections['비유동부채']
        자본_items = sections['자본']
        총계 = sections['총계']

        # 유동자산 총계 (섹션 합계 또는 총계에서)
        유동자산 = 총계.get('유동자산') or sum(item['value'] for item in 유동자산_items) or find_bs_val(['유동자산'], year, ['비유동']) or 0

        # 주요 항목 찾기 (섹션 우선, fallback으로 키워드 검색)
        현금 = find_in_section(유동자산_items, ['현금및현금성자산', '현금']) or find_bs_val(['현금및현금성자산'], year) or 0

        # 단기투자자산: 단기금융상품 + 당기손익-공정가치측정금융자산 + 기타포괄손익-공정가치 등
        단기금융상품_직접 = find_in_section(유동자산_items, ['단기금융상품']) or find_bs_val(['단기금융상품'], year) or 0
        당기손익FVPL_유동 = find_in_section(유동자산_items, ['당기손익-공정가치측정금융자산', '당기손익공정가치측정금융자산', 'FVPL금융자산']) or 0
        기타포괄FVOCI_유동 = find_in_section(유동자산_items, ['기타포괄손익-공정가치측정금융자산', '기타포괄손익공정가치측정금융자산']) or 0
        단기금융상품 = 단기금융상품_직접 + 당기손익FVPL_유동 + 기타포괄FVOCI_유동

        재고자산 = find_in_section(유동자산_items, ['재고자산']) or find_bs_val(['재고자산'], year) or 0

        # 매출채권및기타채권 세부항목 (섹션에서 찾기)
        매출채권 = find_in_section(유동자산_items, ['매출채권'], ['장기', '손실', '처분']) or find_bs_val(['매출채권'], year, ['장기']) or 0
        미수금 = find_in_section(유동자산_items, ['미수금'], ['장기']) or find_bs_val(['미수금'], year) or 0
        미수수익 = find_in_section(유동자산_items, ['미수수익']) or find_bs_val(['미수수익'], year) or 0
        선급금 = find_in_section(유동자산_items, ['선급금']) or find_bs_val(['선급금'], year) or 0
        선급비용 = find_in_section(유동자산_items, ['선급비용']) or find_bs_val(['선급비용'], year) or 0
        계약자산_유동 = find_in_section(유동자산_items, ['계약자산']) or 0
        기타금융자산_유동 = find_in_section(유동자산_items, ['기타금융자산']) or 0
        매출채권및기타채권 = 매출채권 + 미수금 + 미수수익 + 선급금 + 선급비용 + 계약자산_유동 + 기타금융자산_유동

        # 기타비금융자산 (매칭되지 않은 유동자산 항목들)
        matched_유동자산_keywords = [
            ['현금및현금성자산', '현금'], ['단기금융상품'], ['당기손익-공정가치측정금융자산', '당기손익공정가치측정금융자산'],
            ['기타포괄손익-공정가치측정금융자산'], ['재고자산'],
            ['매출채권'], ['미수금'], ['미수수익'], ['선급금'], ['선급비용'], ['계약자산'], ['기타금융자산'],
            ['법인세'], ['매각예정']
        ]
        기타유동자산_items = get_unmatched_items(유동자산_items, matched_유동자산_keywords)
        기타비금융자산 = sum(item['value'] for item in 기타유동자산_items) if 기타유동자산_items else (find_bs_val(['기타유동자산', '기타비금융자산'], year) or 0)

        # ========== 비유동자산 항목 (섹션 기반) ==========
        비유동자산 = 총계.get('비유동자산') or sum(item['value'] for item in 비유동자산_items) or find_bs_val(['비유동자산'], year) or 0
        유형자산 = find_in_section(비유동자산_items, ['유형자산'], ['무형', '처분', '사용권']) or find_bs_val(['유형자산'], year, ['무형', '처분']) or 0
        무형자산 = find_in_section(비유동자산_items, ['무형자산'], ['상각', '손상']) or find_bs_val(['무형자산'], year, ['상각']) or 0
        사용권자산 = find_in_section(비유동자산_items, ['사용권자산']) or 0

        # 장기투자자산 세부항목 (섹션 기반)
        장기금융상품 = find_in_section(비유동자산_items, ['장기금융상품', '장기투자자산']) or find_bs_val(['장기금융상품'], year) or 0
        매도가능증권 = find_in_section(비유동자산_items, ['매도가능증권', '매도가능금융자산', '기타포괄손익-공정가치']) or find_bs_val(['매도가능증권', '매도가능금융자산'], year) or 0
        지분법투자 = find_in_section(비유동자산_items, ['지분법적용투자', '관계기업투자', '종속기업투자']) or find_bs_val(['지분법적용투자', '관계기업투자', '종속기업투자'], year) or 0
        장기투자자산 = 장기금융상품 + 매도가능증권 + 지분법투자

        보증금 = find_in_section(비유동자산_items, ['보증금', '임차보증금']) or find_bs_val(['보증금', '임차보증금'], year) or 0

        # 매출채권및기타채권 (비유동) - 장기매출채권 등
        장기매출채권 = find_in_section(비유동자산_items, ['매출채권', '장기매출채권']) or 0
        자산총계 = 총계.get('자산총계') or find_bs_val(['자산총계'], year) or 0

        # ========== 부채 항목 (섹션 기반) ==========
        유동부채 = 총계.get('유동부채') or sum(item['value'] for item in 유동부채_items) or find_bs_val(['유동부채'], year, ['비유동']) or 0

        # 유동 매입채무및기타채무 세부항목 (섹션 기반)
        매입채무 = find_in_section(유동부채_items, ['매입채무'], ['장기']) or find_bs_val(['매입채무'], year, ['장기']) or 0
        미지급금_유동 = find_in_section(유동부채_items, ['미지급금'], ['장기']) or find_bs_val(['미지급금'], year, ['장기']) or 0
        미지급비용_유동 = find_in_section(유동부채_items, ['미지급비용'], ['장기']) or find_bs_val(['미지급비용'], year, ['장기']) or 0
        선수금_유동 = find_in_section(유동부채_items, ['선수금'], ['장기']) or find_bs_val(['선수금'], year, ['장기']) or 0
        선수수익_유동 = find_in_section(유동부채_items, ['선수수익'], ['장기']) or find_bs_val(['선수수익'], year, ['장기']) or 0
        예수금_유동 = find_in_section(유동부채_items, ['예수금'], ['장기']) or find_bs_val(['예수금'], year, ['장기']) or 0
        계약부채_유동 = find_in_section(유동부채_items, ['계약부채']) or 0
        예수보증금_유동 = find_in_section(유동부채_items, ['예수보증금'], ['장기', '임대']) or find_bs_val(['예수보증금'], year, ['장기', '임대']) or 0
        연차충당부채 = find_in_section(유동부채_items, ['연차충당부채', '연차수당충당부채']) or find_bs_val(['연차충당부채', '연차수당충당부채'], year) or 0
        유동매입채무및기타채무 = 매입채무 + 미지급금_유동 + 미지급비용_유동 + 선수금_유동 + 선수수익_유동 + 예수금_유동 + 예수보증금_유동 + 연차충당부채 + 계약부채_유동

        # 차입금 항목 (섹션 기반) - 전환사채, 전환우선주부채 등 포함
        단기차입금 = find_in_section(유동부채_items, ['단기차입금']) or find_bs_val(['단기차입금'], year) or 0
        유동성장기차입금 = find_in_section(유동부채_items, ['유동성장기부채', '유동성장기차입금']) or find_bs_val(['유동성장기부채', '유동성장기차입금'], year) or 0
        유동성사채 = find_in_section(유동부채_items, ['유동성사채']) or find_bs_val(['유동성사채'], year) or 0
        # 유동성사채가 섹션에 없지만 find_bs_val로 찾았으면 유동부채_items에 추가 (동적 분류용)
        if 유동성사채 and not find_in_section(유동부채_items, ['유동성사채']):
            유동부채_items.append({'name': '유동성사채', 'value': 유동성사채})
        전환사채_유동 = find_in_section(유동부채_items, ['전환사채']) or find_bs_val(['전환사채'], year) or 0
        상환전환우선주부채 = find_in_section(유동부채_items, ['상환전환우선주부채', 'RCPS부채']) or 0
        전환우선주부채 = find_in_section(유동부채_items, ['전환우선주부채', 'CPS부채'], ['상환']) or 0  # 상환전환우선주부채 제외

        # 유동차입부채 합계 (사용자 포맷에 맞춤)
        유동차입부채 = 단기차입금 + 유동성장기차입금 + 유동성사채 + 전환사채_유동 + 상환전환우선주부채 + 전환우선주부채

        # 기타금융부채 (리스부채 등) - 차입금에 포함된 항목 제외
        리스부채_유동 = find_in_section(유동부채_items, ['리스부채', '금융리스부채']) or 0
        파생상품부채_유동 = find_in_section(유동부채_items, ['파생상품부채']) or 0
        기타금융부채_유동 = find_in_section(유동부채_items, ['기타금융부채'], ['리스', '파생']) or 0

        # 기타유동부채 세부항목 (섹션 기반)
        미지급법인세 = find_in_section(유동부채_items, ['미지급법인세', '당기법인세부채']) or find_bs_val(['미지급법인세', '당기법인세부채'], year) or 0
        충당부채_유동 = find_in_section(유동부채_items, ['충당부채', '단기충당부채'], ['연차', '퇴직']) or 0

        # 기타유동부채 (매칭되지 않은 유동부채 항목들)
        matched_유동부채_keywords = [
            ['매입채무'], ['미지급금'], ['미지급비용'], ['선수금'], ['선수수익'], ['예수금'], ['예수보증금'],
            ['연차충당부채'], ['계약부채'], ['단기차입금'], ['유동성장기부채', '유동성장기차입금'],
            ['유동성사채'], ['전환사채'], ['상환전환우선주부채'], ['전환우선주부채'],
            ['리스부채', '금융리스부채'], ['파생상품부채'], ['기타금융부채'],
            ['미지급법인세', '당기법인세부채'], ['충당부채']
        ]
        기타유동부채_items = get_unmatched_items(유동부채_items, matched_유동부채_keywords)
        기타유동부채_기타 = sum(item['value'] for item in 기타유동부채_items) if 기타유동부채_items else (find_bs_val(['기타유동부채'], year) or 0)
        # 기타금융부채 합계 (리스부채, 파생상품부채 등 포함)
        기타금융부채_합계 = 리스부채_유동 + 파생상품부채_유동 + 기타금융부채_유동
        기타유동부채 = 미지급법인세 + 기타유동부채_기타 + 충당부채_유동 + 기타금융부채_합계

        # ========== 비유동부채 항목 (섹션 기반) ==========
        비유동부채 = 총계.get('비유동부채') or sum(item['value'] for item in 비유동부채_items) or find_bs_val(['비유동부채'], year) or 0

        # 비유동 매입채무및기타채무 세부항목 (섹션 기반)
        장기매입채무 = find_in_section(비유동부채_items, ['장기매입채무', '매입채무']) or find_bs_val(['장기매입채무'], year) or 0
        장기미지급금 = find_in_section(비유동부채_items, ['장기미지급금', '비유동미지급금']) or find_bs_val(['장기미지급금'], year) or 0
        장기미지급비용 = find_in_section(비유동부채_items, ['장기미지급비용']) or find_bs_val(['장기미지급비용'], year) or 0
        장기선수금 = find_in_section(비유동부채_items, ['장기선수금']) or find_bs_val(['장기선수금'], year) or 0
        장기선수수익 = find_in_section(비유동부채_items, ['장기선수수익']) or find_bs_val(['장기선수수익'], year) or 0
        장기예수금 = find_in_section(비유동부채_items, ['장기예수금']) or find_bs_val(['장기예수금'], year) or 0
        임대보증금 = find_in_section(비유동부채_items, ['임대보증금']) or find_bs_val(['임대보증금'], year) or 0
        예수보증금_비유동 = find_in_section(비유동부채_items, ['예수보증금', '장기예수보증금']) or find_bs_val(['예수보증금', '장기예수보증금'], year) or 0
        계약부채_비유동 = find_in_section(비유동부채_items, ['계약부채']) or 0
        비유동매입채무및기타채무 = 장기매입채무 + 장기미지급금 + 장기미지급비용 + 장기선수금 + 장기선수수익 + 장기예수금 + 임대보증금 + 예수보증금_비유동 + 계약부채_비유동

        사채 = find_in_section(비유동부채_items, ['사채'], ['유동성']) or find_bs_val(['사채'], year, ['유동성']) or 0
        # 사채가 섹션에 없지만 find_bs_val로 찾았으면 비유동부채_items에 추가 (동적 분류용)
        if 사채 and not find_in_section(비유동부채_items, ['사채'], ['유동성']):
            비유동부채_items.append({'name': '사채', 'value': 사채})
        장기차입금 = find_in_section(비유동부채_items, ['장기차입금', '비유동차입부채'], ['유동성']) or find_bs_val(['장기차입금'], year, ['유동성']) or 0
        퇴직급여채무 = find_in_section(비유동부채_items, ['퇴직급여충당부채', '퇴직급여채무', '확정급여채무', '순확정급여부채']) or find_bs_val(['퇴직급여충당부채', '퇴직급여채무', '확정급여채무'], year) or 0
        기타금융부채_비유동 = find_in_section(비유동부채_items, ['기타금융부채', '금융리스부채', '리스부채']) or 0
        충당부채_비유동 = find_in_section(비유동부채_items, ['충당부채', '장기충당부채'], ['퇴직']) or 0
        부채총계 = 총계.get('부채총계') or find_bs_val(['부채총계'], year) or 0

        # ========== 자본 항목 (섹션 기반) ==========
        자본금 = find_in_section(자본_items, ['자본금'], ['잉여금']) or find_bs_val(['자본금'], year, ['잉여금']) or 0
        이익잉여금 = find_in_section(자본_items, ['이익잉여금', '미처분이익잉여금', '결손금']) or find_bs_val(['이익잉여금', '미처분이익잉여금'], year) or 0

        # 기타자본구성요소 세부항목 (섹션 기반)
        자본잉여금 = find_in_section(자본_items, ['자본잉여금']) or find_bs_val(['자본잉여금'], year) or 0
        자본조정 = find_in_section(자본_items, ['자본조정']) or find_bs_val(['자본조정'], year) or 0
        기타포괄손익누계액 = find_in_section(자본_items, ['기타포괄손익누계액']) or find_bs_val(['기타포괄손익누계액'], year) or 0
        기타자본항목 = find_in_section(자본_items, ['기타자본', '기타자본구성요소', '기타자본항목']) or 0
        기타자본 = 자본잉여금 + 자본조정 + 기타포괄손익누계액 + 기타자본항목

        자본총계 = 총계.get('자본총계') or find_bs_val(['자본총계'], year) or 0
        부채와자본총계 = 총계.get('부채와자본총계') or 총계.get('부채및자본총계') or find_bs_val(['부채와자본총계', '부채및자본총계'], year) or 0

        # ========== 계산 항목 ==========
        nwc = 유동자산 - 유동부채
        # 총차입금: 유동차입부채 + 장기차입금 + 사채 (리스부채는 제외)
        총차입금 = 유동차입부채 + 장기차입금 + 사채
        net_debt = 총차입금 - 현금 - 단기금융상품

        # ========== 동적 VCM 구조 생성 ==========
        # 섹션 원본 항목을 표준 카테고리로 그룹화하고 세부항목도 표시

        # 카테고리 매핑 정의 (키워드 -> 카테고리명)
        def categorize_item(item_name, section_type):
            """원본 항목명을 표준 카테고리로 분류"""
            name_norm = normalize(item_name)

            if section_type == '유동자산':
                if any(k in name_norm for k in ['현금및현금성자산', '현금']):
                    return '현금및현금성자산'
                elif any(k in name_norm for k in ['단기금융상품', '당기손익-공정가치', '당기손익공정가치',
                                                   '기타포괄손익-공정가치', '기타포괄손익공정가치', 'FVPL', 'FVOCI']):
                    return '단기투자자산'
                elif any(k in name_norm for k in ['매출채권', '미수금', '미수수익', '선급금', '선급비용',
                                                   '계약자산', '기타금융자산']):
                    return '매출채권및기타채권'
                elif any(k in name_norm for k in ['재고자산', '상품', '제품', '원재료']):
                    return '재고자산'
                elif any(k in name_norm for k in ['법인세자산', '당기법인세자산']):
                    return '당기법인세자산'
                elif any(k in name_norm for k in ['매각예정', '처분자산집단']):
                    return '매각예정자산'
                else:
                    return '기타유동자산'  # 유동자산의 기타 항목

            elif section_type == '비유동자산':
                if any(k in name_norm for k in ['유형자산']) and '무형' not in name_norm:
                    return '유형자산'
                elif any(k in name_norm for k in ['무형자산']):
                    return '무형자산'
                elif any(k in name_norm for k in ['매출채권', '기타금융자산', '보증금', '임차보증금']):
                    return '매출채권및기타채권'
                elif any(k in name_norm for k in ['장기금융상품', '당기손익-공정가치', '기타포괄손익-공정가치',
                                                   '관계기업투자', '종속기업투자', '지분법', '매도가능']):
                    return '장기투자자산'
                else:
                    return '기타비유동자산'  # 비유동자산의 기타 항목

            elif section_type == '유동부채':
                if any(k in name_norm for k in ['매입채무', '미지급금', '미지급비용', '선수금', '선수수익',
                                                 '예수금', '예수보증금', '계약부채']):
                    return '매입채무및기타채무'
                elif any(k in name_norm for k in ['단기차입금', '유동성장기', '유동성사채', '전환사채',
                                                   '상환전환우선주', '전환우선주']):
                    return '유동차입부채'
                elif any(k in name_norm for k in ['충당부채']) and '리스' not in name_norm:
                    return '단기충당부채'
                elif any(k in name_norm for k in ['매각예정부채', '매각예정비유동부채']):
                    return '매각예정부채'
                elif any(k in name_norm for k in ['법인세부채', '당기법인세부채', '미지급법인세']):
                    return '당기법인세부채'
                elif any(k in name_norm for k in ['리스부채', '파생상품부채', '기타금융부채']):
                    return '기타금융부채'
                else:
                    return '기타비금융부채'

            elif section_type == '비유동부채':
                if any(k in name_norm for k in ['매입채무', '미지급금', '미지급비용', '선수금', '선수수익',
                                                 '예수금', '예수보증금', '임대보증금', '계약부채']):
                    return '매입채무및기타채무'
                elif any(k in name_norm for k in ['장기차입금', '사채', '비유동차입부채']):
                    return '비유동차입부채'
                elif any(k in name_norm for k in ['충당부채', '장기충당부채']) and '리스' not in name_norm:
                    return '장기충당부채'
                elif any(k in name_norm for k in ['리스부채', '파생상품부채', '기타금융부채']):
                    return '기타금융부채'
                else:
                    return '기타비금융부채'

            return None

        # 섹션별로 항목을 카테고리로 그룹화
        def group_items_by_category(items, section_type):
            """섹션 항목들을 카테고리로 그룹화하고 합계 계산"""
            groups = {}
            for item in items:
                category = categorize_item(item['name'], section_type)
                if category:
                    if category not in groups:
                        groups[category] = {'total': 0, 'items': []}
                    groups[category]['total'] += item['value']
                    groups[category]['items'].append(item)
            return groups

        # 유동자산 그룹화
        유동자산_groups = group_items_by_category(유동자산_items, '유동자산')
        비유동자산_groups = group_items_by_category(비유동자산_items, '비유동자산')
        유동부채_groups = group_items_by_category(유동부채_items, '유동부채')
        비유동부채_groups = group_items_by_category(비유동부채_items, '비유동부채')

        # ===== 동적 VCM 생성 설정 =====
        MAX_ITEMS = 6  # 각 섹션당 최대 표시 항목 수

        # 필수 항목 정의 (항상 표시해야 하는 카테고리)
        유동자산_필수 = ['현금및현금성자산', '매출채권및기타채권']
        비유동자산_필수 = []  # 금액 기준으로만 정렬
        유동부채_필수 = ['매입채무및기타채무', '유동차입부채']
        비유동부채_필수 = ['비유동차입부채']

        def is_redundant_child(parent_name, child_name):
            """
            세부항목이 부모와 중복되는지 확인
            - 부모가 자식 이름을 포함하거나 그 반대인 경우 (장기충당부채 vs 충당부채)
            - 둘 다 '기타'를 포함하는 경우 (기타유동자산 vs 기타비금융자산)
            """
            parent_norm = normalize(parent_name).replace('[비유동]', '').replace('[netdebt]', '').replace('[nwc]', '')
            child_norm = normalize(child_name)

            # 이름이 서로 포함 관계인 경우 (장기충당부채 ↔ 충당부채)
            if parent_norm in child_norm or child_norm in parent_norm:
                return True

            # 둘 다 '기타'를 포함하는 경우
            if '기타' in parent_norm and '기타' in child_norm:
                return True

            return False

        def select_top_items(groups, required_cats, max_items, section_name):
            """
            필수 항목 + 금액 큰 순으로 상위 N개 선택, 나머지는 '기타'로 합산
            """
            # 모든 카테고리를 금액 기준 정렬
            all_cats = []
            for cat, grp in groups.items():
                if grp['total'] and grp['total'] != 0:
                    all_cats.append({
                        'name': cat,
                        'total': grp['total'],
                        'items': grp['items'],
                        'is_required': cat in required_cats
                    })

            # 금액 절댓값 기준 내림차순 정렬
            all_cats.sort(key=lambda x: abs(x['total']), reverse=True)

            # 필수 항목과 옵션 항목 분리
            required = [c for c in all_cats if c['is_required']]
            optional = [c for c in all_cats if not c['is_required']]

            # 선택할 항목 결정
            selected = []
            기타_total = 0
            기타_items = []

            # 1. 필수 항목 먼저 추가
            for cat in required:
                selected.append(cat)

            # 2. 남은 슬롯에 금액 큰 순으로 추가
            remaining_slots = max_items - len(selected)
            for i, cat in enumerate(optional):
                if i < remaining_slots:
                    selected.append(cat)
                else:
                    # 나머지는 기타로 합산
                    기타_total += cat['total']
                    기타_items.extend(cat['items'])

            # 선택된 항목들을 금액 기준으로 다시 정렬
            selected.sort(key=lambda x: abs(x['total']), reverse=True)

            return selected, 기타_total, 기타_items

        # 동적 bs_items 생성
        bs_items = []

        # ===== 유동자산 =====
        bs_items.append(('유동자산', '', 유동자산))

        selected_유동자산, 기타유동자산_total, 기타유동자산_items = select_top_items(
            유동자산_groups, 유동자산_필수, MAX_ITEMS, '유동자산'
        )

        for cat_info in selected_유동자산:
            cat = cat_info['name']
            bs_items.append((cat, '', cat_info['total']))
            # 세부 항목 필터링 (카테고리명과 같거나 중복되는 항목 제외)
            valid_items = [
                i for i in cat_info['items']
                if normalize(i['name']) != normalize(cat) and not is_redundant_child(cat, i['name'])
            ]
            # 유효 하위 항목이 1개 이상이면 추가 (툴팁용)
            if len(valid_items) >= 1:
                sorted_items = sorted(valid_items, key=lambda x: abs(x['value']), reverse=True)
                for item in sorted_items[:5]:  # 최대 5개까지
                    bs_items.append((item['name'], cat, item['value']))

        # 기타유동자산 (남은 항목 합계)
        if 기타유동자산_total and abs(기타유동자산_total) > 0:
            bs_items.append(('기타유동자산', '', 기타유동자산_total))
            # 기타유동자산에 포함된 세부항목들 (중복 항목 제외)
            valid_기타유동 = [i for i in 기타유동자산_items if not is_redundant_child('기타유동자산', i['name'])]
            if len(valid_기타유동) >= 1:  # 유효 하위 항목이 1개 이상이면 추가 (툴팁용)
                sorted_기타유동 = sorted(valid_기타유동, key=lambda x: abs(x['value']), reverse=True)
                for item in sorted_기타유동[:5]:
                    bs_items.append((item['name'], '기타유동자산', item['value']))

        # 매각예정자산 (유동자산 섹션에 포함)
        매각예정자산 = find_bs_val(['매각예정비유동자산', '매각예정자산', '처분자산집단'], year) or 0
        if 매각예정자산:
            bs_items.append(('매각예정자산', '', 매각예정자산))

        # ===== 비유동자산 =====
        bs_items.append(('비유동자산', '', 비유동자산))

        # 비유동자산 카테고리명 매핑 (유동과 구분 필요한 항목)
        비유동자산_카테고리_표시명 = {
            '매출채권및기타채권': '매출채권및기타채권[비유동]',
        }

        selected_비유동자산, 기타비유동자산_total, 기타비유동자산_items = select_top_items(
            비유동자산_groups, 비유동자산_필수, MAX_ITEMS, '비유동자산'
        )

        for cat_info in selected_비유동자산:
            cat = cat_info['name']
            cat_display = 비유동자산_카테고리_표시명.get(cat, cat)
            bs_items.append((cat_display, '', cat_info['total']))
            # 세부 항목 필터링 (카테고리명과 같거나 중복되는 항목 제외)
            valid_items = [
                i for i in cat_info['items']
                if normalize(i['name']) != normalize(cat) and not is_redundant_child(cat_display, i['name'])
            ]
            # 유효 하위 항목이 1개 이상이면 추가 (툴팁용)
            if len(valid_items) >= 1:
                sorted_items = sorted(valid_items, key=lambda x: abs(x['value']), reverse=True)
                for item in sorted_items[:5]:  # 최대 5개까지
                    bs_items.append((item['name'], cat_display, item['value']))

        if 기타비유동자산_total and abs(기타비유동자산_total) > 0:
            bs_items.append(('기타비유동자산', '', 기타비유동자산_total))
            # 기타비유동자산에 포함된 세부항목들 (중복 항목 제외)
            valid_기타비유동 = [i for i in 기타비유동자산_items if not is_redundant_child('기타비유동자산', i['name'])]
            if len(valid_기타비유동) >= 1:  # 유효 하위 항목이 1개 이상이면 추가 (툴팁용)
                sorted_기타비유동 = sorted(valid_기타비유동, key=lambda x: abs(x['value']), reverse=True)
                for item in sorted_기타비유동[:5]:
                    bs_items.append((item['name'], '기타비유동자산', item['value']))

        bs_items.append(('자산총계', '', 자산총계))

        # ===== 유동부채 =====
        bs_items.append(('유동부채', '', 유동부채))

        selected_유동부채, 기타유동부채_total, 기타유동부채_items = select_top_items(
            유동부채_groups, 유동부채_필수, MAX_ITEMS, '유동부채'
        )

        for cat_info in selected_유동부채:
            cat = cat_info['name']
            bs_items.append((cat, '', cat_info['total']))
            # 세부 항목 필터링 (카테고리명과 같거나 중복되는 항목 제외)
            valid_items = [
                i for i in cat_info['items']
                if normalize(i['name']) != normalize(cat) and not is_redundant_child(cat, i['name'])
            ]
            # 유효 하위 항목이 1개 이상이면 추가 (툴팁용)
            if len(valid_items) >= 1:
                sorted_items = sorted(valid_items, key=lambda x: abs(x['value']), reverse=True)
                for item in sorted_items[:5]:  # 최대 5개까지
                    bs_items.append((item['name'], cat, item['value']))

        if 기타유동부채_total and abs(기타유동부채_total) > 0:
            bs_items.append(('기타유동부채', '', 기타유동부채_total))
            # 기타유동부채에 포함된 세부항목들 (중복 항목 제외)
            valid_기타유동부채 = [i for i in 기타유동부채_items if not is_redundant_child('기타유동부채', i['name'])]
            if len(valid_기타유동부채) >= 1:  # 유효 하위 항목이 1개 이상이면 추가 (툴팁용)
                sorted_기타유동부채 = sorted(valid_기타유동부채, key=lambda x: abs(x['value']), reverse=True)
                for item in sorted_기타유동부채[:5]:
                    bs_items.append((item['name'], '기타유동부채', item['value']))

        # 매각예정부채 별도 추출 (섹션 외부에 있을 수 있음)
        매각예정부채 = find_bs_val(['매각예정비유동부채', '매각예정부채'], year) or 0
        if 매각예정부채:
            bs_items.append(('매각예정부채', '', 매각예정부채))

        # ===== 비유동부채 =====
        bs_items.append(('비유동부채', '', 비유동부채))

        # 비유동부채 카테고리명 매핑 (유동과 구분 필요한 항목)
        비유동부채_카테고리_표시명 = {
            '매입채무및기타채무': '매입채무및기타채무[비유동]',
            '기타금융부채': '기타금융부채[비유동]',
            '기타비금융부채': '기타비금융부채[비유동]',
        }

        selected_비유동부채, 기타비유동부채_total, 기타비유동부채_items = select_top_items(
            비유동부채_groups, 비유동부채_필수, MAX_ITEMS, '비유동부채'
        )

        for cat_info in selected_비유동부채:
            cat = cat_info['name']
            cat_display = 비유동부채_카테고리_표시명.get(cat, cat)
            bs_items.append((cat_display, '', cat_info['total']))
            # 세부 항목 필터링 (카테고리명과 같거나 중복되는 항목 제외)
            valid_items = [
                i for i in cat_info['items']
                if normalize(i['name']) != normalize(cat) and not is_redundant_child(cat_display, i['name'])
            ]
            # 유효 하위 항목이 1개 이상이면 추가 (툴팁용)
            if len(valid_items) >= 1:
                sorted_items = sorted(valid_items, key=lambda x: abs(x['value']), reverse=True)
                for item in sorted_items[:5]:  # 최대 5개까지
                    bs_items.append((item['name'], cat_display, item['value']))

        if 기타비유동부채_total and abs(기타비유동부채_total) > 0:
            bs_items.append(('기타비유동부채', '', 기타비유동부채_total))
            # 기타비유동부채에 포함된 세부항목들 (중복 항목 제외)
            valid_기타비유동부채 = [i for i in 기타비유동부채_items if not is_redundant_child('기타비유동부채', i['name'])]
            if len(valid_기타비유동부채) >= 1:  # 유효 하위 항목이 1개 이상이면 추가 (툴팁용)
                sorted_기타비유동부채 = sorted(valid_기타비유동부채, key=lambda x: abs(x['value']), reverse=True)
                for item in sorted_기타비유동부채[:5]:
                    bs_items.append((item['name'], '기타비유동부채', item['value']))

        bs_items.append(('부채총계', '', 부채총계))

        # ===== 자본 =====
        bs_items.append(('자본금', '', 자본금))
        bs_items.append(('이익잉여금', '', 이익잉여금))
        bs_items.append(('기타자본구성요소', '', 기타자본))
        # 자본 세부항목
        for item in 자본_items:
            name_norm = normalize(item['name'])
            if '자본금' not in name_norm and '이익잉여금' not in name_norm and '결손금' not in name_norm:
                if any(k in name_norm for k in ['자본잉여금', '주식발행초과금', '자본조정', '기타포괄손익', '기타자본']):
                    bs_items.append((item['name'], '기타자본구성요소', item['value']))

        bs_items.append(('자본총계', '', 자본총계))
        bs_items.append(('부채와자본총계', '', 부채와자본총계))

        # ===== 계산 항목 =====
        bs_items.append(('NWC', '', nwc))
        bs_items.append(('유동자산 [NWC]', 'NWC', 유동자산))
        bs_items.append(('유동부채 [NWC]', 'NWC', 유동부채))

        # Net Debt 계산: 유동차입부채 + 비유동차입부채 - 현금 - 단기투자자산
        유동차입부채_합계 = 유동부채_groups.get('유동차입부채', {}).get('total', 0)
        비유동차입부채_합계 = 비유동부채_groups.get('비유동차입부채', {}).get('total', 0)
        현금_합계 = 유동자산_groups.get('현금및현금성자산', {}).get('total', 0)
        단기투자자산_합계 = 유동자산_groups.get('단기투자자산', {}).get('total', 0)
        net_debt = 유동차입부채_합계 + 비유동차입부채_합계 - 현금_합계 - 단기투자자산_합계

        bs_items.append(('Net Debt', '', net_debt))
        bs_items.append(('유동차입부채 [NetDebt]', 'Net Debt', 유동차입부채_합계))
        bs_items.append(('비유동차입부채 [NetDebt]', 'Net Debt', 비유동차입부채_합계))
        bs_items.append(('현금및현금성자산 [NetDebt]', 'Net Debt', -현금_합계))
        bs_items.append(('단기투자자산 [NetDebt]', 'Net Debt', -단기투자자산_합계))

        # 항목 순서와 값을 저장
        all_bs_items_by_year[year_str] = bs_items

    # ========== 마스터 순서 결정 및 값 병합 ==========
    # 모든 연도의 항목을 올바른 순서로 병합
    master_order = []  # [(항목명, 부모)]
    master_order_set = set()

    # 각 연도의 bs_items를 순회하며 마스터 순서 구축
    # 최신 연도부터 처리하여 최신 데이터의 순서를 우선
    for year_str in sorted(all_bs_items_by_year.keys(), reverse=True):
        bs_items = all_bs_items_by_year[year_str]

        # 첫 번째(최신) 연도는 순서 그대로 사용
        if not master_order:
            for item_name, parent, val in bs_items:
                master_order.append((item_name, parent))
                master_order_set.add(item_name)
        else:
            # 이후 연도: 새 항목만 적절한 위치에 삽입
            insert_idx = 0
            for item_name, parent, val in bs_items:
                if item_name not in master_order_set:
                    # 새 항목: 올바른 위치에 삽입
                    master_order.insert(insert_idx, (item_name, parent))
                    master_order_set.add(item_name)
                else:
                    # 기존 항목: 현재 위치 찾기
                    for i, (name, _) in enumerate(master_order):
                        if name == item_name:
                            insert_idx = i
                            break
                insert_idx += 1

    # 마스터 순서대로 bs_rows 생성
    bs_rows = []
    item_to_row = {}
    for item_name, parent in master_order:
        row = {'항목': item_name, '부모': parent}
        bs_rows.append(row)
        item_to_row[item_name] = row

    # 각 연도의 값 채우기
    for year_str, bs_items in all_bs_items_by_year.items():
        for item_name, parent, val in bs_items:
            if item_name in item_to_row:
                item_to_row[item_name][year_str] = round(val) if val is not None and val != 0 else None

    # 빈 행 필터링 (모든 연도에서 값이 없는 행 제거)
    year_cols = sorted(all_bs_items_by_year.keys())
    filtered_bs_rows = []
    for row in bs_rows:
        has_value = any(row.get(y) is not None and row.get(y) != 0 for y in year_cols)
        if has_value:
            filtered_bs_rows.append(row)
        else:
            print(f"[VCM] 빈 행 제거: {row.get('항목')}")

    # 재무상태표 + 손익계산서 순서로 결합
    all_rows = filtered_bs_rows + rows

    # ========== 타입 컬럼 추가 ==========
    # 타입 결정 함수
    def get_item_type(item_name):
        name = item_name.strip()

        # highlight 타입 (강조 항목)
        highlight_items = ['영업이익', '당기순이익', 'EBITDA']
        if name in highlight_items:
            return 'highlight'

        # total 타입 (합계 항목)
        total_items = ['매출총이익', '자산총계', '부채총계', '자본총계', '부채와자본총계',
                       '법인세비용차감전이익', 'NWC', 'Net Debt']
        if name in total_items:
            return 'total'

        # percent 타입
        if name == '% of Sales' or name.startswith('%'):
            return 'percent'

        # category 타입 (대분류)
        category_items = ['유동자산', '비유동자산', '유동부채', '비유동부채', '자본',
                          '매출', '매출원가', '판매비와관리비', '영업외수익', '영업외비용',
                          '유동자산 [NWC]', '유동부채 [NWC]',
                          '유동차입부채 [NetDebt]', '비유동차입부채 [NetDebt]',
                          '현금및현금성자산 [NetDebt]', '단기투자자산 [NetDebt]']
        if name in category_items:
            return 'category'

        # subitem 타입 (들여쓰기 있는 항목)
        if item_name.startswith('  '):
            return 'subitem'

        # 기본값
        return 'item'

    # 각 행에 타입 추가
    for row in all_rows:
        row['타입'] = get_item_type(row.get('항목', ''))

    # ========== 복사용테이블 생성 (단위: 천만원, 포맷팅 완료) ==========
    # 규칙:
    # - 부모 없는 항목 → 표시
    # - IS 섹션(매출, 매출원가, 판관비, 영업외수익/비용)의 하위항목 → 표시
    # - 그 외 (BS 세부항목, 기타XXX 하위, NWC/NetDebt 하위) → 제외 (툴팁용)
    display_rows = []
    unit_divisor = 10000000  # 원 → 천만원 (1천만 = 10,000,000)

    # IS 섹션: 하위항목 표시 허용
    is_sections_with_subitems = ['매출', '매출원가', '판매비와관리비', '영업외수익', '영업외비용']

    for row in all_rows:
        item_name = row.get('항목', '')
        item_type = row.get('타입', 'item')
        item_parent = row.get('부모', '')

        # 부모 없는 항목 → 표시
        if not item_parent or not str(item_parent).strip():
            pass  # 표시
        # IS 섹션의 하위항목 → 표시
        elif str(item_parent).strip() in is_sections_with_subitems:
            pass  # 표시
        # 그 외 (BS 세부항목, 기타XXX 하위, NWC/NetDebt 하위) → 제외
        else:
            continue

        # 빈 행 필터링: 모든 연도에 값이 없으면 스킵
        has_any_value = any(row.get(col) is not None and row.get(col) != 0 for col in year_cols)
        if not has_any_value:
            continue

        display_row = {'항목': item_name}

        for col in year_cols:
            val = row.get(col)
            if val is not None and val != 0:
                if item_type == 'percent':
                    # 퍼센트는 그대로 (0.127 → "12.7%")
                    display_row[col] = f"{val * 100:.1f}%" if isinstance(val, (int, float)) and val < 1 else val
                else:
                    # 숫자는 천만원 단위로 변환하고 포맷팅
                    converted = val / unit_divisor
                    # 소수점 1자리까지 표시, 천 단위 콤마 (0.0이면 빈값 처리)
                    if abs(converted) < 0.05:
                        display_row[col] = ''  # 너무 작으면 빈값
                    elif converted == int(converted):
                        display_row[col] = f"{int(converted):,}"  # 정수면 소수점 없이
                    else:
                        display_row[col] = f"{converted:,.1f}"  # 소수점 1자리
            else:
                display_row[col] = ''

        display_rows.append(display_row)

    # 컬럼 순서 정리: 항목, 타입, 부모, FY연도들...
    vcm_df = pd.DataFrame(all_rows)
    cols_order = ['항목', '타입', '부모'] + [c for c in vcm_df.columns if c not in ['항목', '타입', '부모']]
    vcm_df = vcm_df[[c for c in cols_order if c in vcm_df.columns]]

    # 복사용테이블 DataFrame
    display_df = pd.DataFrame(display_rows)

    return vcm_df, display_df


def normalize_xbrl_columns(df):
    """XBRL 형식의 복잡한 튜플 컬럼을 간단한 문자열로 변환"""
    if df is None or df.empty:
        return df
    
    # 튜플 컬럼이 있는지 확인
    has_tuple_cols = any(isinstance(c, tuple) for c in df.columns)
    if not has_tuple_cols:
        return df  # 이미 정규화된 경우
    
    print(f"[XBRL] 컬럼 정규화 시작: {len(df.columns)}개 컬럼")
    print(f"[XBRL] 원본 컬럼 타입들: {[(col, type(col).__name__) for col in df.columns if 'FY' in str(col) or '2025' in str(col)]}")
    
    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            # 튜플 컬럼 처리
            first_part = col[0] if len(col) > 0 else ''
            second_part = col[1] if len(col) > 1 else ''
            
            # 날짜 형식 컬럼 ('20240101-20241231', ...) 또는 ('20241231', ...)
            if isinstance(first_part, str):
                # FY로 시작하는 컬럼 (병합된 분기 데이터)
                if first_part.startswith('FY'):
                    new_columns.append(first_part)
                # '20240101-20241231' 형식 (17자)
                elif '-' in first_part and len(first_part) == 17:
                    year = first_part[:4]
                    new_columns.append(f'FY{year}')
                # '20241231' 형식 (8자, 숫자만)
                elif len(first_part) == 8 and first_part.isdigit():
                    year = first_part[:4]
                    new_columns.append(f'FY{year}')
                # label_ko, concept_id 등의 메타데이터 컬럼
                elif isinstance(second_part, str):
                    if 'label_ko' in second_part:
                        new_columns.append('계정과목')
                    elif 'concept_id' in second_part:
                        new_columns.append('개념ID')
                    elif 'label_en' in second_part:
                        new_columns.append('계정과목(영문)')
                    elif 'class0' in second_part:
                        new_columns.append('분류1')
                    elif 'class1' in second_part:
                        new_columns.append('분류2')
                    elif 'class2' in second_part:
                        new_columns.append('분류3')
                    elif 'class3' in second_part:
                        new_columns.append('분류4')
                    else:
                        new_columns.append(str(second_part))
                else:
                    new_columns.append(str(first_part)[:50])
            else:
                new_columns.append(str(first_part)[:50])
        else:
            new_columns.append(col)
    
    df_copy = df.copy()
    df_copy.columns = new_columns
    
    # FY 컬럼을 과거→현재 순서로 정렬 (FY2017, FY2018, ..., FY2024, FY2025 3Q)
    non_fy_cols = [c for c in df_copy.columns if not str(c).startswith('FY')]
    fy_cols = [c for c in df_copy.columns if str(c).startswith('FY')]
    
    # FY 컬럼 정렬 (연도 기준 오름차순, 분기는 뒤에)
    def fy_sort_key(col):
        col_str = str(col)
        # FY2025 3Q -> (2025, 3), FY2024 -> (2024, 0)
        if ' ' in col_str:
            parts = col_str.replace('FY', '').split(' ')
            year = int(parts[0]) if parts[0].isdigit() else 9999
            quarter = int(parts[1].replace('Q', '')) if len(parts) > 1 else 0
            return (year, quarter)
        else:
            year = int(col_str.replace('FY', '')) if col_str.replace('FY', '').isdigit() else 9999
            return (year, 0)
    
    fy_cols_sorted = sorted(fy_cols, key=fy_sort_key)
    
    print(f"[XBRL] 정규화 후 FY 컬럼: {fy_cols_sorted}")
    
    # 메타데이터 컬럼 + 정렬된 FY 컬럼 순서로 재배열
    new_order = non_fy_cols + fy_cols_sorted
    df_copy = df_copy[new_order]
    
    return df_copy


def cleanup_old_excel_files(output_dir='output', days=7):
    """일주일 이상 지난 엑셀 파일 삭제"""
    import glob
    import time

    try:
        # output 디렉토리의 모든 .xlsx 파일 검색
        pattern = os.path.join(output_dir, '*.xlsx')
        excel_files = glob.glob(pattern)

        # 현재 시간
        current_time = time.time()
        # 7일을 초 단위로 변환
        cutoff_time = current_time - (days * 24 * 60 * 60)

        deleted_count = 0
        for file_path in excel_files:
            try:
                # 파일 생성 시간 확인
                file_mtime = os.path.getmtime(file_path)

                # 7일 이상 지난 파일 삭제
                if file_mtime < cutoff_time:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"[Cleanup] 오래된 파일 삭제: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"[Cleanup] 파일 삭제 실패: {file_path}, 오류: {e}")

        if deleted_count > 0:
            print(f"[Cleanup] 총 {deleted_count}개 파일 삭제 완료")
        else:
            print(f"[Cleanup] 삭제할 오래된 파일 없음 (총 {len(excel_files)}개 파일 확인)")

    except Exception as e:
        print(f"[Cleanup] 파일 정리 실패: {e}")


def save_to_excel(fs_data, filepath: str):
    """재무제표를 엑셀 파일로 저장 (주석 테이블 포함)"""
    # 엑셀 저장 전 오래된 파일 삭제
    cleanup_old_excel_files()

    sheet_names = {
        'bs': '재무상태표',
        'is': '손익계산서',
        'cis': '포괄손익계산서',
        'cf': '현금흐름표'
    }
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for key, name in sheet_names.items():
            try:
                df = fs_data[key]
                if df is not None and not df.empty:
                    # XBRL 형식인 경우 컬럼 정규화
                    df_normalized = normalize_xbrl_columns(df)
                    df_normalized.to_excel(writer, sheet_name=name, index=False)
            except Exception:
                pass
        
        # 주석 테이블들 저장 (각 재무제표별)
        notes = fs_data.get('notes', {})
        if notes:
            # 손익계산서 관련 주석들
            is_notes = notes.get('is_notes', [])
            for idx, note in enumerate(is_notes[:5]):  # 최대 5개
                try:
                    note_df = note['df']
                    if note_df is not None and not note_df.empty:
                        note_df_normalized = normalize_xbrl_columns(note_df)
                        sheet_name = f'손익주석{idx+1}'
                        note_df_normalized.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"[Excel] 주석 저장: {sheet_name}")
                except Exception as e:
                    print(f"[Excel] 손익주석 저장 실패: {e}")
            
            # 재무상태표 관련 주석들
            bs_notes = notes.get('bs_notes', [])
            for idx, note in enumerate(bs_notes[:5]):  # 최대 5개
                try:
                    note_df = note['df']
                    if note_df is not None and not note_df.empty:
                        note_df_normalized = normalize_xbrl_columns(note_df)
                        sheet_name = f'재무주석{idx+1}'
                        note_df_normalized.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"[Excel] 주석 저장: {sheet_name}")
                except Exception as e:
                    print(f"[Excel] 재무주석 저장 실패: {e}")
            
            # 현금흐름표 관련 주석들
            cf_notes = notes.get('cf_notes', [])
            for idx, note in enumerate(cf_notes[:3]):  # 최대 3개
                try:
                    note_df = note['df']
                    if note_df is not None and not note_df.empty:
                        note_df_normalized = normalize_xbrl_columns(note_df)
                        sheet_name = f'현금주석{idx+1}'
                        note_df_normalized.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"[Excel] 주석 저장: {sheet_name}")
                except Exception as e:
                    print(f"[Excel] 현금주석 저장 실패: {e}")
        
        # VCM 전용 포맷 시트 추가
        try:
            print(f"[VCM] create_vcm_format 호출 시작...")
            vcm_result = create_vcm_format(fs_data, filepath)

            # 튜플 반환 (vcm_df, display_df) 처리
            if isinstance(vcm_result, tuple):
                vcm_df, display_df = vcm_result
            else:
                vcm_df = vcm_result
                display_df = None

            print(f"[VCM] create_vcm_format 완료: {len(vcm_df) if vcm_df is not None else 0}개 항목")

            if vcm_df is not None and not vcm_df.empty:
                print(f"[VCM] VCM 데이터 저장 중... ({len(vcm_df)}행)")
                vcm_df.to_excel(writer, sheet_name='VCM전용포맷', index=False)
                print(f"[VCM] VCM전용포맷 시트 저장 완료")

            if display_df is not None and not display_df.empty:
                print(f"[VCM] 복사용테이블 저장 중... ({len(display_df)}행)")
                display_df.to_excel(writer, sheet_name='복사용테이블', index=False)
                print(f"[VCM] 복사용테이블 시트 저장 완료")

        except Exception as e:
            import traceback
            print(f"[VCM] VCM 포맷 생성 실패: {e}")
            print(f"[VCM] 상세 에러:\n{traceback.format_exc()}")


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """작업 상태 조회 API"""
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    
    result = {
        "status": task['status'],
        "progress": task['progress'],
        "message": task['message'],
        "filename": task.get('filename')
    }
    
    # 완료 시 미리보기 데이터 포함
    if task['status'] == 'completed' and task.get('preview_data'):
        result['preview_data'] = task['preview_data']
    
    return result


@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str):
    """작업 취소 API"""
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    
    # 완료된 작업은 파일 삭제하지 않고 task 정보만 제거
    if task.get('status') == 'completed':
        # 완료된 작업: 파일은 유지, task 정보만 제거
        del TASKS[task_id]
        return {"success": True, "message": "작업 정보가 정리되었습니다."}
    
    # 진행 중인 작업: 취소 처리
    task['cancelled'] = True
    task['status'] = 'cancelled'
    task['message'] = '취소됨'
    
    # 정리 (진행 중인 작업만 파일 삭제)
    cleanup_task(task_id)
    
    return {"success": True, "message": "작업이 취소되었습니다."}


@app.get("/api/download/{task_id}")
async def download_file(task_id: str):
    """파일 다운로드 API"""
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    
    if task['status'] != 'completed':
        raise HTTPException(status_code=400, detail="작업이 완료되지 않았습니다.")
    
    filepath = task.get('file_path')
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    filename = task.get('filename', 'financial_statement.xlsx')
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """작업 및 파일 삭제 API"""
    task = TASKS.get(task_id)
    if task:
        # 파일 삭제
        if task.get('file_path') and os.path.exists(task['file_path']):
            try:
                os.remove(task['file_path'])
            except Exception:
                pass
        # 작업 삭제
        del TASKS[task_id]
    
    return {"success": True}


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("DART 재무제표 추출기 서버")
    print("=" * 50)
    print("서버 시작: http://localhost:8000")
    print("종료: Ctrl+C")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)
