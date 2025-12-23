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
        
        # is가 없고 cis가 있으면 cis를 is로 사용 (포괄손익계산서 = 손익계산서)
        is_df = fs_data.get('is')
        cis_df = fs_data.get('cis')
        is_empty = is_df is None or (hasattr(is_df, 'empty') and is_df.empty)
        cis_valid = cis_df is not None and hasattr(cis_df, 'empty') and not cis_df.empty
        print(f"[EXTRACT] is_empty={is_empty}, cis_valid={cis_valid}")
        if is_empty and cis_valid:
            fs_data['is'] = fs_data['cis']
            print(f"[EXTRACT] is 데이터 없음, cis를 is로 사용")
        
        for key in ['bs', 'is', 'cis', 'cf']:
            df = fs_data.get(key)
            print(f"[EXTRACT] fs_data[{key}]: 존재={df is not None}, empty={df.empty if df is not None else 'N/A'}")
            if df is not None and not df.empty:
                # XBRL 형식인 경우 컬럼 정규화 후 JSON 변환
                df_normalized = normalize_xbrl_columns(df)
                task['preview_data'][key] = safe_dataframe_to_json(df_normalized)
                print(f"[EXTRACT] preview_data[{key}] 생성: {len(task['preview_data'][key])}개 행")
        
        # 주석 테이블들도 preview_data에 추가 (프론트엔드에서 세부항목 표시용)
        # 항상 Excel에서 주석 데이터 읽기 (fs_data['notes']가 없을 수 있으므로)
        print(f"[EXTRACT] Excel에서 주석 데이터 읽기...")
        import sys
        sys.stdout.flush()
        notes = {'is_notes': [], 'bs_notes': [], 'cf_notes': []}
        try:
            import pandas as pd
            xl = pd.ExcelFile(filepath)
            for sheet in xl.sheet_names:
                if '손익주석' in sheet:
                    df = pd.read_excel(filepath, sheet_name=sheet)
                    notes['is_notes'].append({'df': df, 'name': sheet})
                elif '재무주석' in sheet:
                    df = pd.read_excel(filepath, sheet_name=sheet)
                    notes['bs_notes'].append({'df': df, 'name': sheet})
            print(f"[EXTRACT] Excel 주석 로드 완료: IS={len(notes['is_notes'])}개, BS={len(notes['bs_notes'])}개")
            sys.stdout.flush()
        except Exception as e:
            print(f"[EXTRACT] Excel 주석 로드 실패: {e}")
            notes = None
        
        if notes and isinstance(notes, dict):
            # 손익계산서 주석들의 데이터를 is에 병합
            is_notes = notes.get('is_notes', [])
            print(f"[EXTRACT] is_notes 수: {len(is_notes)}, preview_data['is'] 존재: {task['preview_data'].get('is') is not None}")
            if is_notes:
                # preview_data['is']가 없으면 빈 리스트로 시작
                if not task['preview_data'].get('is'):
                    task['preview_data']['is'] = []
                    print(f"[EXTRACT] preview_data['is'] 초기화")
                merged_is_data = list(task['preview_data']['is'])
                for note in is_notes:
                    try:
                        note_df = note['df']
                        if note_df is not None and not note_df.empty:
                            # 중복 FY 컬럼 제거 (FY2024.1, FY2024.2 등 제거하고 FY2024만 유지)
                            cols_to_keep = []
                            seen_fy = set()
                            for col in note_df.columns:
                                col_str = str(col)
                                # FY로 시작하고 .숫자로 끝나는 중복 컬럼 제거
                                if col_str.startswith('FY') and '.' in col_str:
                                    base_fy = col_str.split('.')[0]
                                    if base_fy in seen_fy:
                                        continue  # 중복 FY 컬럼 건너뜀
                                if col_str.startswith('FY'):
                                    base_fy = col_str.split('.')[0] if '.' in col_str else col_str
                                    seen_fy.add(base_fy)
                                cols_to_keep.append(col)
                            note_df_cleaned = note_df[cols_to_keep]
                            note_json = safe_dataframe_to_json(note_df_cleaned)
                            # 중복 제거하며 병합
                            existing_accounts = {row.get('계정과목', '') for row in merged_is_data}
                            for row in note_json:
                                account = row.get('계정과목', '')
                                if account and account not in existing_accounts:
                                    merged_is_data.append(row)
                                    existing_accounts.add(account)
                                    print(f"[EXTRACT] IS 주석 항목 추가: {account}")
                    except Exception as e:
                        print(f"[EXTRACT] IS 주석 병합 실패: {e}")
                        import traceback
                        print(f"[EXTRACT] 상세: {traceback.format_exc()}")
                task['preview_data']['is'] = merged_is_data
                print(f"[EXTRACT] IS 데이터 병합 완료: {len(merged_is_data)}개 항목")
            
            # 재무상태표 주석들도 병합
            bs_notes = notes.get('bs_notes', [])
            if bs_notes and task['preview_data'].get('bs'):
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
                        print(f"[EXTRACT] BS 주석 병합 실패: {e}")
                task['preview_data']['bs'] = merged_bs_data
                print(f"[EXTRACT] BS 데이터 병합 완료: {len(merged_bs_data)}개 항목")
        
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
    
    fs_data = {'bs': None, 'is': None, 'cis': None, 'cf': None, 'notes': None}
    
    # 현재 연도 확인
    current_year = datetime.now().year
    
    # 1차: 사업보고서에서 추출 시도 (XBRL 데이터)
    xbrl_data = None
    has_current_year_annual = False  # 당해년도 사업보고서 존재 여부
    
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
            periodic_filings = search_filings(
                corp_code=corp_code,
                bgn_de=f"{current_year}0101",
                end_de=end_date,
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
        notes_backup = fs_data.get('notes')
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
    except Exception as e:
        print(f"[PAGES] 페이지 처리 실패: {e}")
    
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


def create_vcm_format(fs_data):
    """VCM 전용 포맷 DataFrame 생성 - 감사보고서(FY컬럼)와 사업보고서(XBRL) 모두 지원"""
    import re
    
    bs_df = fs_data.get('bs')
    is_df = fs_data.get('is')
    
    if bs_df is None or is_df is None:
        return None
    
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
    
    # XBRL에서 계정과목 컬럼 찾기 (label_ko가 포함된 튜플 컬럼)
    if is_xbrl:
        for c in is_df.columns:
            if isinstance(c, tuple):
                for part in c:
                    if isinstance(part, str) and 'label_ko' in part:
                        account_col = c
                        break
    
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
    
    # 값 찾기 함수
    def find_val(df, keywords, year, excludes=[]):
        for _, row in df.iterrows():
            # 계정과목 컬럼에서 값 추출 (XBRL vs 감사보고서)
            acc = normalize(str(row.get(account_col, '')))
            excluded = any(normalize(ex) in acc for ex in excludes)
            if excluded: continue
            for kw in keywords:
                if normalize(kw) in acc:
                    val = row.get(year)
                    if pd.notna(val):
                        try:
                            return float(str(val).replace(',', '')) / 1000000
                        except:
                            pass
        return None
    
    # VCM 항목 정의
    vcm_items = []
    
    # 손익계산서 항목
    is_items = [
        ('매출', [('상품매출액', []), ('제품매출액', []), ('기타매출액', [])], 'sum'),
        ('  상품매출', [('상품매출액', ['원가'])], 'find'),
        ('  제품매출', [('제품매출액', ['원가'])], 'find'),
        ('  기타매출', [('기타매출액', ['원가'])], 'find'),
        ('매출원가', [('상품매출원가', []), ('제품매출원가', [])], 'sum'),
        ('  상품매출원가', [('상품매출원가', [])], 'find'),
        ('  제품매출원가', [('제품매출원가', [])], 'find'),
        ('매출총이익', [], 'calc_gross'),
        ('판매비와관리비', [], 'calc_sga'),
        ('  인건비', [('급여', ['퇴직', '연차'])], 'find'),
        ('  임차료비용', [('지급임차료', [])], 'find'),
        ('  수수료비용', [('지급수수료', [])], 'find'),
        ('  보험료', [('보험료', [])], 'find'),
        ('  여비교통비', [('여비교통비', [])], 'find'),
        ('  연구비', [('경상연구개발비', []), ('경상시험연구비', [])], 'sum'),
        ('  감가상각비', [('감가상각비', ['무형'])], 'find'),
        ('  무형자산상각비', [('무형자산상각비', [])], 'find'),
        ('  기타판매비와관리비', [], 'calc_other_sga'),
        ('영업이익', [], 'calc_op'),
        ('  영업외수익', [('이자수익', []), ('배당금수익', []), ('외환차익', []), ('임대료수익', []), ('잡이익', [])], 'sum'),
        ('  금융수익', [('이자수익', [])], 'find'),
        ('  영업외비용', [('이자비용', []), ('외환차손', []), ('투자자산손상차손', []), ('잡손실', [])], 'sum'),
        ('  금융비용', [('이자비용', [])], 'find'),
        ('법인세비용차감전이익', [], 'calc_ebt'),
        ('  법인세비용', [('법인세비용', ['차감전']), ('법인세등', ['차감전'])], 'find'),
        ('당기순이익', [], 'calc_net'),
        ('EBITDA', [], 'calc_ebitda'),
    ]
    
    # 각 연도별 값 계산
    rows = []
    for year in fy_cols:
        # 표시용 컬럼명 (튜플이면 문자열로 변환)
        year_str = fy_col_map[year]
        row_data = {'항목': '', year_str: None}
        
        # 기본 값들 먼저 계산
        상품매출 = find_val(is_df, ['상품매출액'], year, ['원가']) or 0
        제품매출 = find_val(is_df, ['제품매출액'], year, ['원가']) or 0
        기타매출 = find_val(is_df, ['기타매출액'], year, ['원가']) or 0
        상품원가 = find_val(is_df, ['상품매출원가'], year) or 0
        제품원가 = find_val(is_df, ['제품매출원가'], year) or 0
        # 주요 판관비 항목
        급여 = find_val(is_df, ['급여'], year, ['퇴직', '연차']) or 0
        임차료 = find_val(is_df, ['지급임차료'], year) or 0
        수수료 = find_val(is_df, ['지급수수료'], year) or 0
        보험료 = find_val(is_df, ['보험료'], year) or 0
        여비교통비 = find_val(is_df, ['여비교통비'], year) or 0
        연구비1 = find_val(is_df, ['경상연구개발비'], year) or 0
        연구비2 = find_val(is_df, ['경상시험연구비'], year) or 0
        감가상각비 = find_val(is_df, ['감가상각비'], year, ['무형']) or 0
        무형상각비 = find_val(is_df, ['무형자산상각비'], year) or 0
        
        # 기타 판관비 항목
        퇴직급여 = find_val(is_df, ['퇴직급여'], year) or 0
        복리후생비 = find_val(is_df, ['복리후생비'], year) or 0
        접대비 = find_val(is_df, ['접대비'], year) or 0
        통신비 = find_val(is_df, ['통신비'], year) or 0
        세금과공과 = find_val(is_df, ['세금과공과'], year) or 0
        차량유지비 = find_val(is_df, ['차량유지비'], year) or 0
        운반비 = find_val(is_df, ['운반비'], year) or 0
        교육훈련비 = find_val(is_df, ['교육훈련비'], year) or 0
        도서인쇄비 = find_val(is_df, ['도서인쇄비'], year) or 0
        사무용품비 = find_val(is_df, ['사무용품비'], year) or 0
        소모품비 = find_val(is_df, ['소모품비'], year) or 0
        보관료 = find_val(is_df, ['보관료'], year) or 0
        광고선전비 = find_val(is_df, ['광고선전비'], year) or 0
        건물관리비 = find_val(is_df, ['건물관리비'], year) or 0
        대손상각비 = find_val(is_df, ['대손상각비'], year) or 0
        협회비 = find_val(is_df, ['협회비'], year) or 0
        사택관리비 = find_val(is_df, ['사택관리비'], year) or 0
        폐기물처리비 = find_val(is_df, ['폐기물처리비'], year) or 0
        
        # 영업외 수익/비용
        이자수익 = find_val(is_df, ['이자수익'], year) or 0
        배당금 = find_val(is_df, ['배당금수익'], year) or 0
        외환차익 = find_val(is_df, ['외환차익'], year) or 0
        임대료 = find_val(is_df, ['임대료수익'], year) or 0
        잡이익 = find_val(is_df, ['잡이익'], year) or 0
        유형자산처분이익 = find_val(is_df, ['유형자산처분이익'], year) or 0
        이자비용 = find_val(is_df, ['이자비용'], year) or 0
        외환차손 = find_val(is_df, ['외환차손'], year) or 0
        투자손상 = find_val(is_df, ['투자자산손상차손'], year) or 0
        잡손실 = find_val(is_df, ['잡손실'], year) or 0
        매출채권처분손실 = find_val(is_df, ['매출채권처분손실'], year) or 0
        투자자산처분손실 = find_val(is_df, ['투자자산처분손실'], year) or 0
        법인세 = find_val(is_df, ['법인세비용', '법인세등'], year, ['차감전']) or 0
        
        # 계산 값
        매출 = 상품매출 + 제품매출 + 기타매출
        원가 = 상품원가 + 제품원가
        매출총이익 = 매출 - 원가
        연구비 = 연구비1 + 연구비2
        주요판관비 = 급여 + 임차료 + 수수료 + 보험료 + 여비교통비 + 연구비 + 감가상각비 + 무형상각비
        기타판관비 = (퇴직급여 + 복리후생비 + 접대비 + 통신비 + 세금과공과 + 
            차량유지비 + 운반비 + 교육훈련비 + 도서인쇄비 + 사무용품비 + 소모품비 + 
            보관료 + 광고선전비 + 건물관리비 + 대손상각비 + 협회비 + 사택관리비 + 폐기물처리비)
        영업외수익 = 이자수익 + 배당금 + 외환차익 + 임대료 + 잡이익 + 유형자산처분이익
        영업외비용 = 이자비용 + 외환차손 + 투자손상 + 잡손실 + 매출채권처분손실 + 투자자산처분손실
        
        # 합계 계산
        판관비 = 주요판관비 + 기타판관비
        영업이익 = 매출총이익 - 판관비
        세전이익 = 영업이익 + 영업외수익 - 영업외비용
        당기순이익 = 세전이익 - (법인세 or 0)
        ebitda = 영업이익 + 감가상각비 + 무형상각비
        
        values = {
            '매출': 매출,
            '  상품매출': 상품매출,
            '  제품매출': 제품매출,
            '  기타매출': 기타매출 if 기타매출 else None,
            '매출원가': 원가,
            '  상품매출원가': 상품원가,
            '  제품매출원가': 제품원가,
            '매출총이익': 매출총이익,
            '판매비와관리비': 판관비,
            '  인건비': 급여,
            '  임차료비용': 임차료,
            '  수수료비용': 수수료,
            '  보험료': 보험료,
            '  여비교통비': 여비교통비,
            '  연구비': 연구비,
            '  감가상각비': 감가상각비,
            '  무형자산상각비': 무형상각비,
            '  기타판매비와관리비': 기타판관비,
            '영업이익': 영업이익,
            '  영업외수익': 영업외수익,
            '  금융수익': 이자수익,
            '  영업외비용': 영업외비용,
            '  금융비용': 이자비용,
            '법인세비용차감전이익': 세전이익,
            '  법인세비용': 법인세 if 법인세 else None,
            '당기순이익': 당기순이익,
            'EBITDA': ebitda,
        }
        
        for item_name in values:
            if item_name not in [r['항목'] for r in rows]:
                rows.append({'항목': item_name})
        
        for r in rows:
            item_name = r['항목']
            if item_name in values:
                val = values[item_name]
                r[year_str] = round(val) if val is not None and val != 0 else None
    
    return pd.DataFrame(rows)


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


def save_to_excel(fs_data, filepath: str):
    """재무제표를 엑셀 파일로 저장 (주석 테이블 포함)"""
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
            vcm_df = create_vcm_format(fs_data)
            if vcm_df is not None and not vcm_df.empty:
                vcm_df.to_excel(writer, sheet_name='VCM전용포맷', index=False)
        except Exception as e:
            print(f"[VCM] VCM 포맷 생성 실패: {e}")


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
    print("서버 시작: http://localhost:8080")
    print("종료: Ctrl+C")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
