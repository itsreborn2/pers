"""
DART API를 사용한 재무제표 추출 모듈

이 모듈은 dart-fss 라이브러리를 사용하여 상장/비상장 기업의 재무제표를 추출합니다.
- 재무상태표 (Balance Sheet)
- 손익계산서 (Income Statement)
- 포괄손익계산서 (Comprehensive Income Statement)
- 현금흐름표 (Cash Flow Statement)

사용 전 DART API 키 설정 필요:
    import dart_fss as dart
    dart.set_api_key('YOUR_API_KEY')
    
또는 환경변수 설정:
    set DART_API_KEY=YOUR_API_KEY (Windows)
    export DART_API_KEY=YOUR_API_KEY (Linux/Mac)
"""

import os
import dart_fss as dart
import pandas as pd
from typing import Optional, List, Dict, Any, Union
from datetime import datetime


class DartFinancialExtractor:
    """
    DART API를 사용하여 기업 재무제표를 추출하는 클래스
    
    상장사(코스피, 코스닥, 코넥스)와 비상장사 모두 검색 가능
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        DartFinancialExtractor 초기화
        
        Args:
            api_key: DART API 키. None이면 환경변수 DART_API_KEY 사용
        """
        # API 키 설정
        if api_key:
            dart.set_api_key(api_key)
        elif os.environ.get('DART_API_KEY'):
            dart.set_api_key(os.environ.get('DART_API_KEY'))
        else:
            raise ValueError("DART API 키가 필요합니다. api_key 파라미터 또는 DART_API_KEY 환경변수를 설정하세요.")
        
        # 기업 리스트 로드 (상장/비상장 모두 포함)
        self._corp_list = None
    
    @property
    def corp_list(self):
        """기업 리스트를 lazy loading으로 반환"""
        if self._corp_list is None:
            print("기업 리스트 로딩 중...")
            self._corp_list = dart.get_corp_list()
            print(f"총 {len(self._corp_list.corps)} 개 기업 로드 완료")
        return self._corp_list
    
    def search_company(
        self, 
        company_name: str, 
        exactly: bool = False,
        market: Optional[str] = None
    ) -> List[Any]:
        """
        회사명으로 기업 검색
        
        Args:
            company_name: 검색할 회사명
            exactly: True면 정확히 일치하는 회사만 검색
            market: 시장 구분 필터
                    - 'Y': 코스피
                    - 'K': 코스닥  
                    - 'N': 코넥스
                    - 'E': 기타 (비상장 포함)
                    - None: 전체 검색 (상장/비상장 모두)
                    - 'YK': 코스피 + 코스닥
                    - 'YKN': 코스피 + 코스닥 + 코넥스
        
        Returns:
            검색된 기업 리스트
        """
        corps = self.corp_list.find_by_corp_name(company_name, exactly=exactly, market=market)
        return corps if isinstance(corps, list) else [corps] if corps else []
    
    def search_by_corp_code(self, corp_code: str) -> Any:
        """
        DART 고유번호로 기업 검색
        
        Args:
            corp_code: DART 고유번호 (8자리)
            
        Returns:
            기업 정보 객체
        """
        return self.corp_list.find_by_corp_code(corp_code)
    
    def search_by_stock_code(self, stock_code: str) -> Any:
        """
        주식 종목코드로 기업 검색 (상장사만)
        
        Args:
            stock_code: 주식 종목코드 (6자리)
            
        Returns:
            기업 정보 객체
        """
        return self.corp_list.find_by_stock_code(stock_code)
    
    def extract_financial_statements(
        self,
        corp_code: str,
        start_date: str = '20200101',
        end_date: Optional[str] = None,
        fs_types: List[str] = ['bs', 'is', 'cis', 'cf'],
        separate: bool = False,
        report_tp: str = 'annual',
        lang: str = 'ko',
        separator: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        기업의 재무제표 추출
        
        Args:
            corp_code: DART 고유번호 (8자리)
            start_date: 검색 시작일 (YYYYMMDD)
            end_date: 검색 종료일 (YYYYMMDD), None이면 오늘
            fs_types: 추출할 재무제표 유형
                      - 'bs': 재무상태표 (Balance Sheet)
                      - 'is': 손익계산서 (Income Statement)
                      - 'cis': 포괄손익계산서 (Comprehensive Income Statement)
                      - 'cf': 현금흐름표 (Cash Flow Statement)
            separate: True면 개별재무제표, False면 연결재무제표
            report_tp: 보고서 유형
                       - 'annual': 연간
                       - 'half': 연간 + 반기
                       - 'quarter': 연간 + 반기 + 분기
            lang: 언어 ('ko': 한글, 'en': 영문)
            separator: 1000단위 구분자 표시 여부
            
        Returns:
            재무제표 딕셔너리 {'bs': DataFrame, 'is': DataFrame, ...}
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        try:
            # 재무제표 추출
            fs = dart.fs.extract(
                corp_code=corp_code,
                bgn_de=start_date,
                end_de=end_date,
                fs_tp=fs_types,
                separate=separate,
                report_tp=report_tp,
                lang=lang,
                separator=separator
            )
            
            result = {}
            
            # 재무상태표
            if 'bs' in fs_types:
                try:
                    result['bs'] = fs['bs']
                except:
                    result['bs'] = None
            
            # 손익계산서
            if 'is' in fs_types:
                try:
                    result['is'] = fs['is']
                except:
                    result['is'] = None
            
            # 포괄손익계산서
            if 'cis' in fs_types:
                try:
                    result['cis'] = fs['cis']
                except:
                    result['cis'] = None
            
            # 현금흐름표
            if 'cf' in fs_types:
                try:
                    result['cf'] = fs['cf']
                except:
                    result['cf'] = None
            
            return result

        except Exception as e:
            print(f"재무제표 추출 실패: {e}")
            # 비상장사: 감사보고서에서 재무제표 추출 시도
            print(f"  → 감사보고서에서 재무제표 추출 시도...")
            try:
                audit_result = self.extract_financial_statements_from_audit_report(
                    corp_code=corp_code,
                    start_date=start_date,
                    end_date=end_date
                )
                if audit_result:
                    return audit_result
            except Exception as e2:
                print(f"  → 감사보고서 재무제표 추출도 실패: {e2}")
            return {}

    def extract_financial_statements_from_audit_report(
        self,
        corp_code: str,
        start_date: str = '20200101',
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        비상장사용: 감사보고서에서 재무제표 추출 (HTML 테이블 파싱)

        Args:
            corp_code: DART 고유번호 (8자리)
            start_date: 검색 시작일 (YYYYMMDD)
            end_date: 검색 종료일 (YYYYMMDD)

        Returns:
            재무제표 딕셔너리 {'bs': DataFrame, 'is': DataFrame, 'cf': DataFrame}
        """
        from bs4 import BeautifulSoup
        from io import StringIO
        import re

        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        try:
            corp = self.corp_list.find_by_corp_code(corp_code)
            if not corp:
                return {}

            # 감사보고서 검색
            reports = corp.search_filings(
                bgn_de=start_date,
                end_de=end_date,
                pblntf_ty='F'  # 감사보고서
            )

            if not reports or len(reports) == 0:
                print(f"  → 감사보고서를 찾을 수 없습니다: {corp_code}")
                return {}

            result = {'bs': None, 'is': None, 'cis': None, 'cf': None}
            all_bs_data = []
            all_is_data = []
            all_cf_data = []

            # 여러 연도의 감사보고서에서 데이터 수집
            for report in reports[:5]:  # 최근 5개 연도
                try:
                    pages = report.extract_pages()

                    # 페이지별로 재무제표 찾기
                    for page in pages:
                        title = getattr(page, 'title', '') or ''

                        if '재무상태표' in title or '재 무 상 태 표' in title:
                            df = self._parse_audit_report_table(page.html, 'bs')
                            if df is not None and not df.empty:
                                # 연도 정보 추가
                                year = report.rcept_dt[:4] if hasattr(report, 'rcept_dt') else ''
                                df['report_year'] = year
                                all_bs_data.append(df)

                        elif '손익계산서' in title or '손 익 계 산 서' in title:
                            df = self._parse_audit_report_table(page.html, 'is')
                            if df is not None and not df.empty:
                                year = report.rcept_dt[:4] if hasattr(report, 'rcept_dt') else ''
                                df['report_year'] = year
                                all_is_data.append(df)

                        elif '현금흐름표' in title or '현 금 흐 름 표' in title:
                            df = self._parse_audit_report_table(page.html, 'cf')
                            if df is not None and not df.empty:
                                year = report.rcept_dt[:4] if hasattr(report, 'rcept_dt') else ''
                                df['report_year'] = year
                                all_cf_data.append(df)

                except Exception as e:
                    print(f"  → 보고서 파싱 오류: {e}")
                    continue

            # 데이터 병합
            if all_bs_data:
                result['bs'] = pd.concat(all_bs_data, ignore_index=True)
                print(f"  → 재무상태표: {len(result['bs'])}행")
            if all_is_data:
                result['is'] = pd.concat(all_is_data, ignore_index=True)
                print(f"  → 손익계산서: {len(result['is'])}행")
            if all_cf_data:
                result['cf'] = pd.concat(all_cf_data, ignore_index=True)
                print(f"  → 현금흐름표: {len(result['cf'])}행")

            return result

        except Exception as e:
            print(f"감사보고서 재무제표 추출 실패: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _parse_audit_report_table(self, html: str, fs_type: str) -> Optional[pd.DataFrame]:
        """
        감사보고서 HTML 테이블을 DataFrame으로 파싱

        Args:
            html: HTML 문자열
            fs_type: 재무제표 유형 ('bs', 'is', 'cf')

        Returns:
            파싱된 DataFrame
        """
        from bs4 import BeautifulSoup
        from io import StringIO
        import re

        try:
            soup = BeautifulSoup(html, 'html.parser')
            tables = soup.find_all('table')

            # 가장 큰 테이블 찾기 (메인 재무제표 테이블)
            main_table = None
            max_rows = 0

            for table in tables:
                dfs = pd.read_html(StringIO(str(table)))
                if dfs and len(dfs[0]) > max_rows:
                    max_rows = len(dfs[0])
                    main_table = dfs[0]

            if main_table is None or main_table.empty:
                return None

            df = main_table

            # 컬럼명 정리
            # 첫 번째 컬럼을 'label_ko'로
            df.columns = ['label_ko'] + [f'col_{i}' for i in range(1, len(df.columns))]

            # '과목', '과 목' 등 헤더 행 제거
            if df['label_ko'].iloc[0] in ['과목', '과 목', '구분', '구 분']:
                df = df.iloc[1:]

            # NaN만 있는 행 제거
            df = df.dropna(how='all')

            # label_ko가 NaN인 행 제거
            df = df[df['label_ko'].notna()]

            # 숫자 컬럼 정리 (쉼표, 괄호 처리)
            for col in df.columns[1:]:
                df[col] = df[col].apply(self._clean_number)

            df = df.reset_index(drop=True)

            return df

        except Exception as e:
            print(f"  → 테이블 파싱 오류: {e}")
            return None

    def _clean_number(self, value):
        """숫자 값 정리 (쉼표, 괄호 처리)"""
        import re

        if pd.isna(value):
            return None

        value_str = str(value).strip()

        if value_str in ['-', '', 'NaN', 'nan']:
            return 0

        # 괄호 = 음수
        is_negative = '(' in value_str and ')' in value_str

        # 숫자만 추출
        cleaned = re.sub(r'[^\d.]', '', value_str)

        if not cleaned:
            return None

        try:
            num = float(cleaned)
            return -num if is_negative else num
        except:
            return None

    def extract_financial_summary(
        self,
        corp_code: str,
        start_year: int = 2020,
        end_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        이미지와 유사한 형태의 재무 요약 정보 추출
        
        재무상태표 주요 항목:
        - 유동자산, 비유동자산, 자산총계
        - 유동부채, 비유동부채, 부채총계
        - 자본총계, 부채와자본총계
        - NWC (순운전자본), Net Debt (순차입금)
        
        손익계산서 주요 항목:
        - 매출, 매출원가, 매출총이익
        - 판매비와관리비, 영업이익
        - 영업외수익, 영업외비용
        - 법인세비용차감전이익, 당기순이익
        - EBITDA
        
        Args:
            corp_code: DART 고유번호
            start_year: 시작 연도
            end_year: 종료 연도 (None이면 현재 연도)
            
        Returns:
            재무 요약 딕셔너리
        """
        if end_year is None:
            end_year = datetime.now().year
        
        start_date = f"{start_year}0101"
        end_date = f"{end_year}1231"
        
        # 재무제표 추출
        fs_data = self.extract_financial_statements(
            corp_code=corp_code,
            start_date=start_date,
            end_date=end_date,
            fs_types=['bs', 'is'],
            report_tp='annual',
            separator=False
        )
        
        result = {
            'balance_sheet': {},  # 재무상태표
            'income_statement': {},  # 손익계산서
            'raw_data': fs_data  # 원본 데이터
        }
        
        # 재무상태표 주요 항목 추출
        if fs_data.get('bs') is not None:
            bs = fs_data['bs']
            result['balance_sheet'] = self._extract_bs_items(bs)
        
        # 손익계산서 주요 항목 추출
        if fs_data.get('is') is not None:
            is_df = fs_data['is']
            result['income_statement'] = self._extract_is_items(is_df)
        
        return result
    
    def _extract_bs_items(self, bs_df: pd.DataFrame) -> Dict[str, Any]:
        """재무상태표에서 주요 항목 추출"""
        items = {}
        
        # 주요 계정과목 매핑 (한글명 -> 영문 키)
        bs_mapping = {
            '유동자산': 'current_assets',
            '현금및현금성자산': 'cash_and_equivalents',
            '재고자산': 'inventories',
            '매출채권': 'trade_receivables',
            '비유동자산': 'non_current_assets',
            '유형자산': 'tangible_assets',
            '무형자산': 'intangible_assets',
            '자산총계': 'total_assets',
            '유동부채': 'current_liabilities',
            '매입채무': 'trade_payables',
            '단기차입금': 'short_term_borrowings',
            '비유동부채': 'non_current_liabilities',
            '장기차입금': 'long_term_borrowings',
            '부채총계': 'total_liabilities',
            '자본금': 'capital_stock',
            '이익잉여금': 'retained_earnings',
            '자본총계': 'total_equity',
        }
        
        # DataFrame에서 항목 추출
        if bs_df is not None and not bs_df.empty:
            for kor_name, eng_key in bs_mapping.items():
                try:
                    # label_ko 컬럼에서 해당 항목 찾기
                    if 'label_ko' in bs_df.columns:
                        mask = bs_df['label_ko'].str.contains(kor_name, na=False)
                        if mask.any():
                            items[eng_key] = bs_df[mask].iloc[0].to_dict()
                except Exception:
                    pass
        
        return items
    
    def _extract_is_items(self, is_df: pd.DataFrame) -> Dict[str, Any]:
        """손익계산서에서 주요 항목 추출"""
        items = {}
        
        # 주요 계정과목 매핑
        is_mapping = {
            '매출액': 'revenue',
            '매출': 'revenue',
            '매출원가': 'cost_of_sales',
            '매출총이익': 'gross_profit',
            '판매비와관리비': 'sg_and_a',
            '영업이익': 'operating_income',
            '영업외수익': 'non_operating_income',
            '영업외비용': 'non_operating_expense',
            '법인세비용차감전순이익': 'income_before_tax',
            '법인세비용': 'income_tax_expense',
            '당기순이익': 'net_income',
        }
        
        if is_df is not None and not is_df.empty:
            for kor_name, eng_key in is_mapping.items():
                try:
                    if 'label_ko' in is_df.columns:
                        mask = is_df['label_ko'].str.contains(kor_name, na=False)
                        if mask.any():
                            items[eng_key] = is_df[mask].iloc[0].to_dict()
                except Exception:
                    pass
        
        return items
    
    def get_company_info(self, corp_code: str) -> Dict[str, Any]:
        """
        기업 기본 정보 조회
        
        Args:
            corp_code: DART 고유번호
            
        Returns:
            기업 정보 딕셔너리
        """
        corp = self.corp_list.find_by_corp_code(corp_code)
        if corp:
            corp.load()  # 상세 정보 로드
            return corp.to_dict()
        return {}
    
    def save_to_excel(
        self,
        fs_data: Dict[str, pd.DataFrame],
        filename: str,
        path: str = './'
    ):
        """
        재무제표를 엑셀 파일로 저장
        
        Args:
            fs_data: 재무제표 딕셔너리
            filename: 저장할 파일명 (확장자 제외)
            path: 저장 경로
        """
        filepath = os.path.join(path, f"{filename}.xlsx")
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in fs_data.items():
                if df is not None and not df.empty:
                    # 시트명 매핑
                    sheet_names = {
                        'bs': '재무상태표',
                        'is': '손익계산서',
                        'cis': '포괄손익계산서',
                        'cf': '현금흐름표'
                    }
                    name = sheet_names.get(sheet_name, sheet_name)
                    df.to_excel(writer, sheet_name=name, index=False)
        
        print(f"저장 완료: {filepath}")

    def extract_notes(
        self,
        corp_code: str,
        year: int = None,
        report_type: str = 'annual'
    ) -> Dict[str, Any]:
        """
        사업보고서에서 재무제표 주석 추출

        Args:
            corp_code: DART 고유번호 (8자리)
            year: 대상 연도 (None이면 최근 연도)
            report_type: 보고서 유형 ('annual': 사업보고서)

        Returns:
            {
                'year': 연도,
                'report_name': 보고서명,
                'notes': [주석 섹션 리스트],
                'notes_text': 주석 전체 텍스트
            }
        """
        import re
        from bs4 import BeautifulSoup

        try:
            # 기업 검색
            corp = self.corp_list.find_by_corp_code(corp_code)
            if not corp:
                return {'error': f'기업을 찾을 수 없습니다: {corp_code}'}

            # 사업보고서 검색 (pblntf_detail_ty='a001' = 사업보고서)
            if year:
                bgn_de = f'{year}0101'
                end_de = f'{year}1231'
            else:
                # 최근 2년 검색
                from datetime import datetime
                current_year = datetime.now().year
                bgn_de = f'{current_year - 2}0101'
                end_de = datetime.now().strftime('%Y%m%d')

            # 사업보고서 검색 시도
            reports = None
            report_source = '사업보고서'
            try:
                reports = corp.search_filings(
                    bgn_de=bgn_de,
                    end_de=end_de,
                    pblntf_detail_ty='a001'  # 사업보고서
                )
            except Exception as e:
                print(f"  → 사업보고서 검색 실패: {e}")
                reports = None

            # 사업보고서가 없으면 감사보고서에서 시도 (비상장사)
            if not reports or len(reports) == 0:
                print(f"  → 사업보고서 없음, 감사보고서에서 시도...")
                try:
                    reports = corp.search_filings(
                        bgn_de=bgn_de,
                        end_de=end_de,
                        pblntf_ty='F'  # 감사보고서
                    )
                    report_source = '감사보고서'
                except Exception as e:
                    print(f"  → 감사보고서 검색도 실패: {e}")
                    reports = None

            if not reports or len(reports) == 0:
                return {'error': f'사업보고서/감사보고서를 찾을 수 없습니다: {corp_code}, {year}'}

            # 가장 최근 사업보고서 선택
            report = reports[0]

            # 모든 페이지 추출
            all_pages = report.extract_pages()

            # 주석 관련 페이지 필터링
            notes_keywords = ['주석', '재무제표에 대한 주석', '연결재무제표주석', '별도재무제표주석']
            pages = []
            for page in all_pages:
                page_title = getattr(page, 'title', '') or ''
                if any(keyword in page_title for keyword in notes_keywords):
                    pages.append(page)

            notes_sections = []
            notes_text_parts = []

            for page in pages:
                try:
                    # HTML을 텍스트로 변환
                    if hasattr(page, 'html'):
                        soup = BeautifulSoup(page.html, 'html.parser')

                        # 테이블 내용 추출 (주석에 많은 수치 정보가 테이블에 있음)
                        text = soup.get_text(separator='\n', strip=True)

                        # 불필요한 공백 제거
                        text = re.sub(r'\n{3,}', '\n\n', text)

                        if text.strip():
                            section_title = page.title if hasattr(page, 'title') else '주석'
                            notes_sections.append({
                                'title': section_title,
                                'content': text[:50000]  # 최대 50000자로 제한
                            })
                            notes_text_parts.append(f"### {section_title}\n{text[:50000]}")

                    elif hasattr(page, 'text'):
                        text = page.text
                        if text.strip():
                            section_title = page.title if hasattr(page, 'title') else '주석'
                            notes_sections.append({
                                'title': section_title,
                                'content': text[:50000]
                            })
                            notes_text_parts.append(f"### {section_title}\n{text[:50000]}")

                except Exception as e:
                    print(f"  주석 페이지 파싱 오류: {e}")
                    continue

            # 연도 추출
            report_year = year
            if not report_year and hasattr(report, 'rcept_dt'):
                report_year = int(report.rcept_dt[:4])

            result = {
                'year': report_year,
                'report_name': report.report_nm if hasattr(report, 'report_nm') else report_source,
                'report_source': report_source,
                'notes': notes_sections,
                'notes_text': '\n\n'.join(notes_text_parts) if notes_text_parts else '주석 내용을 찾을 수 없습니다.',
                'notes_count': len(notes_sections)
            }

            print(f"  → 주석 추출 완료: {len(notes_sections)}개 섹션 ({report_source})")
            return result

        except Exception as e:
            print(f"주석 추출 실패: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'notes': [], 'notes_text': ''}


def main():
    """사용 예시"""
    # API 키 설정 (환경변수 또는 직접 입력)
    # os.environ['DART_API_KEY'] = 'YOUR_API_KEY'
    
    try:
        extractor = DartFinancialExtractor()
        
        # 1. 회사 검색 (상장/비상장 모두)
        print("\n=== 회사 검색 ===")
        companies = extractor.search_company("삼성전자", exactly=True)
        for corp in companies:
            print(f"회사명: {corp.corp_name}, 고유번호: {corp.corp_code}, 종목코드: {corp.stock_code}")
        
        if companies:
            corp = companies[0]
            corp_code = corp.corp_code
            
            # 2. 재무제표 추출
            print(f"\n=== {corp.corp_name} 재무제표 추출 ===")
            fs_data = extractor.extract_financial_statements(
                corp_code=corp_code,
                start_date='20200101',
                fs_types=['bs', 'is'],
                report_tp='annual'
            )
            
            # 3. 재무상태표 출력
            if fs_data.get('bs') is not None:
                print("\n[재무상태표]")
                print(fs_data['bs'].head(20))
            
            # 4. 손익계산서 출력
            if fs_data.get('is') is not None:
                print("\n[손익계산서]")
                print(fs_data['is'].head(20))
            
            # 5. 엑셀로 저장
            # extractor.save_to_excel(fs_data, f"{corp.corp_name}_재무제표")
            
    except ValueError as e:
        print(f"오류: {e}")
        print("DART API 키를 설정해주세요.")
        print("  - 환경변수: set DART_API_KEY=YOUR_API_KEY")
        print("  - 또는 코드에서: DartFinancialExtractor(api_key='YOUR_API_KEY')")


if __name__ == "__main__":
    main()
