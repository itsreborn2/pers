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
            return {}
    
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
