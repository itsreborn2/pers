"""
DART 기업개황정보 추출 전용 모듈

기존 dart_financial_extractor.py가 재무제표 추출에 특화되어 있어서,
기업개황정보만 빠르게 가져오는 경량 스크립트입니다.

기업개황정보 항목:
- 회사명/영문명
- 대표자명
- 법인번호/사업자번호
- 주소
- 홈페이지/전화번호
- 업종
- 결산월/설립일/상장일
- 주식수/액면가 등
"""

import os
import dart_fss as dart
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class CompanyInfo:
    """기업개황정보 데이터 클래스"""
    corp_code: str = ""         # DART 고유번호
    corp_name: str = ""         # 회사명
    corp_name_eng: str = ""     # 영문 회사명
    stock_code: str = ""        # 종목코드 (상장사만)
    ceo_nm: str = ""            # 대표자명
    corp_cls: str = ""          # 법인구분 (Y:유가, K:코스닥, N:코넥스, E:기타)
    jurir_no: str = ""          # 법인번호
    bizr_no: str = ""           # 사업자번호
    adres: str = ""             # 주소
    hm_url: str = ""            # 홈페이지
    ir_url: str = ""            # IR 홈페이지
    phn_no: str = ""            # 전화번호
    fax_no: str = ""            # 팩스번호
    induty_code: str = ""       # 업종코드
    est_dt: str = ""            # 설립일 (YYYYMMDD)
    acc_mt: str = ""            # 결산월 (MM)
    # 상장 정보
    stock_name: str = ""        # 종목명
    market: str = ""            # 시장구분

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    def get_market_name(self) -> str:
        """시장구분 한글명"""
        market_map = {
            'Y': '코스피',
            'K': '코스닥',
            'N': '코넥스',
            'E': '기타(비상장)'
        }
        return market_map.get(self.corp_cls, '알 수 없음')

    def get_formatted_est_dt(self) -> str:
        """설립일 포맷팅 (YYYY-MM-DD)"""
        if len(self.est_dt) == 8:
            return f"{self.est_dt[:4]}-{self.est_dt[4:6]}-{self.est_dt[6:]}"
        return self.est_dt

    def get_formatted_acc_mt(self) -> str:
        """결산월 포맷팅"""
        if self.acc_mt:
            return f"{self.acc_mt}월"
        return ""


class DartCompanyInfo:
    """DART 기업개황정보 추출 클래스"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: DART API 키. None이면 환경변수 DART_API_KEY 사용
        """
        if api_key:
            dart.set_api_key(api_key)
        elif os.environ.get('DART_API_KEY'):
            dart.set_api_key(os.environ.get('DART_API_KEY'))
        else:
            raise ValueError("DART API 키가 필요합니다.")

        self._corp_list = None

    @property
    def corp_list(self):
        """기업 리스트 lazy loading"""
        if self._corp_list is None:
            print("기업 리스트 로딩 중...")
            self._corp_list = dart.get_corp_list()
            print(f"총 {len(self._corp_list.corps)} 개 기업 로드 완료")
        return self._corp_list

    def search(
        self,
        keyword: str,
        exactly: bool = False,
        market: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        회사명으로 검색 (기본 정보만)

        Args:
            keyword: 검색어
            exactly: 정확히 일치
            market: 시장구분 ('Y', 'K', 'N', 'E', 'YK', 'YKN', None)

        Returns:
            검색 결과 리스트 [{'corp_code': ..., 'corp_name': ..., 'stock_code': ...}, ...]
        """
        corps = self.corp_list.find_by_corp_name(keyword, exactly=exactly, market=market)
        if not corps:
            return []
        if not isinstance(corps, list):
            corps = [corps]

        return [
            {
                'corp_code': c.corp_code,
                'corp_name': c.corp_name,
                'stock_code': c.stock_code or '',
                'corp_cls': c.corp_cls
            }
            for c in corps
        ]

    def get_info(self, corp_code: str) -> Optional[CompanyInfo]:
        """
        기업개황정보 조회

        Args:
            corp_code: DART 고유번호 (8자리)

        Returns:
            CompanyInfo 객체 또는 None
        """
        try:
            corp = self.corp_list.find_by_corp_code(corp_code)
            if not corp:
                return None

            # 상세 정보 로드 (API 호출)
            corp.load()

            # CompanyInfo 객체 생성
            info = CompanyInfo(
                corp_code=corp.corp_code or "",
                corp_name=corp.corp_name or "",
                corp_name_eng=getattr(corp, 'corp_name_eng', "") or "",
                stock_code=corp.stock_code or "",
                ceo_nm=getattr(corp, 'ceo_nm', "") or "",
                corp_cls=corp.corp_cls or "",
                jurir_no=getattr(corp, 'jurir_no', "") or "",
                bizr_no=getattr(corp, 'bizr_no', "") or "",
                adres=getattr(corp, 'adres', "") or "",
                hm_url=getattr(corp, 'hm_url', "") or "",
                ir_url=getattr(corp, 'ir_url', "") or "",
                phn_no=getattr(corp, 'phn_no', "") or "",
                fax_no=getattr(corp, 'fax_no', "") or "",
                induty_code=getattr(corp, 'induty_code', "") or "",
                est_dt=getattr(corp, 'est_dt', "") or "",
                acc_mt=getattr(corp, 'acc_mt', "") or "",
                stock_name=getattr(corp, 'stock_name', "") or "",
                market=corp.corp_cls or ""
            )
            return info

        except Exception as e:
            print(f"기업개황정보 조회 실패: {e}")
            return None

    def get_info_by_name(
        self,
        company_name: str,
        exactly: bool = True
    ) -> Optional[CompanyInfo]:
        """
        회사명으로 기업개황정보 조회 (편의 메서드)

        Args:
            company_name: 회사명
            exactly: 정확히 일치

        Returns:
            CompanyInfo 객체 또는 None
        """
        results = self.search(company_name, exactly=exactly)
        if results:
            return self.get_info(results[0]['corp_code'])
        return None

    def get_info_by_stock_code(self, stock_code: str) -> Optional[CompanyInfo]:
        """
        종목코드로 기업개황정보 조회 (상장사만)

        Args:
            stock_code: 종목코드 (6자리)

        Returns:
            CompanyInfo 객체 또는 None
        """
        try:
            corp = self.corp_list.find_by_stock_code(stock_code)
            if corp:
                return self.get_info(corp.corp_code)
        except Exception as e:
            print(f"종목코드 검색 실패: {e}")
        return None


def main():
    """테스트 실행"""
    try:
        client = DartCompanyInfo()

        # 회사 검색
        print("\n=== 회사 검색: 삼성전자 ===")
        results = client.search("삼성전자")
        for r in results[:5]:
            print(f"  {r['corp_name']} ({r['corp_code']}) - 종목코드: {r['stock_code']}")

        # 기업개황정보 조회
        if results:
            print(f"\n=== {results[0]['corp_name']} 기업개황정보 ===")
            info = client.get_info(results[0]['corp_code'])
            if info:
                print(f"  회사명: {info.corp_name}")
                print(f"  영문명: {info.corp_name_eng}")
                print(f"  대표자: {info.ceo_nm}")
                print(f"  시장: {info.get_market_name()}")
                print(f"  종목코드: {info.stock_code}")
                print(f"  법인번호: {info.jurir_no}")
                print(f"  사업자번호: {info.bizr_no}")
                print(f"  주소: {info.adres}")
                print(f"  홈페이지: {info.hm_url}")
                print(f"  전화번호: {info.phn_no}")
                print(f"  업종코드: {info.induty_code}")
                print(f"  설립일: {info.get_formatted_est_dt()}")
                print(f"  결산월: {info.get_formatted_acc_mt()}")

        # 종목코드로 조회
        print("\n=== 종목코드로 조회: 005930 ===")
        info = client.get_info_by_stock_code("005930")
        if info:
            print(f"  {info.corp_name} - {info.ceo_nm}")

    except ValueError as e:
        print(f"오류: {e}")
        print("DART_API_KEY 환경변수를 설정하세요.")


if __name__ == "__main__":
    main()
