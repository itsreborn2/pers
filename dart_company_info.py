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
            self.api_key = api_key
        elif os.environ.get('DART_API_KEY'):
            dart.set_api_key(os.environ.get('DART_API_KEY'))
            self.api_key = os.environ.get('DART_API_KEY')
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

    def get_address_from_report(self, corp_code: str) -> Optional[str]:
        """
        최근 사업보고서에서 본점소재지(주소) 추출

        Args:
            corp_code: DART 고유번호 (8자리)

        Returns:
            본점소재지 문자열 또는 None
        """
        import requests
        import zipfile
        import io
        import re
        from bs4 import BeautifulSoup

        try:
            # 1. 최근 사업보고서 검색 (A001: 사업보고서)
            url_list = "https://opendart.fss.or.kr/api/list.json"
            params = {
                "crtfc_key": self.api_key,
                "corp_code": corp_code,
                "bgn_de": "20200101",
                "pblntf_detail_ty": "A001",  # 사업보고서
                "page_count": "5"
            }
            res = requests.get(url_list, params=params, timeout=10)
            data = res.json()

            if data.get('status') != '000' or not data.get('list'):
                print(f"사업보고서 없음: {data.get('message', 'No reports')}")
                return None

            # 가장 최근 사업보고서의 접수번호
            rcept_no = data['list'][0]['rcept_no']
            print(f"최근 사업보고서 접수번호: {rcept_no}")

            # 2. 공시서류원본파일 다운로드
            url_doc = "https://opendart.fss.or.kr/api/document.xml"
            params = {
                "crtfc_key": self.api_key,
                "rcept_no": rcept_no
            }
            response = requests.get(url_doc, params=params, timeout=30)

            # 3. Content-Type에 따라 처리
            content_type = response.headers.get("Content-Type", "").lower()
            xml_content = None

            if "xml" in content_type:
                xml_content = response.content
            elif "zip" in content_type or "msdownload" in content_type:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    # ZIP 내 XML 파일 찾기
                    for filename in z.namelist():
                        if filename.endswith('.xml'):
                            xml_content = z.read(filename)
                            break

            if not xml_content:
                print("XML 컨텐츠를 찾을 수 없음")
                return None

            # 4. XML 파싱하여 본점소재지 추출
            soup = BeautifulSoup(xml_content, "lxml-xml")
            text = soup.get_text("\n", strip=True)

            # 본점소재지 패턴 검색
            patterns = [
                r'본점[의\s]*소재지[:\s]*([^\n]{10,100})',
                r'본점[의\s]*주소[:\s]*([^\n]{10,100})',
                r'주\s*소[:\s]*([가-힣]+[도시]\s+[가-힣]+[시군구][^\n]{5,80})',
            ]

            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    address = match.group(1).strip()
                    # 불필요한 문자 제거
                    address = re.sub(r'[\(\)].*$', '', address).strip()
                    address = re.sub(r'\s+', ' ', address)
                    if len(address) > 10:
                        print(f"사업보고서 주소 추출 성공: {address}")
                        return address

            print("본점소재지 패턴 매칭 실패")
            return None

        except Exception as e:
            print(f"사업보고서 주소 추출 실패: {e}")
            return None

    def get_company_info_from_report(self, corp_code: str) -> dict:
        """
        최근 사업보고서에서 대표자명과 본점소재지 추출

        Args:
            corp_code: DART 고유번호 (8자리)

        Returns:
            {'ceo': 대표자명 또는 None, 'address': 본점소재지 또는 None}
        """
        import requests
        import zipfile
        import io
        import re
        from bs4 import BeautifulSoup

        result = {'ceo': None, 'address': None}

        try:
            # 1. 최근 보고서 검색 (사업보고서 > 반기보고서 > 분기보고서 > 감사보고서 순)
            url_list = "https://opendart.fss.or.kr/api/list.json"
            report_types = [
                ("A001", "사업보고서"),
                ("A002", "반기보고서"),
                ("A003", "분기보고서"),
                ("F001", "감사보고서"),  # 정기보고서 없는 회사용
            ]

            rcept_no = None
            report_type_name = None

            for report_type, type_name in report_types:
                params = {
                    "crtfc_key": self.api_key,
                    "corp_code": corp_code,
                    "bgn_de": "20200101",
                    "pblntf_detail_ty": report_type,
                    "page_count": "5"
                }
                res = requests.get(url_list, params=params, timeout=10)
                data = res.json()

                if data.get('status') == '000' and data.get('list'):
                    rcept_no = data['list'][0]['rcept_no']
                    report_type_name = type_name
                    print(f"최근 {type_name} 접수번호: {rcept_no}")
                    break
                else:
                    print(f"{type_name} 없음")

            if not rcept_no:
                print(f"보고서 없음: 사업/반기/분기/감사보고서 모두 없음")
                return {'ceo': None, 'address': None, 'no_report': True}

            # 2. 공시서류원본파일 다운로드
            url_doc = "https://opendart.fss.or.kr/api/document.xml"
            params = {
                "crtfc_key": self.api_key,
                "rcept_no": rcept_no
            }
            response = requests.get(url_doc, params=params, timeout=30)

            # 3. Content-Type에 따라 처리
            content_type = response.headers.get("Content-Type", "").lower()
            xml_content = None

            if "xml" in content_type:
                xml_content = response.content
            elif "zip" in content_type or "msdownload" in content_type:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    for filename in z.namelist():
                        if filename.endswith('.xml'):
                            xml_content = z.read(filename)
                            break

            if not xml_content:
                print("XML 컨텐츠를 찾을 수 없음")
                return result

            # 4. XML 파싱
            soup = BeautifulSoup(xml_content, "lxml-xml")
            text = soup.get_text("\n", strip=True)

            # 대표자 패턴 검색
            # 제외할 단어 (이름이 아닌 일반 단어)
            excluded_words = {'등의', '등이', '외의', '기타', '성명', '이름', '직위', '직책', '대표', '이사', '사내', '사외', '상무', '전무', '부사장', '회장', '부회장', '확인', '사장'}

            # 1단계: 표지 패턴 우선 검색 (띄어쓰기 포함된 표지 형식)
            cover_patterns = [
                r'대\s*표\s*이\s*사\s*[:：]\s*([가-힣]{2,10})',  # "대 표 이 사 : 김진국"
                r'대\s*표\s*자\s*[:：]\s*([가-힣]{2,10})',       # "대 표 자 : 김진국"
            ]
            for pattern in cover_patterns:
                match = re.search(pattern, text)
                if match:
                    ceo = match.group(1).strip()
                    if 2 <= len(ceo) <= 10 and ceo not in excluded_words:
                        print(f"사업보고서 표지 대표자 추출 성공: {ceo}")
                        result['ceo'] = ceo
                        break

            # 2단계: 표지에서 못 찾은 경우 기존 패턴으로 폴백
            if not result['ceo']:
                ceo_patterns = [
                    r'대표이사\s+([가-힣]{2,10})\s*$',  # "대표이사 신학철" (줄 끝)
                    r'대표이사\s+([가-힣]{2,10})\n',    # "대표이사 신학철\n"
                    r'대표이사[:\s]+([가-힣]{2,10})',
                    r'대표이사\s*성\s*명[:\s]+([가-힣]{2,10})',
                    r'대표이사\([^)]*\)[:\s]+([가-힣]{2,10})',
                ]
                for pattern in ceo_patterns:
                    matches = re.findall(pattern, text, re.MULTILINE)
                    for ceo in matches:
                        ceo = ceo.strip()
                        if 2 <= len(ceo) <= 10 and ceo not in excluded_words:
                            print(f"사업보고서 대표자 추출 성공: {ceo}")
                            result['ceo'] = ceo
                            break
                    if result['ceo']:
                        break

            # 본점소재지 패턴 검색 (더 정교한 패턴)
            # 시/도 목록 (약어 + 전체 이름)
            provinces = r'(?:서울특별시|서울|부산광역시|부산|대구광역시|대구|인천광역시|인천|광주광역시|광주|대전광역시|대전|울산광역시|울산|세종특별자치시|세종|경기도|경기|강원특별자치도|강원도|강원|충청북도|충북|충청남도|충남|전북특별자치도|전라북도|전북|전라남도|전남|경상북도|경북|경상남도|경남|제주특별자치도|제주)'

            address_patterns = [
                # 패턴 1: "본점의 소재지" 또는 "본점 소재지" 뒤에 같은 줄에 주소
                rf'본점[의\s]*소재지[는:\s]*({provinces}[^\n\.]+)',
                # 패턴 2: "본점 주소" 뒤에 오는 주소
                rf'본점[의\s]*주소[는:\s]*({provinces}[^\n\.]+)',
                # 패턴 3: "주된 사무소 소재지"
                rf'주된\s*사무소[의\s]*소재지[는:\s]*({provinces}[^\n\.]+)',
            ]

            # 먼저 같은 줄 패턴 시도
            for pattern in address_patterns:
                match = re.search(pattern, text)
                if match:
                    address = match.group(1).strip()
                    address = re.sub(r'입니다.*$', '', address).strip()
                    address = re.sub(r'이며.*$', '', address).strip()
                    # 전화번호/홈페이지/팩스 등 주소 이후 정보 제거
                    address = re.sub(r'\s*(?:전화|TEL|tel|Tel|팩스|FAX|fax|Fax|홈페이지|http|www\.).*$', '', address, flags=re.IGNORECASE).strip()
                    # 끝에 남은 하이픈/특수문자 정리
                    address = re.sub(r'[\s\-–—]+$', '', address)
                    address = re.sub(r'\s+', ' ', address)
                    if len(address) >= 10 and re.search(r'[시구군]', address):
                        print(f"사업보고서 주소 추출 성공: {address}")
                        result['address'] = address
                        break

            # 같은 줄에서 못 찾으면 다음 줄에서 찾기 (줄바꿈 형식)
            if not result['address']:
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    if '본점' in line and '소재지' in line:
                        # 다음 몇 줄에서 주소 찾기
                        for j in range(1, 5):
                            if i + j < len(lines):
                                next_line = lines[i + j].strip()
                                # 시/도로 시작하는 줄 찾기
                                if re.match(provinces, next_line):
                                    address = next_line
                                    # 괄호 안 동/리 정보까지만 포함
                                    match_addr = re.match(rf'({provinces}[가-힣0-9\s\-]+(?:\([가-힣0-9\-]+\))?)', address)
                                    if match_addr:
                                        address = match_addr.group(1).strip()
                                        if len(address) >= 10:
                                            print(f"사업보고서 주소 추출 성공 (다음 줄): {address}")
                                            result['address'] = address
                                            break
                        if result['address']:
                            break

            return result

        except Exception as e:
            print(f"사업보고서 정보 추출 실패: {e}")
            return result


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
