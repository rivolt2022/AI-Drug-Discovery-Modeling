"""
모델 학습을 위한 데이터 로딩 모듈

이 모듈은 read_excel.py와 read_csv.py의 함수들을 활용하여
모델 학습에 필요한 정제된 데이터를 제공합니다.
"""

import pandas as pd
import numpy as np
import os
import sys

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing.read_excel import read_cas_excel, read_ligand_smiles_sheet, read_ic50_sheet
from data.preprocessing.read_csv import read_test_csv, read_chembl_csv, get_clean_training_data

class DrugDiscoveryDataLoader:
    """AI 신약개발 모델링을 위한 데이터 로더 클래스"""
    
    def __init__(self, data_dir='data'):
        """
        데이터 로더 초기화
        
        Args:
            data_dir (str): 데이터 디렉토리 경로
        """
        self.data_dir = data_dir
        self.cache = {}  # 데이터 캐싱을 위한 딕셔너리
        
    def load_test_data(self, use_cache=True):
        """
        테스트 데이터 로드
        
        Args:
            use_cache (bool): 캐시된 데이터 사용 여부
            
        Returns:
            pd.DataFrame: 테스트 데이터 (ID, SMILES)
        """
        if use_cache and 'test_data' in self.cache:
            print("📋 캐시된 테스트 데이터 사용")
            return self.cache['test_data']
        
        print("�� 테스트 데이터 로딩 중...")
        test_df = read_test_csv()
        
        if test_df is not None:
            # 컬럼명 정규화 (Smiles -> SMILES)
            if 'Smiles' in test_df.columns:
                test_df = test_df.rename(columns={'Smiles': 'SMILES'})
            
            # 필요한 컬럼만 선택
            required_cols = ['ID', 'SMILES']
            available_cols = [col for col in required_cols if col in test_df.columns]
            
            if len(available_cols) == 2:
                test_df = test_df[available_cols]
                print(f"✅ 테스트 데이터 로드 완료: {len(test_df)}개 화합물")
                self.cache['test_data'] = test_df
                return test_df
            else:
                print(f"❌ 필요한 컬럼이 부족합니다. 사용 가능: {available_cols}")
                return None
        else:
            print("❌ 테스트 데이터 로드 실패")
            return None
    
    def load_training_data_from_chembl(self, use_cache=True):
        """
        ChEMBL 데이터에서 훈련 데이터 로드
        
        Args:
            use_cache (bool): 캐시된 데이터 사용 여부
            
        Returns:
            pd.DataFrame: 훈련 데이터 (SMILES, IC50_nM, pIC50)
        """
        if use_cache and 'training_data_chembl' in self.cache:
            print("📋 캐시된 ChEMBL 훈련 데이터 사용")
            return self.cache['training_data_chembl']
        
        print("�� ChEMBL 훈련 데이터 로딩 중...")
        training_df = get_clean_training_data()
        
        if training_df is not None:
            # 컬럼명 정규화
            if 'Smiles' in training_df.columns:
                training_df = training_df.rename(columns={'Smiles': 'SMILES'})
            
            # 필요한 컬럼만 선택
            required_cols = ['SMILES', 'IC50_nM', 'pIC50']
            available_cols = [col for col in required_cols if col in training_df.columns]
            
            if len(available_cols) >= 2:
                training_df = training_df[available_cols]
                print(f"✅ ChEMBL 훈련 데이터 로드 완료: {len(training_df)}개 화합물")
                self.cache['training_data_chembl'] = training_df
                return training_df
            else:
                print(f"❌ 필요한 컬럼이 부족합니다. 사용 가능: {available_cols}")
                return None
        else:
            print("❌ ChEMBL 훈련 데이터 로드 실패")
            return None
    
    def load_training_data_from_cas(self, use_cache=True):
        """
        CAS 엑셀 데이터에서 훈련 데이터 로드
        
        Args:
            use_cache (bool): 캐시된 데이터 사용 여부
            
        Returns:
            pd.DataFrame: 훈련 데이터 (SMILES, IC50_nM, pIC50)
        """
        if use_cache and 'training_data_cas' in self.cache:
            print("📋 캐시된 CAS 훈련 데이터 사용")
            return self.cache['training_data_cas']
        
        print("�� CAS 훈련 데이터 로딩 중...")
        
        try:
            # CAS 엑셀 파일 읽기
            cas_df = read_cas_excel()
            
            if cas_df is None:
                print("❌ CAS 엑셀 파일 읽기 실패")
                return None
            
            # SMILES와 IC50 관련 컬럼 찾기
            smiles_col = None
            ic50_col = None
            
            for col in cas_df.columns:
                col_lower = str(col).lower()
                if 'smiles' in col_lower:
                    smiles_col = col
                if 'ic50' in col_lower or 'activity' in col_lower:
                    ic50_col = col
            
            if smiles_col is None or ic50_col is None:
                print(f"❌ SMILES 또는 IC50 컬럼을 찾을 수 없습니다.")
                print(f"   사용 가능한 컬럼: {list(cas_df.columns)}")
                return None
            
            # 데이터 정제
            training_df = cas_df[[smiles_col, ic50_col]].copy()
            training_df = training_df.rename(columns={smiles_col: 'SMILES', ic50_col: 'IC50_raw'})
            
            # 결측치 제거
            training_df = training_df.dropna()
            
            # IC50 값을 수치형으로 변환
            training_df['IC50_nM'] = pd.to_numeric(training_df['IC50_raw'], errors='coerce')
            training_df = training_df.dropna(subset=['IC50_nM'])
            
            # pIC50 계산
            training_df['pIC50'] = -np.log10(training_df['IC50_nM'] * 1e-9)
            
            # 최종 컬럼만 선택
            final_df = training_df[['SMILES', 'IC50_nM', 'pIC50']].copy()
            
            print(f"✅ CAS 훈련 데이터 로드 완료: {len(final_df)}개 화합물")
            self.cache['training_data_cas'] = final_df
            return final_df
            
        except Exception as e:
            print(f"❌ CAS 데이터 로드 중 오류: {e}")
            return None
    
    def load_combined_training_data(self, use_cache=True):
        """
        모든 소스의 훈련 데이터를 결합하여 로드
        
        Args:
            use_cache (bool): 캐시된 데이터 사용 여부
            
        Returns:
            pd.DataFrame: 결합된 훈련 데이터
        """
        if use_cache and 'combined_training_data' in self.cache:
            print("📋 캐시된 결합 훈련 데이터 사용")
            return self.cache['combined_training_data']
        
        print("📋 결합 훈련 데이터 로딩 중...")
        
        # 각 소스에서 데이터 로드
        datasets = []
        
        # ChEMBL 데이터
        chembl_data = self.load_training_data_from_chembl(use_cache=False)
        if chembl_data is not None:
            chembl_data['source'] = 'ChEMBL'
            datasets.append(chembl_data)
            print(f"  - ChEMBL: {len(chembl_data)}개 화합물")
        
        # CAS 데이터
        cas_data = self.load_training_data_from_cas(use_cache=False)
        if cas_data is not None:
            cas_data['source'] = 'CAS'
            datasets.append(cas_data)
            print(f"  - CAS: {len(cas_data)}개 화합물")
        
        if not datasets:
            print("❌ 사용 가능한 훈련 데이터가 없습니다.")
            return None
        
        # 데이터 결합
        combined_df = pd.concat(datasets, ignore_index=True)
        
        # 중복 SMILES 제거 (가장 높은 pIC50 값 유지)
        combined_df = combined_df.sort_values('pIC50', ascending=False)
        combined_df = combined_df.drop_duplicates(subset=['SMILES'], keep='first')
        
        print(f"✅ 결합 훈련 데이터 로드 완료: {len(combined_df)}개 화합물")
        print(f"   - 중복 제거 후: {len(combined_df)}개 화합물")
        
        self.cache['combined_training_data'] = combined_df
        return combined_df
    
    def get_data_summary(self):
        """
        데이터 요약 정보 반환
        
        Returns:
            dict: 데이터 요약 정보
        """
        summary = {}
        
        # 테스트 데이터 요약
        test_data = self.load_test_data()
        if test_data is not None:
            summary['test'] = {
                'count': len(test_data),
                'columns': list(test_data.columns)
            }
        
        # 훈련 데이터 요약
        training_data = self.load_combined_training_data()
        if training_data is not None:
            summary['training'] = {
                'count': len(training_data),
                'columns': list(training_data.columns),
                'pIC50_stats': {
                    'min': training_data['pIC50'].min(),
                    'max': training_data['pIC50'].max(),
                    'mean': training_data['pIC50'].mean(),
                    'median': training_data['pIC50'].median()
                },
                'source_distribution': training_data['source'].value_counts().to_dict()
            }
        
        return summary
    
    def validate_smiles(self, df, smiles_col='SMILES'):
        """
        SMILES 데이터 유효성 검사
        
        Args:
            df (pd.DataFrame): 검사할 데이터프레임
            smiles_col (str): SMILES 컬럼명
            
        Returns:
            pd.DataFrame: 유효한 SMILES만 포함된 데이터프레임
        """
        if smiles_col not in df.columns:
            print(f"❌ {smiles_col} 컬럼이 없습니다.")
            return df
        
        original_count = len(df)
        
        # 빈 문자열 제거
        df = df[df[smiles_col].str.strip() != '']
        
        # None/NaN 값 제거
        df = df.dropna(subset=[smiles_col])
        
        # 최소 길이 검사 (SMILES는 보통 5자 이상)
        df = df[df[smiles_col].str.len() >= 5]
        
        # 기본적인 SMILES 패턴 검사 (원자 기호 포함)
        valid_atoms = df[smiles_col].str.contains(r'[COHNSFPClBrI]', regex=True)
        df = df[valid_atoms]
        
        # 분자 구조 기호 포함 검사 (괄호, 결합 기호, 숫자)
        valid_structure = df[smiles_col].str.contains(r'[()=#123456789]', regex=True)
        df = df[valid_structure]
        
        final_count = len(df)
        removed_count = original_count - final_count
        
        if removed_count > 0:
            print(f"⚠️ SMILES 검증: {removed_count}개 유효하지 않은 SMILES 제거")
            print(f"   - 원본: {original_count}개 → 검증 후: {final_count}개")
        
        return df

    def prepare_model_data(self, test_size=0.2, random_state=42):
        """
        모델 학습을 위한 데이터 준비
        
        Args:
            test_size (float): 테스트 세트 비율
            random_state (int): 랜덤 시드
            
        Returns:
            dict: 모델 학습용 데이터 딕셔너리
        """
        print(" 모델 학습용 데이터 준비 중...")
        
        # 훈련 데이터 로드
        training_data = self.load_combined_training_data()
        if training_data is None:
            print("❌ 훈련 데이터를 로드할 수 없습니다.")
            return None
        
        # SMILES 유효성 검사
        training_data = self.validate_smiles(training_data)
        
        # 테스트 데이터 로드
        test_data = self.load_test_data()
        if test_data is None:
            print("❌ 테스트 데이터를 로드할 수 없습니다.")
            return None
        
        # SMILES 유효성 검사
        test_data = self.validate_smiles(test_data)
        
        # 훈련/검증 데이터 분할
        from sklearn.model_selection import train_test_split
        
        train_df, val_df = train_test_split(
            training_data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=None  # pIC50 값이 연속형이므로 stratify 사용 불가
        )
        
        # 결과 데이터 구성
        model_data = {
            'train': train_df,
            'validation': val_df,
            'test': test_data,
            'full_training': training_data
        }
        
        print(f"✅ 모델 데이터 준비 완료:")
        print(f"   - 훈련 세트: {len(train_df)}개 화합물")
        print(f"   - 검증 세트: {len(val_df)}개 화합물")
        print(f"   - 테스트 세트: {len(test_data)}개 화합물")
        print(f"   - 전체 훈련 데이터: {len(training_data)}개 화합물")
        
        return model_data

def create_data_loader(data_dir='data'):
    """
    데이터 로더 인스턴스 생성
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
        
    Returns:
        DrugDiscoveryDataLoader: 데이터 로더 인스턴스
    """
    return DrugDiscoveryDataLoader(data_dir)

# 사용 예시
if __name__ == "__main__":
    # 데이터 로더 생성
    loader = create_data_loader()
    
    # 데이터 요약 출력
    summary = loader.get_data_summary()
    print("\n📊 데이터 요약:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 모델 데이터 준비
    model_data = loader.prepare_model_data()
    
    if model_data:
        print("\n✅ 모델 데이터 준비 완료!")
        print("사용 가능한 데이터:")
        for key, df in model_data.items():
            print(f"  - {key}: {len(df)}개 화합물")
    else:
        print("\n❌ 모델 데이터 준비 실패")