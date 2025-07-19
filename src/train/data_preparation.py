"""
MAP3K5 IC50 활성값 예측 모델을 위한 데이터 준비 모듈

이 모듈은 구현된 data_loader.py를 사용하여 모델 훈련에 필요한 데이터를 준비합니다.
CAS, ChEMBL, PubChem 등 모든 소스의 데이터를 효율적으로 활용합니다.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader import DrugDiscoveryDataLoader, create_data_loader

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MAP3K5DataPreparation:
    """
    MAP3K5 IC50 예측 모델을 위한 데이터 준비 클래스
    
    이 클래스는 다음 기능을 제공합니다:
    1. 다양한 소스에서 훈련 데이터 로드 (ChEMBL, CAS, PubChem)
    2. 테스트 데이터 로드
    3. 데이터 품질 검증 및 전처리
    4. 모델 훈련을 위한 데이터셋 분할
    5. 데이터 통합 및 중복 제거
    """
    
    def __init__(self, data_dir: str = 'data', random_state: int = 42):
        """
        데이터 준비 클래스 초기화
        
        Args:
            data_dir (str): 데이터 디렉토리 경로
            random_state (int): 재현성을 위한 랜덤 시드
        """
        self.data_dir = data_dir
        self.random_state = random_state
        self.data_loader = create_data_loader(data_dir)
        
        # 데이터 저장소
        self.training_data = None
        self.test_data = None
        self.validation_data = None
        
        logger.info(f"MAP3K5 데이터 준비 클래스 초기화 완료 (데이터 디렉토리: {data_dir})")
    
    def load_cas_data_directly(self) -> Optional[pd.DataFrame]:
        """
        CAS 데이터를 직접 로드하여 처리
        
        Returns:
            pd.DataFrame: 처리된 CAS 데이터
        """
        logger.info("CAS 데이터 직접 로딩 시작...")
        
        try:
            # CAS 엑셀 파일의 IC50 시트 직접 읽기
            from data.preprocessing.read_excel import read_ic50_sheet
            
            cas_df = read_ic50_sheet()
            
            if cas_df is None:
                logger.warning("CAS 엑셀 파일 읽기 실패")
                return None
            
            # 필요한 컬럼 확인
            required_cols = ['SMILES', 'Single Value (Parsed)', 'pX Value']
            available_cols = [col for col in required_cols if col in cas_df.columns]
            
            if len(available_cols) < 2:
                logger.warning(f"CAS 데이터에 필요한 컬럼이 부족합니다. 사용 가능: {available_cols}")
                return None
            
            # 데이터 정제
            training_df = cas_df[available_cols].copy()
            
            # SMILES 결측치 제거
            training_df = training_df.dropna(subset=['SMILES'])
            
            # IC50 값 처리
            if 'Single Value (Parsed)' in training_df.columns:
                # Single Value (Parsed)를 IC50_nM으로 변환
                training_df['IC50_nM'] = pd.to_numeric(training_df['Single Value (Parsed)'], errors='coerce')
                training_df = training_df.dropna(subset=['IC50_nM'])
                
                # pIC50 계산 (Single Value가 이미 nM 단위라고 가정)
                training_df['pIC50'] = -np.log10(training_df['IC50_nM'] * 1e-9)
            
            elif 'pX Value' in training_df.columns:
                # pX Value가 이미 pIC50 값인 경우
                training_df['pIC50'] = pd.to_numeric(training_df['pX Value'], errors='coerce')
                training_df = training_df.dropna(subset=['pIC50'])
                
                # IC50_nM 계산
                training_df['IC50_nM'] = 10**(-training_df['pIC50']) * 1e9
            
            # 최종 컬럼만 선택
            final_df = training_df[['SMILES', 'IC50_nM', 'pIC50']].copy()
            
            # 중복 SMILES 제거 (가장 높은 pIC50 값 유지)
            final_df = final_df.sort_values('pIC50', ascending=False)
            final_df = final_df.drop_duplicates(subset=['SMILES'], keep='first')
            
            logger.info(f"✅ CAS 데이터 로드 완료: {len(final_df)}개 화합물")
            logger.info(f"   - pIC50 범위: {final_df['pIC50'].min():.2f} ~ {final_df['pIC50'].max():.2f}")
            logger.info(f"   - IC50 범위: {final_df['IC50_nM'].min():.2e} ~ {final_df['IC50_nM'].max():.2e} nM")
            
            return final_df
            
        except Exception as e:
            logger.error(f"CAS 데이터 로드 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_pubchem_data(self) -> Optional[pd.DataFrame]:
        """
        PubChem 데이터 로드
        
        Returns:
            pd.DataFrame: 처리된 PubChem 데이터
        """
        logger.info("PubChem 데이터 로딩 시작...")
        
        try:
            # PubChem CSV 파일 읽기
            pubchem_file = os.path.join(self.data_dir, 'Pubchem_ASK1.csv')
            
            if not os.path.exists(pubchem_file):
                logger.warning(f"PubChem 파일이 없습니다: {pubchem_file}")
                return None
            
            # CSV 파일 읽기 (low_memory=False로 설정하여 경고 제거)
            df = pd.read_csv(pubchem_file, low_memory=False)
            
            logger.info(f"📊 PubChem 데이터 크기: {df.shape}")
            logger.info(f"📋 컬럼: {list(df.columns)}")
            
            # IC50 타입의 데이터만 필터링
            if 'Activity_Type' in df.columns:
                ic50_data = df[df['Activity_Type'] == 'IC50'].copy()
                logger.info(f"📊 IC50 타입 데이터: {len(ic50_data)}개")
            else:
                ic50_data = df.copy()
                logger.info(f"📊 Activity_Type 컬럼이 없어 전체 데이터 사용: {len(ic50_data)}개")
            
            # 필요한 컬럼 확인
            required_cols = ['SMILES', 'Activity_Value']
            available_cols = [col for col in required_cols if col in ic50_data.columns]
            
            if len(available_cols) < 2:
                logger.warning(f"PubChem 데이터에 필요한 컬럼이 부족합니다. 사용 가능: {available_cols}")
                logger.warning(f"전체 컬럼: {list(ic50_data.columns)}")
                return None
            
            # 데이터 정제
            training_df = ic50_data[available_cols].copy()
            training_df = training_df.rename(columns={'Activity_Value': 'IC50_raw'})
            
            # 결측치 제거
            training_df = training_df.dropna()
            
            # IC50 값을 수치형으로 변환
            training_df['IC50_nM'] = pd.to_numeric(training_df['IC50_raw'], errors='coerce')
            training_df = training_df.dropna(subset=['IC50_nM'])
            
            # IC50 값이 양수인지 확인
            training_df = training_df[training_df['IC50_nM'] > 0]
            
            # pIC50 계산 (Activity_Value가 이미 μM 단위라고 가정)
            training_df['pIC50'] = -np.log10(training_df['IC50_nM'] * 1e-6)
            
            # 최종 컬럼만 선택
            final_df = training_df[['SMILES', 'IC50_nM', 'pIC50']].copy()
            
            # 중복 SMILES 제거 (가장 높은 pIC50 값 유지)
            final_df = final_df.sort_values('pIC50', ascending=False)
            final_df = final_df.drop_duplicates(subset=['SMILES'], keep='first')
            
            logger.info(f"✅ PubChem 데이터 로드 완료: {len(final_df)}개 화합물")
            logger.info(f"   - pIC50 범위: {final_df['pIC50'].min():.2f} ~ {final_df['pIC50'].max():.2f}")
            logger.info(f"   - IC50 범위: {final_df['IC50_nM'].min():.2e} ~ {final_df['IC50_nM'].max():.2e} μM")
            
            return final_df
            
        except Exception as e:
            logger.error(f"PubChem 데이터 로드 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        모든 데이터를 로드하고 반환
        
        Returns:
            Dict[str, pd.DataFrame]: 로드된 데이터 딕셔너리
        """
        logger.info("모든 데이터 로딩 시작...")
        
        data_dict = {}
        
        # 1. 테스트 데이터 로드
        logger.info("테스트 데이터 로딩 중...")
        test_data = self.data_loader.load_test_data()
        if test_data is not None:
            data_dict['test'] = test_data
            self.test_data = test_data
            logger.info(f"✅ 테스트 데이터 로드 완료: {len(test_data)}개 화합물")
        else:
            logger.warning("❌ 테스트 데이터 로드 실패")
        
        # 2. ChEMBL 데이터 로드
        logger.info("ChEMBL 데이터 로딩 중...")
        chembl_data = self.data_loader.load_training_data_from_chembl()
        if chembl_data is not None:
            data_dict['chembl'] = chembl_data
            logger.info(f"✅ ChEMBL 데이터: {len(chembl_data)}개 화합물")
        else:
            logger.warning("❌ ChEMBL 데이터 로드 실패")
        
        # 3. CAS 데이터 직접 로드
        logger.info("CAS 데이터 직접 로딩 중...")
        cas_data = self.load_cas_data_directly()
        if cas_data is not None:
            data_dict['cas'] = cas_data
            logger.info(f"✅ CAS 데이터: {len(cas_data)}개 화합물")
        else:
            logger.warning("❌ CAS 데이터 로드 실패")
        
        # 4. PubChem 데이터 로드
        logger.info("PubChem 데이터 로딩 중...")
        pubchem_data = self.load_pubchem_data()
        if pubchem_data is not None:
            data_dict['pubchem'] = pubchem_data
            logger.info(f"✅ PubChem 데이터: {len(pubchem_data)}개 화합물")
        else:
            logger.warning("❌ PubChem 데이터 로드 실패")
        
        # 5. 결합된 훈련 데이터 생성
        logger.info("결합된 훈련 데이터 생성 중...")
        combined_data = self.combine_all_training_data(data_dict)
        if combined_data is not None:
            data_dict['combined'] = combined_data
            self.training_data = combined_data
            logger.info(f"✅ 결합된 훈련 데이터: {len(combined_data)}개 화합물")
        
        return data_dict
    
    def combine_all_training_data(self, data_dict: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        모든 훈련 데이터를 결합
        
        Args:
            data_dict: 데이터 딕셔너리
            
        Returns:
            pd.DataFrame: 결합된 훈련 데이터
        """
        datasets = []
        
        # 각 소스의 데이터를 수집
        for source, data in data_dict.items():
            if source in ['chembl', 'cas', 'pubchem'] and data is not None:
                # 소스 정보 추가
                data_with_source = data.copy()
                data_with_source['source'] = source
                datasets.append(data_with_source)
                logger.info(f"  - {source.upper()}: {len(data)}개 화합물")
        
        if not datasets:
            logger.error("결합할 훈련 데이터가 없습니다.")
            return None
        
        # 데이터 결합
        combined_df = pd.concat(datasets, ignore_index=True)
        
        # 중복 SMILES 제거 (가장 높은 pIC50 값 유지)
        original_count = len(combined_df)
        combined_df = combined_df.sort_values('pIC50', ascending=False)
        combined_df = combined_df.drop_duplicates(subset=['SMILES'], keep='first')
        final_count = len(combined_df)
        
        logger.info(f"📊 데이터 결합 결과:")
        logger.info(f"   - 원본 총 화합물: {original_count}개")
        logger.info(f"   - 중복 제거 후: {final_count}개")
        logger.info(f"   - 제거된 중복: {original_count - final_count}개")
        
        # 소스별 분포
        source_dist = combined_df['source'].value_counts()
        logger.info(f"   - 소스별 분포:")
        for source, count in source_dist.items():
            logger.info(f"     {source.upper()}: {count}개")
        
        return combined_df
    
    def validate_data_quality(self, data: pd.DataFrame, data_type: str = "training") -> Dict[str, any]:
        """
        데이터 품질 검증
        
        Args:
            data (pd.DataFrame): 검증할 데이터
            data_type (str): 데이터 타입 ("training", "test")
            
        Returns:
            Dict[str, any]: 검증 결과
        """
        logger.info(f"{data_type} 데이터 품질 검증 시작...")
        
        validation_results = {
            'total_count': len(data),
            'missing_values': {},
            'duplicates': 0,
            'smiles_validation': {},
            'ic50_stats': {}
        }
        
        # 1. 결측치 검사
        missing_counts = data.isnull().sum()
        validation_results['missing_values'] = missing_counts.to_dict()
        
        # 2. 중복 검사
        if 'SMILES' in data.columns:
            duplicates = data.duplicated(subset=['SMILES']).sum()
            validation_results['duplicates'] = duplicates
        
        # 3. SMILES 유효성 검사
        if 'SMILES' in data.columns:
            # SMILES 길이 분포
            smiles_lengths = data['SMILES'].str.len()
            validation_results['smiles_validation'] = {
                'min_length': smiles_lengths.min(),
                'max_length': smiles_lengths.max(),
                'mean_length': smiles_lengths.mean(),
                'empty_smiles': (data['SMILES'] == '').sum(),
                'whitespace_only': (data['SMILES'].str.strip() == '').sum()
            }
        
        # 4. IC50 통계 (훈련 데이터인 경우)
        if data_type == "training" and 'pIC50' in data.columns:
            validation_results['ic50_stats'] = {
                'min': data['pIC50'].min(),
                'max': data['pIC50'].max(),
                'mean': data['pIC50'].mean(),
                'median': data['pIC50'].median(),
                'std': data['pIC50'].std()
            }
        
        # 검증 결과 출력
        logger.info(f"📊 {data_type} 데이터 품질 검증 결과:")
        logger.info(f"   - 총 화합물 수: {validation_results['total_count']}")
        logger.info(f"   - 중복 화합물: {validation_results['duplicates']}")
        
        if validation_results['missing_values']:
            logger.info(f"   - 결측치: {validation_results['missing_values']}")
        
        if validation_results['smiles_validation']:
            smiles_info = validation_results['smiles_validation']
            logger.info(f"   - SMILES 길이: {smiles_info['min_length']}~{smiles_info['max_length']} (평균: {smiles_info['mean_length']:.1f})")
        
        if validation_results['ic50_stats']:
            ic50_info = validation_results['ic50_stats']
            logger.info(f"   - pIC50 범위: {ic50_info['min']:.2f}~{ic50_info['max']:.2f} (평균: {ic50_info['mean']:.2f})")
        
        return validation_results
    
    def prepare_model_datasets(self, test_size: float = 0.2, validation_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        """
        모델 훈련을 위한 데이터셋 준비
        
        Args:
            test_size (float): 테스트 세트 비율
            validation_size (float): 검증 세트 비율
            
        Returns:
            Dict[str, pd.DataFrame]: 훈련/검증/테스트 데이터셋
        """
        logger.info("모델 훈련용 데이터셋 준비 시작...")
        
        # 데이터 로드
        if self.training_data is None:
            all_data = self.load_all_data()
            if 'combined' in all_data:
                self.training_data = all_data['combined']
        
        if self.test_data is None:
            self.test_data = self.data_loader.load_test_data()
        
        if self.training_data is None:
            logger.error("훈련 데이터를 로드할 수 없습니다.")
            return {}
        
        # 데이터 품질 검증
        training_validation = self.validate_data_quality(self.training_data, "training")
        
        # 모델 데이터 준비 (data_loader의 prepare_model_data 사용)
        model_data = self.data_loader.prepare_model_data(
            test_size=validation_size,  # validation_size를 test_size로 사용
            random_state=self.random_state
        )
        
        if model_data is None:
            logger.error("모델 데이터 준비에 실패했습니다.")
            return {}
        
        # 결과 정리
        datasets = {
            'train': model_data['train'],
            'validation': model_data['validation'],
            'test': self.test_data,  # 실제 테스트 데이터
            'full_training': model_data['full_training']
        }
        
        # 데이터셋 정보 출력
        logger.info("📊 모델 데이터셋 준비 완료:")
        logger.info(f"   - 훈련 세트: {len(datasets['train'])}개 화합물")
        logger.info(f"   - 검증 세트: {len(datasets['validation'])}개 화합물")
        logger.info(f"   - 테스트 세트: {len(datasets['test'])}개 화합물")
        logger.info(f"   - 전체 훈련: {len(datasets['full_training'])}개 화합물")
        
        return datasets
    
    def get_data_summary(self) -> Dict[str, any]:
        """
        전체 데이터 요약 정보 반환
        
        Returns:
            Dict[str, any]: 데이터 요약 정보
        """
        logger.info("데이터 요약 정보 생성 중...")
        
        summary = {}
        
        # 테스트 데이터 요약
        if self.test_data is not None:
            summary['test'] = {
                'count': len(self.test_data),
                'columns': list(self.test_data.columns)
            }
        
        # 훈련 데이터 요약
        if self.training_data is not None:
            summary['training'] = {
                'count': len(self.training_data),
                'columns': list(self.training_data.columns),
                'pIC50_stats': {
                    'min': self.training_data['pIC50'].min(),
                    'max': self.training_data['pIC50'].max(),
                    'mean': self.training_data['pIC50'].mean(),
                    'median': self.training_data['pIC50'].median()
                },
                'source_distribution': self.training_data['source'].value_counts().to_dict()
            }
        
        # 추가 정보
        if self.training_data is not None:
            summary['training_quality'] = self.validate_data_quality(self.training_data, "training")
        
        if self.test_data is not None:
            summary['test_quality'] = self.validate_data_quality(self.test_data, "test")
        
        return summary
    
    def save_prepared_data(self, output_dir: str = "prepared_data") -> None:
        """
        준비된 데이터를 파일로 저장
        
        Args:
            output_dir (str): 출력 디렉토리
        """
        logger.info(f"준비된 데이터를 {output_dir}에 저장 중...")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 데이터셋 준비
        datasets = self.prepare_model_datasets()
        
        # 각 데이터셋 저장
        for name, data in datasets.items():
            if data is not None and len(data) > 0:
                output_path = os.path.join(output_dir, f"{name}_data.csv")
                data.to_csv(output_path, index=False)
                logger.info(f"✅ {name} 데이터 저장 완료: {output_path} ({len(data)}개 화합물)")
        
        # 데이터 요약 저장
        summary = self.get_data_summary()
        summary_path = os.path.join(output_dir, "data_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("MAP3K5 IC50 예측 모델 데이터 요약\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in summary.items():
                f.write(f"{key}:\n")
                f.write(str(value))
                f.write("\n\n")
        
        logger.info(f"✅ 데이터 요약 저장 완료: {summary_path}")


def main():
    """메인 실행 함수"""
    logger.info("MAP3K5 데이터 준비 프로세스 시작")
    
    # 데이터 준비 클래스 초기화
    data_prep = MAP3K5DataPreparation()
    
    # 모든 데이터 로드
    all_data = data_prep.load_all_data()
    
    # 데이터 품질 검증
    for data_name, data in all_data.items():
        if data is not None:
            data_prep.validate_data_quality(data, data_name)
    
    # 모델 데이터셋 준비
    model_datasets = data_prep.prepare_model_datasets()
    
    # 데이터 저장
    data_prep.save_prepared_data()
    
    # 최종 요약 출력
    summary = data_prep.get_data_summary()
    logger.info("🎉 MAP3K5 데이터 준비 프로세스 완료!")
    
    return model_datasets


if __name__ == "__main__":
    main() 