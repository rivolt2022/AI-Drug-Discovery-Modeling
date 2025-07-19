"""
데이터 로딩 기능 테스트 스크립트

이 스크립트는 src/data_loader.py의 기능들을 테스트합니다.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DrugDiscoveryDataLoader, create_data_loader

class TestDataLoading(unittest.TestCase):
    """데이터 로딩 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.loader = create_data_loader()
        self.test_data = None
        self.training_data = None
    
    def test_loader_creation(self):
        """데이터 로더 생성 테스트"""
        print("🧪 데이터 로더 생성 테스트")
        
        self.assertIsNotNone(self.loader)
        self.assertIsInstance(self.loader, DrugDiscoveryDataLoader)
        self.assertEqual(self.loader.data_dir, 'data')
        
        print("✅ 데이터 로더 생성 테스트 통과")
    
    def test_test_data_loading(self):
        """테스트 데이터 로딩 테스트"""
        print("🧪 테스트 데이터 로딩 테스트")
        
        test_data = self.loader.load_test_data()
        
        # 기본 검증
        self.assertIsNotNone(test_data)
        self.assertIsInstance(test_data, pd.DataFrame)
        self.assertGreater(len(test_data), 0)
        
        # 컬럼 검증
        required_cols = ['ID', 'SMILES']
        for col in required_cols:
            self.assertIn(col, test_data.columns)
        
        # 데이터 타입 검증
        self.assertIsInstance(test_data['ID'].iloc[0], str)
        self.assertIsInstance(test_data['SMILES'].iloc[0], str)
        
        # SMILES 유효성 검증
        self.assertGreater(test_data['SMILES'].iloc[0].__len__(), 5)
        
        self.test_data = test_data
        print(f"✅ 테스트 데이터 로딩 테스트 통과: {len(test_data)}개 화합물")
    
    def test_chembl_training_data_loading(self):
        """ChEMBL 훈련 데이터 로딩 테스트"""
        print("�� ChEMBL 훈련 데이터 로딩 테스트")
        
        training_data = self.loader.load_training_data_from_chembl()
        
        if training_data is not None:
            # 기본 검증
            self.assertIsInstance(training_data, pd.DataFrame)
            self.assertGreater(len(training_data), 0)
            
            # 컬럼 검증
            required_cols = ['SMILES', 'IC50_nM', 'pIC50']
            for col in required_cols:
                self.assertIn(col, training_data.columns)
            
            # 데이터 타입 검증
            self.assertIsInstance(training_data['SMILES'].iloc[0], str)
            self.assertIsInstance(training_data['IC50_nM'].iloc[0], (int, float, np.number))
            self.assertIsInstance(training_data['pIC50'].iloc[0], (int, float, np.number))
            
            # 값 범위 검증
            self.assertGreater(training_data['IC50_nM'].min(), 0)
            self.assertGreater(training_data['pIC50'].min(), 0)
            
            self.training_data = training_data
            print(f"✅ ChEMBL 훈련 데이터 로딩 테스트 통과: {len(training_data)}개 화합물")
        else:
            print("⚠️ ChEMBL 데이터가 없어 테스트를 건너뜁니다.")
    
    def test_cas_training_data_loading(self):
        """CAS 훈련 데이터 로딩 테스트"""
        print("🧪 CAS 훈련 데이터 로딩 테스트")
        
        training_data = self.loader.load_training_data_from_cas()
        
        if training_data is not None:
            # 기본 검증
            self.assertIsInstance(training_data, pd.DataFrame)
            self.assertGreater(len(training_data), 0)
            
            # 컬럼 검증
            required_cols = ['SMILES', 'IC50_nM', 'pIC50']
            for col in required_cols:
                self.assertIn(col, training_data.columns)
            
            # 데이터 타입 검증
            self.assertIsInstance(training_data['SMILES'].iloc[0], str)
            self.assertIsInstance(training_data['IC50_nM'].iloc[0], (int, float, np.number))
            self.assertIsInstance(training_data['pIC50'].iloc[0], (int, float, np.number))
            
            self.training_data = training_data
            print(f"✅ CAS 훈련 데이터 로딩 테스트 통과: {len(training_data)}개 화합물")
        else:
            print("⚠️ CAS 데이터가 없어 테스트를 건너뜁니다.")
    
    def test_combined_training_data_loading(self):
        """결합 훈련 데이터 로딩 테스트"""
        print("🧪 결합 훈련 데이터 로딩 테스트")
        
        training_data = self.loader.load_combined_training_data()
        
        if training_data is not None:
            # 기본 검증
            self.assertIsInstance(training_data, pd.DataFrame)
            self.assertGreater(len(training_data), 0)
            
            # 컬럼 검증
            required_cols = ['SMILES', 'IC50_nM', 'pIC50', 'source']
            for col in required_cols:
                self.assertIn(col, training_data.columns)
            
            # 소스 분포 검증
            sources = training_data['source'].unique()
            self.assertGreater(len(sources), 0)
            
            # 중복 제거 검증
            unique_smiles = training_data['SMILES'].nunique()
            total_smiles = len(training_data)
            self.assertEqual(unique_smiles, total_smiles)  # 중복이 제거되어야 함
            
            print(f"✅ 결합 훈련 데이터 로딩 테스트 통과: {len(training_data)}개 화합물")
            print(f"   - 데이터 소스: {list(sources)}")
        else:
            print("⚠️ 결합 훈련 데이터가 없어 테스트를 건너뜁니다.")
    
    def test_smiles_validation(self):
        """SMILES 유효성 검사 테스트"""
        print("🧪 SMILES 유효성 검사 테스트")
        
        # 테스트용 데이터 생성
        test_df = pd.DataFrame({
            'SMILES': [
                'CC(=O)OC1=CC=CC=C1C(=O)O',  # 유효한 SMILES
                '',  # 빈 문자열
                '   ',  # 공백만
                'CC',  # 너무 짧음
                'INVALID_SMILES',  # 유효하지 않은 패턴
                'CC(=O)OC1=CC=CC=C1C(=O)O'  # 유효한 SMILES
            ],
            'pIC50': [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        })
        
        # 유효성 검사 실행
        validated_df = self.loader.validate_smiles(test_df)
        
        # 검증 결과 확인
        self.assertIsInstance(validated_df, pd.DataFrame)
        self.assertLess(len(validated_df), len(test_df))  # 일부 데이터가 제거되어야 함
        
        # 유효한 SMILES만 남아있어야 함
        for smiles in validated_df['SMILES']:
            self.assertGreater(len(smiles), 5)
            # 더 구체적인 SMILES 패턴 검사
            self.assertTrue(
                any(char in smiles for char in ['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']),
                f"유효한 원자 기호가 없는 SMILES: {smiles}"
            )
            # 괄호나 숫자가 포함되어야 함 (분자 구조를 나타내므로)
            self.assertTrue(
                any(char in smiles for char in ['(', ')', '=', '#', '1', '2', '3', '4', '5', '6', '7', '8', '9']),
                f"분자 구조 기호가 없는 SMILES: {smiles}"
            )
        
        print(f"✅ SMILES 유효성 검사 테스트 통과: {len(validated_df)}개 유효한 SMILES")
    
    def test_model_data_preparation(self):
        """모델 데이터 준비 테스트"""
        print("🧪 모델 데이터 준비 테스트")
        
        model_data = self.loader.prepare_model_data(test_size=0.2, random_state=42)
        
        if model_data is not None:
            # 기본 구조 검증
            required_keys = ['train', 'validation', 'test', 'full_training']
            for key in required_keys:
                self.assertIn(key, model_data)
                self.assertIsInstance(model_data[key], pd.DataFrame)
            
            # 데이터 크기 검증
            train_size = len(model_data['train'])
            val_size = len(model_data['validation'])
            test_size = len(model_data['test'])
            full_size = len(model_data['full_training'])
            
            self.assertGreater(train_size, 0)
            self.assertGreater(val_size, 0)
            self.assertGreater(test_size, 0)
            self.assertGreater(full_size, 0)
            
            # 훈련/검증 분할 비율 검증 (대략적)
            split_ratio = val_size / (train_size + val_size)
            self.assertAlmostEqual(split_ratio, 0.2, delta=0.1)
            
            # 컬럼 검증
            for key in ['train', 'validation', 'full_training']:
                required_cols = ['SMILES', 'pIC50']
                for col in required_cols:
                    self.assertIn(col, model_data[key].columns)
            
            # 테스트 데이터 컬럼 검증
            self.assertIn('SMILES', model_data['test'].columns)
            
            print(f"✅ 모델 데이터 준비 테스트 통과:")
            print(f"   - 훈련 세트: {train_size}개")
            print(f"   - 검증 세트: {val_size}개")
            print(f"   - 테스트 세트: {test_size}개")
            print(f"   - 전체 훈련: {full_size}개")
        else:
            print("⚠️ 모델 데이터 준비에 실패했습니다.")
    
    def test_data_summary(self):
        """데이터 요약 테스트"""
        print("🧪 데이터 요약 테스트")
        
        summary = self.loader.get_data_summary()
        
        # 기본 구조 검증
        self.assertIsInstance(summary, dict)
        
        # 테스트 데이터 요약 검증
        if 'test' in summary:
            test_summary = summary['test']
            self.assertIn('count', test_summary)
            self.assertIn('columns', test_summary)
            self.assertGreater(test_summary['count'], 0)
        
        # 훈련 데이터 요약 검증
        if 'training' in summary:
            training_summary = summary['training']
            self.assertIn('count', training_summary)
            self.assertIn('columns', training_summary)
            self.assertIn('pIC50_stats', training_summary)
            self.assertIn('source_distribution', training_summary)
            
            if training_summary['count'] > 0:
                stats = training_summary['pIC50_stats']
                self.assertIn('min', stats)
                self.assertIn('max', stats)
                self.assertIn('mean', stats)
                self.assertIn('median', stats)
        
        print("✅ 데이터 요약 테스트 통과")
    
    def test_cache_functionality(self):
        """캐시 기능 테스트"""
        print("🧪 캐시 기능 테스트")
        
        # 첫 번째 로드
        test_data_1 = self.loader.load_test_data(use_cache=True)
        
        # 두 번째 로드 (캐시 사용)
        test_data_2 = self.loader.load_test_data(use_cache=True)
        
        # 캐시 확인
        self.assertIn('test_data', self.loader.cache)
        self.assertIs(test_data_1, test_data_2)  # 같은 객체여야 함
        
        # 캐시 비활성화
        test_data_3 = self.loader.load_test_data(use_cache=False)
        self.assertIsNot(test_data_1, test_data_3)  # 다른 객체여야 함
        
        print("✅ 캐시 기능 테스트 통과")

def run_tests():
    """테스트 실행"""
    print("🚀 데이터 로딩 테스트 시작")
    print("=" * 50)
    
    # 테스트 스위트 생성
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestDataLoading)
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)
    print(f"실행된 테스트: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"오류: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ 실패한 테스트:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ 오류가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 