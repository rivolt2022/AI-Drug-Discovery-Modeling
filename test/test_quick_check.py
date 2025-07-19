"""
빠른 데이터 확인 스크립트

이 스크립트는 데이터 로딩이 제대로 작동하는지 빠르게 확인합니다.
"""

import sys
import os

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import create_data_loader

def quick_check():
    """빠른 데이터 확인"""
    print("🔍 빠른 데이터 확인")
    print("=" * 30)
    
    try:
        # 데이터 로더 생성
        print("1️⃣ 데이터 로더 생성...")
        loader = create_data_loader()
        print("✅ 데이터 로더 생성 완료")
        
        # 테스트 데이터 확인
        print("\n2️⃣ 테스트 데이터 확인...")
        test_data = loader.load_test_data()
        if test_data is not None:
            print(f"✅ 테스트 데이터: {len(test_data)}개 화합물")
            print(f"   - 컬럼: {list(test_data.columns)}")
            print(f"   - 샘플 SMILES: {test_data['SMILES'].iloc[0]}")
        else:
            print("❌ 테스트 데이터 로드 실패")
            return False
        
        # 훈련 데이터 확인
        print("\n3️⃣ 훈련 데이터 확인...")
        training_data = loader.load_combined_training_data()
        if training_data is not None:
            print(f"✅ 훈련 데이터: {len(training_data)}개 화합물")
            print(f"   - 컬럼: {list(training_data.columns)}")
            print(f"   - pIC50 범위: {training_data['pIC50'].min():.2f} ~ {training_data['pIC50'].max():.2f}")
            print(f"   - 데이터 소스: {list(training_data['source'].unique())}")
        else:
            print("❌ 훈련 데이터 로드 실패")
            return False
        
        # 모델 데이터 준비 확인
        print("\n4️⃣ 모델 데이터 준비 확인...")
        model_data = loader.prepare_model_data()
        if model_data is not None:
            print(f"✅ 모델 데이터 준비 완료")
            print(f"   - 훈련 세트: {len(model_data['train'])}개")
            print(f"   - 검증 세트: {len(model_data['validation'])}개")
            print(f"   - 테스트 세트: {len(model_data['test'])}개")
        else:
            print("❌ 모델 데이터 준비 실패")
            return False
        
        print("\n 모든 데이터 로딩이 정상적으로 작동합니다!")
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_check()
    sys.exit(0 if success else 1)
