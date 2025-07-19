"""
데이터 로딩 사용 예시

이 파일은 src/data_loader.py의 사용법을 보여줍니다.
"""

import sys
import os

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import create_data_loader

def main():
    """메인 함수 - 데이터 로딩 예시"""
    
    print("🚀 AI 신약개발 데이터 로딩 예시")
    print("=" * 50)
    
    # 1. 데이터 로더 생성
    print("\n1️⃣ 데이터 로더 생성")
    loader = create_data_loader()
    
    # 2. 데이터 요약 확인
    print("\n2️⃣ 데이터 요약")
    summary = loader.get_data_summary()
    
    for key, value in summary.items():
        print(f"\n📊 {key.upper()} 데이터:")
        if key == 'test':
            print(f"   - 화합물 수: {value['count']}")
            print(f"   - 컬럼: {value['columns']}")
        elif key == 'training':
            print(f"   - 화합물 수: {value['count']}")
            print(f"   - 컬럼: {value['columns']}")
            print(f"   - pIC50 통계:")
            stats = value['pIC50_stats']
            print(f"     * 최소값: {stats['min']:.2f}")
            print(f"     * 최대값: {stats['max']:.2f}")
            print(f"     * 평균값: {stats['mean']:.2f}")
            print(f"     * 중앙값: {stats['median']:.2f}")
            print(f"   - 데이터 소스 분포:")
            for source, count in value['source_distribution'].items():
                print(f"     * {source}: {count}개")
    
    # 3. 모델 데이터 준비
    print("\n3️⃣ 모델 데이터 준비")
    model_data = loader.prepare_model_data(test_size=0.2, random_state=42)
    
    if model_data:
        print("\n✅ 모델 데이터 준비 완료!")
        
        # 각 데이터셋의 샘플 출력
        for key, df in model_data.items():
            print(f"\n📋 {key.upper()} 데이터 샘플:")
            print(df.head(3))
            
            if 'pIC50' in df.columns:
                print(f"   pIC50 범위: {df['pIC50'].min():.2f} ~ {df['pIC50'].max():.2f}")
    
    # 4. 개별 데이터 로딩 예시
    print("\n4️⃣ 개별 데이터 로딩 예시")
    
    # 테스트 데이터만 로드
    test_data = loader.load_test_data()
    if test_data is not None:
        print(f"✅ 테스트 데이터: {len(test_data)}개 화합물")
    
    # ChEMBL 훈련 데이터만 로드
    chembl_data = loader.load_training_data_from_chembl()
    if chembl_data is not None:
        print(f"✅ ChEMBL 훈련 데이터: {len(chembl_data)}개 화합물")
    
    # CAS 훈련 데이터만 로드
    cas_data = loader.load_training_data_from_cas()
    if cas_data is not None:
        print(f"✅ CAS 훈련 데이터: {len(cas_data)}개 화합물")
    
    print("\n🎉 데이터 로딩 예시 완료!")

if __name__ == "__main__":
    main()
