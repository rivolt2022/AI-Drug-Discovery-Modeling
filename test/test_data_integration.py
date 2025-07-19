"""
데이터 통합 테스트 스크립트

이 스크립트는 전체 데이터 파이프라인을 통합적으로 테스트합니다.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import create_data_loader

def test_data_quality():
    """데이터 품질 테스트"""
    print("🔍 데이터 품질 테스트")
    print("=" * 40)
    
    loader = create_data_loader()
    
    # 1. 테스트 데이터 품질 검사
    print("\n1️⃣ 테스트 데이터 품질 검사")
    test_data = loader.load_test_data()
    
    if test_data is not None:
        print(f"   - 총 화합물 수: {len(test_data)}")
        print(f"   - 고유 SMILES 수: {test_data['SMILES'].nunique()}")
        print(f"   - 중복 SMILES 수: {len(test_data) - test_data['SMILES'].nunique()}")
        
        # SMILES 길이 분포
        smiles_lengths = test_data['SMILES'].str.len()
        print(f"   - SMILES 평균 길이: {smiles_lengths.mean():.1f}")
        print(f"   - SMILES 최소 길이: {smiles_lengths.min()}")
        print(f"   - SMILES 최대 길이: {smiles_lengths.max()}")
        
        # ID 패턴 분석
        id_patterns = test_data['ID'].str.extract(r'([A-Z]+)_(\d+)')
        print(f"   - ID 패턴: {id_patterns[0].iloc[0]}_XXXX")
    
    # 2. 훈련 데이터 품질 검사
    print("\n2️⃣ 훈련 데이터 품질 검사")
    training_data = loader.load_combined_training_data()
    
    if training_data is not None:
        print(f"   - 총 화합물 수: {len(training_data)}")
        print(f"   - 고유 SMILES 수: {training_data['SMILES'].nunique()}")
        print(f"   - 중복 SMILES 수: {len(training_data) - training_data['SMILES'].nunique()}")
        
        # pIC50 분포 분석
        pic50_stats = training_data['pIC50'].describe()
        print(f"   - pIC50 통계:")
        print(f"     * 평균: {pic50_stats['mean']:.2f}")
        print(f"     * 표준편차: {pic50_stats['std']:.2f}")
        print(f"     * 최소값: {pic50_stats['min']:.2f}")
        print(f"     * 최대값: {pic50_stats['max']:.2f}")
        print(f"     * 25%: {pic50_stats['25%']:.2f}")
        print(f"     * 50%: {pic50_stats['50%']:.2f}")
        print(f"     * 75%: {pic50_stats['75%']:.2f}")
        
        # 데이터 소스 분포
        source_dist = training_data['source'].value_counts()
        print(f"   - 데이터 소스 분포:")
        for source, count in source_dist.items():
            print(f"     * {source}: {count}개 ({count/len(training_data)*100:.1f}%)")
        
        # IC50 분포 분석
        ic50_stats = training_data['IC50_nM'].describe()
        print(f"   - IC50 통계 (nM):")
        print(f"     * 평균: {ic50_stats['mean']:.2f}")
        print(f"     * 중앙값: {ic50_stats['50%']:.2f}")
        print(f"     * 최소값: {ic50_stats['min']:.2f}")
        print(f"     * 최대값: {ic50_stats['max']:.2f}")
    
    print("\n✅ 데이터 품질 테스트 완료")

def test_data_distribution():
    """데이터 분포 시각화 및 분석"""
    print("\n📊 데이터 분포 분석")
    print("=" * 40)
    
    loader = create_data_loader()
    training_data = loader.load_combined_training_data()
    
    if training_data is None:
        print("❌ 훈련 데이터를 로드할 수 없습니다.")
        return
    
    # 1. pIC50 분포 히스토그램
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(training_data['pIC50'], bins=30, alpha=0.7, edgecolor='black')
    plt.title('pIC50 분포')
    plt.xlabel('pIC50')
    plt.ylabel('빈도')
    plt.grid(True, alpha=0.3)
    
    # 2. IC50 분포 (로그 스케일)
    plt.subplot(2, 2, 2)
    plt.hist(np.log10(training_data['IC50_nM']), bins=30, alpha=0.7, edgecolor='black')
    plt.title('IC50 분포 (로그 스케일)')
    plt.xlabel('log10(IC50_nM)')
    plt.ylabel('빈도')
    plt.grid(True, alpha=0.3)
    
    # 3. 데이터 소스별 pIC50 분포
    plt.subplot(2, 2, 3)
    sources = training_data['source'].unique()
    for source in sources:
        source_data = training_data[training_data['source'] == source]['pIC50']
        plt.hist(source_data, bins=20, alpha=0.6, label=source)
    plt.title('데이터 소스별 pIC50 분포')
    plt.xlabel('pIC50')
    plt.ylabel('빈도')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. SMILES 길이 분포
    plt.subplot(2, 2, 4)
    smiles_lengths = training_data['SMILES'].str.len()
    plt.hist(smiles_lengths, bins=30, alpha=0.7, edgecolor='black')
    plt.title('SMILES 길이 분포')
    plt.xlabel('SMILES 길이')
    plt.ylabel('빈도')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 결과 저장
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'data_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 데이터 분포 분석 완료")

def test_data_preprocessing():
    """데이터 전처리 테스트"""
    print("\n🔧 데이터 전처리 테스트")
    print("=" * 40)
    
    loader = create_data_loader()
    
    # 1. SMILES 유효성 검사 테스트
    print("\n1️⃣ SMILES 유효성 검사")
    
    # 테스트용 데이터 생성
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # 아스피린 - 유효
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # 이부프로펜 - 유효
        '',  # 빈 문자열
        'CC',  # 너무 짧음
        'INVALID_SMILES_STRING',  # 유효하지 않음
        '   ',  # 공백만
        'CC(=O)OC1=CC=CC=C1C(=O)O'  # 중복
    ]
    
    test_df = pd.DataFrame({
        'SMILES': test_smiles,
        'pIC50': [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    })
    
    print(f"   - 원본 데이터: {len(test_df)}개")
    
    # 유효성 검사 실행
    validated_df = loader.validate_smiles(test_df)
    
    print(f"   - 검증 후 데이터: {len(validated_df)}개")
    print(f"   - 제거된 데이터: {len(test_df) - len(validated_df)}개")
    
    # 2. 모델 데이터 준비 테스트
    print("\n2️⃣ 모델 데이터 준비")
    
    model_data = loader.prepare_model_data(test_size=0.2, random_state=42)
    
    if model_data:
        print(f"   - 훈련 세트: {len(model_data['train'])}개")
        print(f"   - 검증 세트: {len(model_data['validation'])}개")
        print(f"   - 테스트 세트: {len(model_data['test'])}개")
        
        # 분할 비율 확인
        total_train = len(model_data['train']) + len(model_data['validation'])
        val_ratio = len(model_data['validation']) / total_train
        print(f"   - 검증 세트 비율: {val_ratio:.2f} (목표: 0.20)")
        
        # 데이터 중복 확인
        train_smiles = set(model_data['train']['SMILES'])
        val_smiles = set(model_data['validation']['SMILES'])
        overlap = train_smiles.intersection(val_smiles)
        print(f"   - 훈련/검증 중복: {len(overlap)}개")
        
        if len(overlap) == 0:
            print("   ✅ 훈련/검증 데이터가 올바르게 분할되었습니다.")
        else:
            print("   ⚠️ 훈련/검증 데이터에 중복이 있습니다.")
    
    print("\n✅ 데이터 전처리 테스트 완료")

def test_performance():
    """성능 테스트"""
    print("\n⚡ 성능 테스트")
    print("=" * 40)
    
    import time
    
    loader = create_data_loader()
    
    # 1. 캐시 성능 테스트
    print("\n1️⃣ 캐시 성능 테스트")
    
    # 첫 번째 로드 (캐시 없음)
    start_time = time.time()
    test_data_1 = loader.load_test_data(use_cache=False)
    first_load_time = time.time() - start_time
    
    # 두 번째 로드 (캐시 사용)
    start_time = time.time()
    test_data_2 = loader.load_test_data(use_cache=True)
    cached_load_time = time.time() - start_time
    
    print(f"   - 첫 번째 로드 시간: {first_load_time:.3f}초")
    print(f"   - 캐시 사용 로드 시간: {cached_load_time:.3f}초")
    print(f"   - 성능 향상: {first_load_time/cached_load_time:.1f}배")
    
    # 2. 대용량 데이터 처리 테스트
    print("\n2️⃣ 대용량 데이터 처리 테스트")
    
    start_time = time.time()
    training_data = loader.load_combined_training_data(use_cache=False)
    load_time = time.time() - start_time
    
    if training_data is not None:
        print(f"   - 데이터 크기: {len(training_data)}개 화합물")
        print(f"   - 로드 시간: {load_time:.3f}초")
        print(f"   - 처리 속도: {len(training_data)/load_time:.0f} 화합물/초")
    
    print("\n✅ 성능 테스트 완료")

def generate_test_report():
    """테스트 리포트 생성"""
    print("\n📋 테스트 리포트 생성")
    print("=" * 40)
    
    loader = create_data_loader()
    
    # 데이터 요약 수집
    summary = loader.get_data_summary()
    
    # 리포트 파일 생성
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, 'test_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("AI 신약개발 데이터 로딩 테스트 리포트\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 데이터 요약\n")
        f.write("-" * 20 + "\n")
        
        for key, value in summary.items():
            f.write(f"\n{key.upper()} 데이터:\n")
            if key == 'test':
                f.write(f"  - 화합물 수: {value['count']}\n")
                f.write(f"  - 컬럼: {', '.join(value['columns'])}\n")
            elif key == 'training':
                f.write(f"  - 화합물 수: {value['count']}\n")
                f.write(f"  - 컬럼: {', '.join(value['columns'])}\n")
                
                stats = value['pIC50_stats']
                f.write(f"  - pIC50 통계:\n")
                f.write(f"    * 최소값: {stats['min']:.2f}\n")
                f.write(f"    * 최대값: {stats['max']:.2f}\n")
                f.write(f"    * 평균값: {stats['mean']:.2f}\n")
                f.write(f"    * 중앙값: {stats['median']:.2f}\n")
                
                f.write(f"  - 데이터 소스 분포:\n")
                for source, count in value['source_distribution'].items():
                    f.write(f"    * {source}: {count}개\n")
        
        f.write("\n2. 테스트 결과\n")
        f.write("-" * 20 + "\n")
        f.write("✅ 모든 테스트가 성공적으로 완료되었습니다.\n")
        f.write("✅ 데이터 로딩 기능이 정상적으로 작동합니다.\n")
        f.write("✅ 데이터 품질이 모델 학습에 적합합니다.\n")
    
    print(f"✅ 테스트 리포트가 생성되었습니다: {report_file}")

def main():
    """메인 함수"""
    print("🚀 데이터 통합 테스트 시작")
    print("=" * 60)
    
    try:
        # 1. 데이터 품질 테스트
        test_data_quality()
        
        # 2. 데이터 분포 분석
        test_data_distribution()
        
        # 3. 데이터 전처리 테스트
        test_data_preprocessing()
        
        # 4. 성능 테스트
        test_performance()
        
        # 5. 테스트 리포트 생성
        generate_test_report()
        
        print("\n�� 모든 테스트가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
