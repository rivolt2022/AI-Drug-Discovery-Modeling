import pandas as pd
import numpy as np
import os

def read_test_csv():
    """테스트 데이터 CSV 파일 읽기"""
    print("=== 테스트 데이터 CSV 읽기 ===")
    
    try:
        # 테스트 데이터 읽기
        test_file = 'data/test.csv'
        df = pd.read_csv(test_file)
        
        print(f"�� 파일: {test_file}")
        print(f"📋 데이터 크기: {df.shape}")
        print(f"📝 컬럼 수: {len(df.columns)}")
        
        print(f"\n�� 컬럼명:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\n�� 처음 5개 행:")
        print(df.head())
        
        print(f"\n📊 데이터 타입:")
        print(df.dtypes)
        
        print(f"\n🔍 결측치 확인:")
        missing_data = df.isnull().sum()
        print(missing_data)
        
        print(f"\n📈 결측치 비율:")
        missing_ratio = (missing_data / len(df)) * 100
        for col, ratio in missing_ratio.items():
            print(f"  {col}: {ratio:.2f}%")
        
        # SMILES 컬럼 분석
        if 'Smiles' in df.columns:
            print(f"\n🔬 SMILES 컬럼 분석:")
            smiles_col = df['Smiles']
            print(f"  - 총 SMILES 수: {len(smiles_col)}")
            print(f"  - 고유 SMILES 수: {smiles_col.nunique()}")
            print(f"  - 결측치: {smiles_col.isnull().sum()}")
            
            # SMILES 길이 분석
            smiles_lengths = smiles_col.dropna().astype(str).str.len()
            print(f"  - 평균 길이: {smiles_lengths.mean():.1f}")
            print(f"  - 최소 길이: {smiles_lengths.min()}")
            print(f"  - 최대 길이: {smiles_lengths.max()}")
            
            # SMILES 샘플 출력
            print(f"\n�� SMILES 샘플 (처음 5개):")
            for i, smiles in enumerate(smiles_col.head()):
                print(f"  {i+1}. {smiles}")
        
        # ID 컬럼 분석
        if 'ID' in df.columns:
            print(f"\n🔬 ID 컬럼 분석:")
            id_col = df['ID']
            print(f"  - 총 ID 수: {len(id_col)}")
            print(f"  - 고유 ID 수: {id_col.nunique()}")
            print(f"  - 결측치: {id_col.isnull().sum()}")
            
            # ID 패턴 분석
            print(f"  - ID 패턴 샘플:")
            for i, id_val in enumerate(id_col.head()):
                print(f"    {i+1}. {id_val}")
        
        return df
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_chembl_csv():
    """ChEMBL ASK1 IC50 데이터 CSV 파일 읽기"""
    print(f"\n=== ChEMBL ASK1 IC50 데이터 CSV 읽기 ===")
    
    try:
        # ChEMBL 데이터 읽기 (세미콜론 구분자 사용)
        chembl_file = 'data/ChEMBL_ASK1(IC50).csv'
        df = pd.read_csv(chembl_file, sep=';')
        
        print(f"📊 파일: {chembl_file}")
        print(f"📋 데이터 크기: {df.shape}")
        print(f"📝 컬럼 수: {len(df.columns)}")
        
        print(f"\n�� 컬럼명:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\n�� 처음 3개 행:")
        print(df.head(3))
        
        print(f"\n📊 데이터 타입:")
        print(df.dtypes)
        
        print(f"\n🔍 결측치 확인:")
        missing_data = df.isnull().sum()
        print(missing_data)
        
        # 중요 컬럼들 분석
        important_cols = ['Smiles', 'Standard Value', 'Standard Units', 'pChEMBL Value', 'Molecule Name']
        
        print(f"\n🔬 중요 컬럼 분석:")
        for col in important_cols:
            if col in df.columns:
                print(f"\n📋 {col} 컬럼:")
                col_data = df[col]
                print(f"  - 총 데이터 수: {len(col_data)}")
                print(f"  - 결측치: {col_data.isnull().sum()}")
                print(f"  - 고유값 수: {col_data.nunique()}")
                
                # 수치형 컬럼인 경우 통계 정보
                if col in ['Standard Value', 'pChEMBL Value']:
                    try:
                        numeric_data = pd.to_numeric(col_data, errors='coerce')
                        non_null_numeric = numeric_data.dropna()
                        if len(non_null_numeric) > 0:
                            print(f"  - 수치형 데이터 수: {len(non_null_numeric)}")
                            print(f"  - 최소값: {non_null_numeric.min()}")
                            print(f"  - 최대값: {non_null_numeric.max()}")
                            print(f"  - 평균값: {non_null_numeric.mean():.4f}")
                            print(f"  - 중앙값: {non_null_numeric.median():.4f}")
                    except:
                        print(f"  - 수치형 변환 불가")
                
                # SMILES 컬럼인 경우 길이 분석
                if col == 'Smiles':
                    smiles_lengths = col_data.dropna().astype(str).str.len()
                    print(f"  - 평균 길이: {smiles_lengths.mean():.1f}")
                    print(f"  - 최소 길이: {smiles_lengths.min()}")
                    print(f"  - 최대 길이: {smiles_lengths.max()}")
                    
                    # SMILES 샘플 출력
                    print(f"  - SMILES 샘플 (처음 3개):")
                    for i, smiles in enumerate(col_data.dropna().head(3)):
                        print(f"    {i+1}. {smiles}")
        
        # IC50 값 분포 분석
        if 'Standard Value' in df.columns and 'Standard Units' in df.columns:
            print(f"\n🔬 IC50 값 분포 분석:")
            ic50_data = df[['Standard Value', 'Standard Units']].copy()
            
            # nM 단위 데이터만 필터링
            nm_data = ic50_data[ic50_data['Standard Units'] == 'nM']
            print(f"  - nM 단위 데이터 수: {len(nm_data)}")
            
            if len(nm_data) > 0:
                try:
                    ic50_values = pd.to_numeric(nm_data['Standard Value'], errors='coerce')
                    valid_ic50 = ic50_values.dropna()
                    
                    if len(valid_ic50) > 0:
                        print(f"  - 유효한 IC50 값 수: {len(valid_ic50)}")
                        print(f"  - IC50 최소값: {valid_ic50.min():.2f} nM")
                        print(f"  - IC50 최대값: {valid_ic50.max():.2f} nM")
                        print(f"  - IC50 평균값: {valid_ic50.mean():.2f} nM")
                        print(f"  - IC50 중앙값: {valid_ic50.median():.2f} nM")
                        
                        # pIC50 계산
                        pic50_values = -np.log10(valid_ic50 * 1e-9)  # nM -> M 변환 후 -log10
                        print(f"  - pIC50 최소값: {pic50_values.min():.2f}")
                        print(f"  - pIC50 최대값: {pic50_values.max():.2f}")
                        print(f"  - pIC50 평균값: {pic50_values.mean():.2f}")
                        print(f"  - pIC50 중앙값: {pic50_values.median():.2f}")
                except Exception as e:
                    print(f"  - IC50 값 분석 중 오류: {e}")
        
        return df
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_pubchem_csv():
    """PubChem ASK1 데이터 CSV 파일 읽기"""
    print(f"\n=== PubChem ASK1 데이터 CSV 읽기 ===")
    
    try:
        # PubChem 데이터 읽기
        pubchem_file = 'data/Pubchem_ASK1.csv'
        
        # 파일이 큰 경우 처음 몇 행만 읽어서 구조 확인
        print(f"📊 파일: {pubchem_file}")
        
        # 파일 크기 확인
        file_size = os.path.getsize(pubchem_file)
        print(f"📁 파일 크기: {file_size / (1024*1024):.2f} MB")
        
        # 처음 몇 행만 읽어서 구조 확인
        df_sample = pd.read_csv(pubchem_file, nrows=10)
        
        print(f"📋 샘플 데이터 크기: {df_sample.shape}")
        print(f"📝 컬럼 수: {len(df_sample.columns)}")
        
        print(f"\n�� 컬럼명:")
        for i, col in enumerate(df_sample.columns):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\n�� 처음 5개 행:")
        print(df_sample.head())
        
        print(f"\n📊 데이터 타입:")
        print(df_sample.dtypes)
        
        # 전체 파일 읽기 (메모리 허용 시)
        try:
            print(f"\n🔄 전체 파일 읽기 중...")
            df = pd.read_csv(pubchem_file)
            
            print(f"📋 전체 데이터 크기: {df.shape}")
            
            print(f"\n🔍 결측치 확인:")
            missing_data = df.isnull().sum()
            print(missing_data)
            
            # 중요 컬럼들 찾기
            smiles_cols = []
            ic50_cols = []
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'smiles' in col_lower or 'structure' in col_lower:
                    smiles_cols.append(col)
                if 'ic50' in col_lower or 'activity' in col_lower or 'value' in col_lower:
                    ic50_cols.append(col)
            
            print(f"\n🔬 중요 컬럼 분석:")
            print(f"  - SMILES 관련 컬럼: {smiles_cols}")
            print(f"  - IC50 관련 컬럼: {ic50_cols}")
            
            # SMILES 컬럼이 있다면 분석
            if smiles_cols:
                for col in smiles_cols:
                    print(f"\n📋 {col} 컬럼 분석:")
                    col_data = df[col]
                    print(f"  - 총 데이터 수: {len(col_data)}")
                    print(f"  - 고유값 수: {col_data.nunique()}")
                    print(f"  - 결측치: {col_data.isnull().sum()}")
                    
                    # SMILES 길이 분석
                    smiles_lengths = col_data.dropna().astype(str).str.len()
                    print(f"  - 평균 길이: {smiles_lengths.mean():.1f}")
                    print(f"  - 최소 길이: {smiles_lengths.min()}")
                    print(f"  - 최대 길이: {smiles_lengths.max()}")
            
            return df
            
        except MemoryError:
            print(f"⚠️ 메모리 부족으로 전체 파일을 읽을 수 없습니다.")
            print(f"📋 샘플 데이터만 반환합니다.")
            return df_sample
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_sample_submission_csv():
    """샘플 제출 파일 CSV 읽기"""
    print(f"\n=== 샘플 제출 파일 CSV 읽기 ===")
    
    try:
        # 샘플 제출 파일 읽기
        submission_file = 'data/sample_submission.csv'
        df = pd.read_csv(submission_file)
        
        print(f"📊 파일: {submission_file}")
        print(f"📋 데이터 크기: {df.shape}")
        print(f"📝 컬럼 수: {len(df.columns)}")
        
        print(f"\n�� 컬럼명:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\n📋 처음 10개 행:")
        print(df.head(10))
        
        print(f"\n📊 데이터 타입:")
        print(df.dtypes)
        
        print(f"\n🔍 결측치 확인:")
        missing_data = df.isnull().sum()
        print(missing_data)
        
        # ID 컬럼 분석
        if 'ID' in df.columns:
            print(f"\n🔬 ID 컬럼 분석:")
            id_col = df['ID']
            print(f"  - 총 ID 수: {len(id_col)}")
            print(f"  - 고유 ID 수: {id_col.nunique()}")
            print(f"  - 결측치: {id_col.isnull().sum()}")
            
            # ID 패턴 분석
            print(f"  - ID 패턴 샘플:")
            for i, id_val in enumerate(id_col.head()):
                print(f"    {i+1}. {id_val}")
        
        # ASK1_IC50_nM 컬럼 분석
        if 'ASK1_IC50_nM' in df.columns:
            print(f"\n�� ASK1_IC50_nM 컬럼 분석:")
            ic50_col = df['ASK1_IC50_nM']
            print(f"  - 총 데이터 수: {len(ic50_col)}")
            print(f"  - 결측치: {ic50_col.isnull().sum()}")
            
            # 수치형 분석
            try:
                numeric_data = pd.to_numeric(ic50_col, errors='coerce')
                non_null_numeric = numeric_data.dropna()
                if len(non_null_numeric) > 0:
                    print(f"  - 수치형 데이터 수: {len(non_null_numeric)}")
                    print(f"  - 최소값: {non_null_numeric.min()}")
                    print(f"  - 최대값: {non_null_numeric.max()}")
                    print(f"  - 평균값: {non_null_numeric.mean():.4f}")
                    print(f"  - 중앙값: {non_null_numeric.median():.4f}")
                    
                    # 고유값 분석
                    unique_values = non_null_numeric.unique()
                    print(f"  - 고유값 수: {len(unique_values)}")
                    print(f"  - 고유값들: {sorted(unique_values)}")
            except:
                print(f"  - 수치형 변환 불가")
        
        return df
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_all_csv_files():
    """모든 CSV 파일 분석"""
    print("=" * 60)
    print("�� 모든 CSV 파일 분석")
    print("=" * 60)
    
    # 1. 테스트 데이터 읽기
    test_df = read_test_csv()
    
    # 2. ChEMBL 데이터 읽기
    chembl_df = read_chembl_csv()
    
    # 3. PubChem 데이터 읽기
    pubchem_df = read_pubchem_csv()
    
    # 4. 샘플 제출 파일 읽기
    submission_df = read_sample_submission_csv()
    
    # 데이터 요약
    print(f"\n" + "=" * 60)
    print(f"�� 데이터 요약")
    print(f"=" * 60)
    
    if test_df is not None:
        print(f"  ✅ 테스트 데이터: {len(test_df)}개 화합물")
    if chembl_df is not None:
        print(f"  ✅ ChEMBL 데이터: {len(chembl_df)}개 측정값")
    if pubchem_df is not None:
        print(f"  ✅ PubChem 데이터: {len(pubchem_df)}개 데이터")
    if submission_df is not None:
        print(f"  ✅ 샘플 제출 파일: {len(submission_df)}개 예측값")
    
    print(f"\n✅ 모든 CSV 파일 분석 완료!")
    
    return {
        'test': test_df,
        'chembl': chembl_df,
        'pubchem': pubchem_df,
        'submission': submission_df
    }

def get_clean_training_data():
    """정제된 훈련 데이터 반환 (ChEMBL + PubChem)"""
    print(f"\n=== 정제된 훈련 데이터 생성 ===")
    
    try:
        # ChEMBL 데이터 읽기
        chembl_df = read_chembl_csv()
        
        if chembl_df is None:
            print("❌ ChEMBL 데이터를 읽을 수 없습니다.")
            return None
        
        # 필요한 컬럼만 선택
        required_cols = ['Smiles', 'Standard Value', 'Standard Units']
        available_cols = [col for col in required_cols if col in chembl_df.columns]
        
        if len(available_cols) < 2:
            print("❌ 필요한 컬럼이 부족합니다.")
            return None
        
        # 데이터 정제
        clean_df = chembl_df[available_cols].copy()
        
        # nM 단위 데이터만 필터링
        if 'Standard Units' in clean_df.columns:
            clean_df = clean_df[clean_df['Standard Units'] == 'nM']
        
        # 결측치 제거
        clean_df = clean_df.dropna()
        
        # IC50 값을 수치형으로 변환
        if 'Standard Value' in clean_df.columns:
            clean_df['IC50_nM'] = pd.to_numeric(clean_df['Standard Value'], errors='coerce')
            clean_df = clean_df.dropna(subset=['IC50_nM'])
            
            # pIC50 계산
            clean_df['pIC50'] = -np.log10(clean_df['IC50_nM'] * 1e-9)
        
        print(f"�� 정제된 데이터 크기: {clean_df.shape}")
        print(f"�� 컬럼: {list(clean_df.columns)}")
        
        if 'IC50_nM' in clean_df.columns:
            print(f"🔬 IC50 통계:")
            print(f"  - 최소값: {clean_df['IC50_nM'].min():.2f} nM")
            print(f"  - 최대값: {clean_df['IC50_nM'].max():.2f} nM")
            print(f"  - 평균값: {clean_df['IC50_nM'].mean():.2f} nM")
            print(f"  - 중앙값: {clean_df['IC50_nM'].median():.2f} nM")
        
        if 'pIC50' in clean_df.columns:
            print(f"🔬 pIC50 통계:")
            print(f"  - 최소값: {clean_df['pIC50'].min():.2f}")
            print(f"  - 최대값: {clean_df['pIC50'].max():.2f}")
            print(f"  - 평균값: {clean_df['pIC50'].mean():.2f}")
            print(f"  - 중앙값: {clean_df['pIC50'].median():.2f}")
        
        return clean_df
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 모든 CSV 파일 분석
    data_dict = analyze_all_csv_files()
    
    # 정제된 훈련 데이터 생성
    clean_training_data = get_clean_training_data()
    
    if clean_training_data is not None:
        print(f"\n📋 정제된 훈련 데이터 샘플:")
        print(clean_training_data.head())
    
    print(f"\n✅ CSV 파일 분석 완료!") 