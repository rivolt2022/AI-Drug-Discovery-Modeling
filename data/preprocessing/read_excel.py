import pandas as pd
import numpy as np
import os

def analyze_excel_sheets():
    """엑셀 파일의 모든 시트 분석"""
    print("=== 엑셀 파일 시트 분석 ===")
    
    try:
        # 엑셀 파일의 모든 시트 확인
        excel_file = 'data/CAS_KPBMA_MAP3K5_IC50s.xlsx'
        xl = pd.ExcelFile(excel_file)
        
        print(f"📊 엑셀 파일: {excel_file}")
        print(f"📋 총 시트 수: {len(xl.sheet_names)}")
        
        print(f"\n📝 시트 목록:")
        for i, sheet_name in enumerate(xl.sheet_names):
            print(f"  {i+1:2d}. {sheet_name}")
        
        return xl.sheet_names
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_ligand_smiles_sheet():
    """Ligand Number Names SMILES 시트 읽기"""
    print(f"\n=== Ligand Number Names SMILES 시트 읽기 ===")
    
    try:
        excel_file = 'data/CAS_KPBMA_MAP3K5_IC50s.xlsx'
        
        # 첫 번째 행은 저작권 정보, 두 번째 행이 실제 헤더
        df = pd.read_excel(excel_file, sheet_name='Ligand Number Names SMILES', header=1)
        
        print(f"📊 데이터 크기: {df.shape}")
        print(f"📋 컬럼 수: {len(df.columns)}")
        
        print(f"\n📝 컬럼명:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\n📋 처음 5개 행:")
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
        if 'SMILES' in df.columns:
            print(f"\n🔬 SMILES 컬럼 분석:")
            smiles_col = df['SMILES']
            print(f"  - 총 SMILES 수: {len(smiles_col)}")
            print(f"  - 고유 SMILES 수: {smiles_col.nunique()}")
            print(f"  - 결측치: {smiles_col.isnull().sum()}")
            
            # SMILES 길이 분석
            smiles_lengths = smiles_col.dropna().astype(str).str.len()
            print(f"  - 평균 길이: {smiles_lengths.mean():.1f}")
            print(f"  - 최소 길이: {smiles_lengths.min()}")
            print(f"  - 최대 길이: {smiles_lengths.max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_ic50_sheet():
    """MAP3K5 Ligand IC50s 시트 읽기"""
    print(f"\n=== MAP3K5 Ligand IC50s 시트 읽기 ===")
    
    try:
        excel_file = 'data/CAS_KPBMA_MAP3K5_IC50s.xlsx'
        
        # 첫 번째 행은 저작권 정보, 두 번째 행이 실제 헤더
        df = pd.read_excel(excel_file, sheet_name='MAP3K5 Ligand IC50s', header=1)
        
        print(f"📊 데이터 크기: {df.shape}")
        print(f"📋 컬럼 수: {len(df.columns)}")
        
        print(f"\n📝 컬럼명:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\n📋 처음 5개 행:")
        print(df.head())
        
        print(f"\n📊 데이터 타입:")
        print(df.dtypes)
        
        print(f"\n🔍 결측치 확인:")
        missing_data = df.isnull().sum()
        print(missing_data)
        
        # IC50 관련 컬럼 분석
        ic50_cols = ['Single Value (Parsed)', 'pX Value', 'Display Measurement']
        for col in ic50_cols:
            if col in df.columns:
                print(f"\n🔬 {col} 분석:")
                col_data = df[col]
                print(f"  - 총 데이터 수: {len(col_data)}")
                print(f"  - 결측치: {col_data.isnull().sum()}")
                
                # 수치형으로 변환 가능한지 확인
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
        
        # SMILES 컬럼 분석
        if 'SMILES' in df.columns:
            print(f"\n🔬 SMILES 컬럼 분석:")
            smiles_col = df['SMILES']
            print(f"  - 총 SMILES 수: {len(smiles_col)}")
            print(f"  - 고유 SMILES 수: {smiles_col.nunique()}")
            print(f"  - 결측치: {smiles_col.isnull().sum()}")
        
        return df
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_data_dictionary():
    """Data Dictionary 시트 읽기"""
    print(f"\n=== Data Dictionary 시트 읽기 ===")
    
    try:
        excel_file = 'data/CAS_KPBMA_MAP3K5_IC50s.xlsx'
        
        # 첫 번째 행은 시트 제목, 두 번째 행이 컬럼명, 세 번째 행이 설명
        df = pd.read_excel(excel_file, sheet_name='Data Dictionary', header=1)
        
        print(f"📊 데이터 크기: {df.shape}")
        print(f"📋 컬럼 수: {len(df.columns)}")
        
        print(f"\n📝 컬럼명:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\n📋 데이터 내용:")
        print(df)
        
        return df
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_first_header_and_data(sheet_name=None):
    """첫 번째 헤더와 첫 번째 데이터만 읽기"""
    print(f"\n=== 첫 번째 헤더와 데이터 분석 ===")
    
    try:
        excel_file = 'data/CAS_KPBMA_MAP3K5_IC50s.xlsx'
        
        if sheet_name:
            print(f"📋 시트: {sheet_name}")
            # 특정 시트 읽기
            df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        else:
            print(f"📋 기본 시트 (첫 번째)")
            # 기본 시트 읽기
            df_raw = pd.read_excel(excel_file, header=None)
        
        print(f"📊 데이터 크기: {df_raw.shape}")
        
        # 첫 번째 행 (헤더) 분석
        print(f"\n🔍 첫 번째 행 (헤더) 분석:")
        first_row = df_raw.iloc[0]
        print("헤더 내용:")
        for i, val in enumerate(first_row):
            print(f"  컬럼 {i+1:2d}: {val}")
        
        # 두 번째 행 (첫 번째 데이터) 분석
        print(f"\n🔍 두 번째 행 (첫 번째 데이터) 분석:")
        second_row = df_raw.iloc[1]
        print("첫 번째 데이터:")
        for i, val in enumerate(second_row):
            print(f"  컬럼 {i+1:2d}: {val}")
        
        # 세 번째 행도 확인 (데이터 패턴 파악)
        if len(df_raw) > 2:
            print(f"\n🔍 세 번째 행 분석:")
            third_row = df_raw.iloc[2]
            print("두 번째 데이터:")
            for i, val in enumerate(third_row):
                print(f"  컬럼 {i+1:2d}: {val}")
        
        # 데이터 타입 분석
        print(f"\n📊 데이터 타입 분석:")
        for i in range(min(3, len(df_raw))):
            row = df_raw.iloc[i]
            print(f"행 {i+1} 데이터 타입:")
            for j, val in enumerate(row):
                if pd.notna(val):
                    print(f"  컬럼 {j+1}: {type(val).__name__} = {val}")
                else:
                    print(f"  컬럼 {j+1}: None/NaN")
        
        return df_raw
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_all_sheets_headers():
    """모든 시트의 헤더만 분석"""
    print(f"\n=== 모든 시트 헤더 분석 ===")
    
    try:
        excel_file = 'data/CAS_KPBMA_MAP3K5_IC50s.xlsx'
        xl = pd.ExcelFile(excel_file)
        
        for sheet_name in xl.sheet_names:
            print(f"\n📋 시트: {sheet_name}")
            print("-" * 50)
            
            # 각 시트의 첫 번째 행만 읽기
            df_header = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, nrows=1)
            
            print(f"헤더 컬럼 수: {len(df_header.columns)}")
            print("헤더 내용:")
            for i, val in enumerate(df_header.iloc[0]):
                print(f"  컬럼 {i+1:2d}: {val}")
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()

def read_cas_excel():
    """CAS 엑셀 파일 읽기 및 구조 확인"""
    print("=== CAS 엑셀 파일 읽기 ===")
    
    try:
        # 엑셀 파일 읽기 (헤더 없이 먼저 확인)
        df_raw = pd.read_excel('data/CAS_KPBMA_MAP3K5_IC50s.xlsx', header=None)
        
        print(f"📊 원본 데이터 크기: {df_raw.shape}")
        print(f"📋 컬럼 수: {len(df_raw.columns)}")
        
        # 처음 몇 행을 확인하여 헤더 위치 찾기
        print(f"\n🔍 처음 10개 행 확인:")
        print(df_raw.head(10))
        
        # 실제 헤더 찾기 (첫 번째 행이 헤더인지 확인)
        print(f"\n🔍 헤더 분석:")
        first_row = df_raw.iloc[0]
        print("첫 번째 행:")
        for i, val in enumerate(first_row):
            print(f"  컬럼 {i}: {val}")
        
        # 두 번째 행도 확인
        second_row = df_raw.iloc[1]
        print("\n두 번째 행:")
        for i, val in enumerate(second_row):
            print(f"  컬럼 {i}: {val}")
        
        # 헤더가 첫 번째 행에 있는 것으로 보이므로 다시 읽기
        print(f"\n🔄 헤더를 첫 번째 행으로 설정하여 다시 읽기:")
        df = pd.read_excel('data/CAS_KPBMA_MAP3K5_IC50s.xlsx', header=0)
        
        print(f"📊 처리된 데이터 크기: {df.shape}")
        print(f"📋 컬럼 수: {len(df.columns)}")
        
        print("\n📝 컬럼명 목록:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\n📋 처음 5개 행:")
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
        
        # 각 컬럼의 고유값 개수 확인
        print(f"\n🔢 각 컬럼의 고유값 개수:")
        for col in df.columns:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count}개")
        
        # SMILES와 IC50 관련 컬럼 찾기
        print(f"\n🔍 SMILES/IC50 관련 컬럼 찾기:")
        smiles_cols = []
        ic50_cols = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'smiles' in col_lower or 'structure' in col_lower or 'molecular' in col_lower:
                smiles_cols.append(col)
                print(f"  SMILES 관련: {col}")
            if 'ic50' in col_lower or 'activity' in col_lower or 'value' in col_lower or 'concentration' in col_lower:
                ic50_cols.append(col)
                print(f"  IC50 관련: {col}")
        
        # 수치형 컬럼 찾기
        print(f"\n🔢 수치형 컬럼 분석:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("수치형 컬럼들:")
            for col in numeric_cols:
                print(f"  {col}:")
                print(f"    - 최소값: {df[col].min()}")
                print(f"    - 최대값: {df[col].max()}")
                print(f"    - 평균값: {df[col].mean():.4f}")
                print(f"    - 중앙값: {df[col].median():.4f}")
        else:
            print("수치형 컬럼이 없습니다.")
        
        # 문자열 컬럼 분석
        print(f"\n📝 문자열 컬럼 분석:")
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            print(f"  {col}:")
            print(f"    - 고유값 개수: {df[col].nunique()}")
            print(f"    - 가장 긴 값 길이: {df[col].astype(str).str.len().max()}")
            print(f"    - 가장 짧은 값 길이: {df[col].astype(str).str.len().min()}")
        
        return df
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_data_structure(df):
    """데이터 구조 상세 분석"""
    if df is None:
        return
    
    print(f"\n=== 데이터 구조 상세 분석 ===")
    
    # 각 컬럼의 샘플 데이터 확인
    print(f"\n📋 각 컬럼의 샘플 데이터:")
    for col in df.columns:
        print(f"\n{col}:")
        # 결측치가 아닌 첫 5개 값 출력
        non_null_values = df[col].dropna().head(5)
        for i, val in enumerate(non_null_values):
            print(f"  {i+1}. {val}")
        
        # 결측치 개수
        null_count = df[col].isnull().sum()
        print(f"  결측치: {null_count}개")

if __name__ == "__main__":
    # 1. 엑셀 파일의 모든 시트 확인
    sheet_names = analyze_excel_sheets()
    
    if sheet_names:
        # 2. 각 시트를 올바른 헤더로 읽기
        print(f"\n{'='*60}")
        print(f"📊 각 시트 정확한 데이터 읽기")
        print(f"{'='*60}")
        
        # Ligand Number Names SMILES 시트
        ligand_df = read_ligand_smiles_sheet()
        
        # MAP3K5 Ligand IC50s 시트
        ic50_df = read_ic50_sheet()
        
        # Data Dictionary 시트
        dict_df = read_data_dictionary()
        
        print(f"\n✅ 모든 시트 분석 완료!")
        
        # 데이터 요약
        print(f"\n📊 데이터 요약:")
        if ligand_df is not None:
            print(f"  - Ligand SMILES: {len(ligand_df)}개 화합물")
        if ic50_df is not None:
            print(f"  - IC50 데이터: {len(ic50_df)}개 측정값")
        if dict_df is not None:
            print(f"  - 데이터 사전: {len(dict_df)}개 설명")
    
    print(f"\n✅ 분석 완료!") 