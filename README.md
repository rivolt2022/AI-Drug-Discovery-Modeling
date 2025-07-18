아래는 **AI 신약개발 모델링** 프로젝트의 데이터 사이언스 워크플로우를 Markdown (`.md`) 파일 스타일로 예쁘게 정리한 예시입니다.
각 단계에 적절한 코드/설명 공간, 체크리스트, 강조 포인트, 그리고 마크다운 시각적 요소(이모지, 박스, 리스트 등)를 사용했습니다.

---

````markdown
# 🧬 AI Drug Discovery Modeling Project Workflow

본 프로젝트는 **AI 기반 신약개발(Drug Discovery)**을 위해 분자 구조 데이터와 생물학적 활성 데이터를 통합하여, 머신러닝 기반 예측 모델을 구축하는 과정을 단계별로 정리한 문서입니다.

---

## 📁 1단계. 데이터 확인 및 정제

- [x] 각 데이터 파일(CAS, ChEMBL, PubChem 등)을 `pandas`로 읽기
- [x] 컬럼명/내용 확인 (예: `SMILES`, `IC50`, `Activity_Value` 등)
- [x] 필요한 컬럼만 남기기  
- [x] 컬럼명 통일 (`SMILES`, `IC50` 등)
- [x] 여러 파일을 하나로 합치기
- [x] IC50 값이 없는 행, 0 이하 행, 중복 SMILES 모두 제거  
- [ ] (선택) 값이 너무 튀는 이상치는 분석 후 제거

<details>
<summary>예시 코드</summary>

```python
import pandas as pd

chembl = pd.read_csv('ChEMBL.csv', sep=';')
pubchem = pd.read_csv('PubChem.csv', sep=',')

# 컬럼명 맞추기
chembl = chembl.rename(columns={'Activity_Value': 'IC50'})
pubchem = pubchem.rename(columns={'Activity': 'IC50'})

# 필요한 컬럼만 남기기
chembl = chembl[['SMILES', 'IC50']]
pubchem = pubchem[['SMILES', 'IC50']]

# 합치기
df = pd.concat([chembl, pubchem], ignore_index=True)

# IC50 결측/0이하/중복 제거
df = df.dropna(subset=['SMILES', 'IC50'])
df = df[df['IC50'] > 0]
df = df.drop_duplicates(subset=['SMILES'])
````

</details>

---

## 🧪 2단계. 특성(피처) 생성 및 전처리

* [x] SMILES → Fingerprint 변환 (예: RDKit)
* [x] SMILES → Molecular Descriptor 변환
* [x] Fingerprint & Descriptor 합치기
* [x] 결측치는 평균 등으로 채우기
* [x] 피처 표준화 (예: `StandardScaler`)

<details>
<summary>예시 코드</summary>

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
import numpy as np

def smiles_to_fp(smiles, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    else:
        return None

fps = [smiles_to_fp(s) for s in df['SMILES']]
# 디스크립터도 추가
desc = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in df['SMILES']]

# 결측 처리
desc = np.nan_to_num(desc)
features = np.column_stack([fps, desc])
scaler = StandardScaler()
X = scaler.fit_transform(features)
```

</details>

---

## 🎯 3단계. 목표값(pIC50) 변환

* [x] IC50 → pIC50 변환

  > `pIC50 = 9 - log10(IC50)`
* [x] 라벨은 pIC50 사용
* [x] (제출 시) 예측된 pIC50 → IC50로 역변환

<details>
<summary>예시 코드</summary>

```python
import numpy as np
df['pIC50'] = 9 - np.log10(df['IC50'])
```

</details>

---

## 🤖 4단계. 모델 학습 및 평가

* [x] 머신러닝 모델 학습 (예: LightGBM, RandomForest 등)
* [x] KFold 교차검증
* [x] 평가 함수 구현 (예: RMSE, R²)
* [x] 결과/Feature Importance 확인

<details>
<summary>예시 코드</summary>

```python
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X):
    model = RandomForestRegressor()
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[val_idx])
    print('R2:', r2_score(y[val_idx], preds))
    print('RMSE:', mean_squared_error(y[val_idx], preds, squared=False))
```

</details>

---

## 🛠️ 5단계. 하이퍼파라미터 튜닝

* [x] Optuna 등으로 파라미터 탐색
* [x] 최적의 파라미터 저장

<details>
<summary>예시 코드</summary>

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds, squared=False)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
```

</details>

---

## 📊 6단계. 테스트셋 예측 및 제출 파일 생성

* [x] test.csv 파일 읽기, 피처 생성
* [x] 학습 모델로 예측 (pIC50 → IC50 변환)
* [x] sample\_submission.csv 포맷에 맞게 저장

<details>
<summary>예시 코드</summary>

```python
test = pd.read_csv('test.csv')
test_fp = [smiles_to_fp(s) for s in test['SMILES']]
test_desc = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in test['SMILES']]
test_features = np.column_stack([test_fp, test_desc])
test_X = scaler.transform(test_features)

test_pIC50 = model.predict(test_X)
test['IC50'] = 10 ** (9 - test_pIC50)

submission = test[['ID', 'IC50']]
submission.to_csv('submission.csv', index=False)
```

</details>

---

## 🔄 7단계. 반복 실험 & 고도화

* [x] 다양한 피처/모델/정제 방법 시도
* [x] 결과 기록 및 점수 비교
* [x] 최종 모델 및 실험 로그 문서화

---

## ✨ 프로젝트 팁 & 체크포인트

* **모델 성능**은 데이터 전처리, 피처 다양성, 하이퍼파라미터 튜닝에서 크게 달라질 수 있습니다.
* \*\*실험 결과와 코드, 환경(시드, 버전 등)\*\*을 항상 기록하세요.
* **IC50 값**은 log 스케일 변환이 성능에 큰 영향을 미칩니다.
