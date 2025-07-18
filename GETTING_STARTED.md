MAP3K5 IC50 예측 모델 개발 가이드
1. 데이터 읽기 및 정제
1. 개념 설명:
데이터 읽기는 컴퓨터가 대회에서 제공한 데이터를 불러오는 과정입니다. 예를 들어 CSV(Comma-Separated Values) 파일 형태의 train.csv와 test.csv를 읽어와 데이터프레임(표 형태 데이터)으로 만드는 것입니다. 정제는 이렇게 읽은 데이터에 불필요한 부분이나 오류가 없는지 점검하고 손질하는 단계입니다. 초보자도 이해하기 쉽게 말하면, 데이터를 깨끗하게 청소하는 과정입니다. 여기에는 데이터에서 결측치(비어있는 값)가 있는지 확인하고 채우거나 제거하는 작업, 숫자형/문자형 타입이 올바르게 지정되었는지 확인하는 작업 등이 포함됩니다. 데이터 정제를 통해 신뢰할 수 있는 깨끗한 데이터셋을 만들어야 이후 단계(특징 생성, 모델 학습 등)가 원활하게 진행될 수 있습니다. 2. 해야 할 일:
제공된 train.csv (훈련 데이터)와 test.csv (테스트 데이터) 파일을 불러와 데이터프레임으로 생성하기
데이터의 구성 확인하기 (컬럼 이름, 행 개수 등)
불필요한 열이 있다면 제거하거나 필요한 형태로 가공하기
결측치(NaN 등) 존재 여부 확인 및 처리하기 (예: 제거하거나 평균/중앙값으로 채우기 등)
훈련에 영향을 줄 수 있는 이상치 또는 오류가 없는지 점검하기
3. 쉬운 코드 예시:
아래는 pandas 라이브러리를 이용해 CSV 파일을 읽고 간단히 정제하는 예시입니다 (각 단계에 주석을 추가했습니다).
import pandas as pd

# CSV 파일 읽어서 데이터프레임 만들기
train_df = pd.read_csv('train.csv')   # 훈련용 데이터 읽기
test_df = pd.read_csv('test.csv')     # 테스트용 데이터 읽기

# 데이터 일부 확인하기 (상위 5개 행 출력)
print(train_df.head())        # train_df의 처음 몇 행 출력
print(train_df.info())        # train_df 컬럼별 데이터 타입 및 null 개수 등 출력

# (예시) 불필요한 열 제거하기 
# 만약 'Unnamed: 0' 같은 의미 없는 인덱스 열이 있다면 제거
if 'Unnamed: 0' in train_df.columns:
    train_df = train_df.drop('Unnamed: 0', axis=1)
if 'Unnamed: 0' in test_df.columns:
    test_df = test_df.drop('Unnamed: 0', axis=1)

# 결측치(NaN) 확인하기
print(train_df.isnull().sum())  # 각 열별 결측치 개수 출력

# (예시) 결측치 간단 처리: 결측치가 있다면 해당 행 제거
train_df = train_df.dropna()
# 필요한 경우 평균값 등으로 채우는 방식도 가능
# train_df['컬럼명'].fillna(train_df['컬럼명'].mean(), inplace=True)

print(train_df.shape)  # 정제 후 데이터프레임 크기 확인 (행, 열 개수)
위 코드에서는 데이터를 DataFrame으로 불러온 후, 헤더와 정보(head(), info())를 출력하여 내용을 확인합니다. 그 다음 예시로 불필요한 인덱스 열을 제거하고, 결측치가 있으면 간단히 제거(dropna())했습니다. 실제 대회에서는 데이터에 맞게 추가 정제가 필요할 수 있습니다.
2. SMILES → 특징 벡터 변환
1. 개념 설명:
SMILES는 화합물(분자)의 구조를 사람이 읽을 수 있는 문자열로 표현한 형식입니다. 쉽게 말해 분자의 구조식을 문자열로 적어놓은 것입니다. 예를 들어 아스피린 분자의 SMILES는 CC(=O)OC1=CC=CC=C1C(=O)O처럼 생겼습니다. SMILES는 **“단순화된 분자입력 라인표기법”**이라고 불리며, 화학종의 구조를 ASCII 문자로 나타내는 라인 표기법입니다
en.wikipedia.org
. 이러한 문자열 자체로는 컴퓨터가 바로 예측 모델에 활용하기 어렵기 때문에, **분자 구조를 숫자 벡터(특징)**로 변환해야 합니다. 이를 **특징 생성(Feature Generation)**이라고 합니다. 분자를 숫자로 표현하는 한 가지 방법은 **“분자 지문(fingerprint)”**이라는 벡터를 만드는 것입니다. 분자 지문은 분자의 다양한 **부분 구조(substructure)**들의 존재 여부를 0과 1로 표시한 이진 벡터입니다. 예를 들어 Morgan Fingerprint(Extended Connectivity Fingerprint, ECFP)라는 방법은 분자 내 원자 주변 일정 반경 내의 구조들이 어떤 것이 있는지 해싱(hashing)하여 고정 길이의 0/1 벡터로 표현합니다
pmc.ncbi.nlm.nih.gov
. 간단히 말하면, 분자 내 특징적인 조각들이 들어있으면 1, 없으면 0인 자리로 표시한 긴 비트열이라고 생각하면 됩니다. 이렇게 하면 각 분자는 길이가 일정한 숫자 배열(예: 1024차원의 0/1 배열)로 변환되고, 머신러닝 모델에 입력할 수 있습니다. 2. 해야 할 일:
SMILES 문자열을 다룰 수 있는 화학 정보학 라이브러리 설치 (예: RDKit 등)
각 SMILES 문자열을 분자 객체로 변환 (예: RDKit의 MolFromSmiles 함수를 사용)
변환된 분자 객체로부터 특징 벡터 생성 (예: Morgan 지문 벡터 계산)
특징 벡터는 보통 고정 길이의 배열로 생성 (예: 1024 비트 길이의 Morgan fingerprint)
이렇게 생성된 벡터를 설명변수 X로 사용 (훈련용 입력 데이터)
3. 쉬운 코드 예시:
아래는 RDKit 라이브러리를 이용해 SMILES를 분자 객체로 변환하고, Morgan 지문 벡터를 생성하는 간단한 예시입니다 (주석을 통해 주요 부분을 설명합니다).
!pip install rdkit-pypi  # RDKit 설치 (주피터 환경 등에서 필요시)

from rdkit import Chem
from rdkit.Chem import AllChem

# 하나의 SMILES 예시 (아스피린 분자)
smiles_str = 'CC(=O)OC1=CC=CC=C1C(=O)O'  
mol = Chem.MolFromSmiles(smiles_str)    # SMILES 문자열 -> RDKit 분자 객체 생성

# Morgan Fingerprint 계산 (반경=2, 길이=1024 비트짜리 벡터)
fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
# fingerprint는 RDKit의 ExplicitBitVect 객체로 0/1 비트 벡터를 담고 있음

# ExplicitBitVect를 파이썬 list로 변환하여 0/1 배열 확인
bit_vector = list(fingerprint)  
print(f"Fingerprint length: {len(bit_vector)}")        # 1024 길이
print(f"First 20 bits: {bit_vector[:20]}")  # 앞쪽 20개의 비트 출력 (0이나 1의 값들)
위 코드에서 Chem.MolFromSmiles 함수를 통해 SMILES 문자열을 분자 객체로 바꾸고, AllChem.GetMorganFingerprintAsBitVect를 사용하여 Morgan 지문을 생성했습니다. 반경(radius)을 2로, 출력 벡터 길이(nBits)를 1024로 지정하였으며, 이는 흔히 ECFP4 지문이라고 불리는 설정입니다 (반경 2 → 지름 4 연결 구조 반영). 출력된 fingerprint는 1024차원의 이진 벡터로, list(fingerprint)를 통해 파이썬 리스트 형태로 변환할 수 있습니다. 이렇게 하면 각 분자의 SMILES를 고정 길이 숫자 벡터로 변환하여 머신러닝 알고리즘에 입력할 수 있습니다. 실제로는 훈련 데이터의 모든 SMILES에 대해 이 과정을 수행하여 훈련용 특징행렬 X를 얻고, 테스트 데이터의 SMILES에 대해서도 동일하게 변환하여 테스트용 특징행렬을 얻게 됩니다.
3. IC50 → pIC50 변환
1. 개념 설명:
IC50은 어떤 억제제(예: 약물)가 생물학적 또는 생화학적 기능을 50% 억제하는 데 필요한 농도를 뜻하는 값입니다
ko.wikipedia.org
. 쉽게 말하면, “이 화합물이 절반 효과를 내려면 얼마나 많이 필요한가”를 나타내는 수치이며, **약물의 효능(potency)**을 나타내는 지표입니다. IC50 값은 보통 농도 단위(예: 몰 농도 M 또는 nM 등)로 표현되며, 값이 작을수록 적은 농도로도 50% 억제가 가능하므로 더 강한 효능을 가진 약물로 여겨집니다. 모델링을 할 때는 IC50 값을 그대로 쓰기보다는 pIC50 값으로 변환하는 경우가 많습니다. pIC50은 IC50의 **음의 로그값(-log10)**으로 변환한 것입니다
ko.wikipedia.org
. 예를 들어 IC50 = 1 µM일 경우 pIC50 = 6, IC50 = 100 nM(=0.1 µM)일 경우 pIC50 = 7이 됩니다. 로그 변환을 하는 이유는 농도 값의 범위가 매우 클 때 값 분포를 안정화시키고, 모델이 학습하기 쉽게 값의 스케일을 줄여주기 위해서입니다. 또한 pIC50이 높을수록 (IC50이 낮을수록) 약물이 강력함을 직관적으로 보여주기 때문에, 연구자들은 IC50 대신 pIC50을 많이 사용합니다
ko.wikipedia.org
. 요약하면, pIC50 = -log10(IC50) (여기서 IC50은 **몰 농도(M)**로 넣어야 합니다)입니다. 이 변환으로 값이 큰 쪽이 강한 활성을 뜻하게 바뀌고, 숫자 범위도 다루기 쉬워집니다. 2. 해야 할 일:
IC50 단위 확인: 주어진 IC50 값의 단위를 파악 (예: nM인지 µM인지). 대회 데이터 설명을 참고하여 판단
IC50 값을 몰 농도(M) 기준으로 변환 (예: IC50이 nM라면 값 * 1e-9로 M 단위 변환)
변환한 값을 -log10 연산하여 pIC50 계산
계산한 pIC50 값을 새로운 컬럼으로 데이터프레임에 추가 (또는 기존 IC50 열을 대체)
값 변환 후 특이값 또는 이상치가 없는지 확인 (예: IC50 값이 0이거나 매우 작으면 로그 변환시 조심)
3. 쉬운 코드 예시:
아래는 numpy를 사용하여 IC50 값을 pIC50으로 변환하는 예시입니다. IC50이 nM(나노몰 농도) 단위라고 가정하고 진행합니다.
import numpy as np

# (가정) train_df의 IC50 컬럼이 nM 단위의 IC50 값을 가지고 있음
# IC50 -> pIC50 변환: pIC50 = -log10(IC50 (M))
# nM -> M 변환: nM 값 * 1e-9

train_df['pIC50'] = -np.log10(train_df['IC50'] * 1e-9)  # IC50을 M로 바꾼 후 -log10
print(train_df[['IC50','pIC50']].head(5))  # 변환 결과 상위 5개 확인

# 필요에 따라 기존 IC50 열 삭제하고 pIC50만 남길 수도 있음
# train_df = train_df.drop('IC50', axis=1)
위 코드에서는 예시로 train_df의 IC50 값이 nM로 주어졌다고 가정하고, 이를 M로 변환한 뒤 로그 변환하여 pIC50 열을 만들었습니다. 예를 들어 IC50이 100 nM인 경우 100 * 1e-9 = 1e-7 M이고, -log10(1e-7) = 7이므로 pIC50이 7로 저장됩니다. 출력 결과로 IC50과 pIC50을 나란히 보며 제대로 변환되었는지 확인할 수 있습니다. (만약 IC50 단위가 이미 µM인 경우 1e-6을 곱하는 등 단위에 맞게 조정해야 합니다.) 변환 후에는 **pIC50 값을 모델의 목표값(target)**으로 사용하면 됩니다.
4. 머신러닝 모델 학습
1. 개념 설명:
모델 학습이란, 준비된 데이터를 이용해 머신러닝 알고리즘(기법)을 훈련시키는 과정입니다. 모델이 **특징(X)**과 **정답(y)**을 여러 사례에서 계속 맞춰보면서 패턴을 찾아내는 단계입니다. 이 경우 특징(X)은 앞 단계에서 생성한 분자 특징 벡터이고, 정답(y)은 해당 분자의 실제 pIC50 값입니다. 학습을 통해 모델은 “이런 분자 구조일 때 이런 활성을 보이는구나”라는 규칙을 스스로 익히게 됩니다. 초보자에게 친숙한 모델로는 **랜덤 포레스트(Random Forest)**나 LightGBM 등이 있습니다. 랜덤 포레스트는 **여러 개의 결정 트리(decision tree)를 앙상블(ensemble)**하여 예측을 수행하는 알고리즘입니다
geeksforgeeks.org
. 각 결정트리는 하나의 간단한 예측 모델이지만, 여러 트리를 서로 다르게 만들어 투표나 평균을 내면 더 정확한 결과를 얻을 수 있습니다
geeksforgeeks.org
. 예를 들어 회귀 문제(숫자 예측)에서는 각 트리가 예측한 값들의 평균을 내서 최종 예측값으로 사용합니다. 이렇게 하면 개별 트리보다 안정적이고 일반화 능력이 뛰어난 모델이 됩니다. LightGBM은 마이크로소프트가 개발한 그래디언트 부스팅 결정트리(GBDT) 기반의 알고리즘입니다. 여러 트리를 레벨-wise가 아니라 leaf-wise로 성장시키고, 히스토그램 최적화 등의 기법으로 매우 빠르고 효율적으로 동작하는 것이 특징입니다
lightgbm.readthedocs.io
github.com
. 쉽게 말해, LightGBM도 여러 결정트리를 이용한다는 점에서는 랜덤 포레스트와 비슷하지만, 이전 트리의 오류를 보정하는 방식으로 순차적으로 트리를 추가하여 점점 오차를 줄여나가는 부스팅(Boosting) 기법을 사용합니다. 이 모델은 속도가 빠르고 성능이 우수해서 대회에서도 많이 쓰이지만, 초보자라면 우선 구조가 상대적으로 단순한 랜덤 포레스트로 시작해 볼 수 있습니다. 요약하면, 모델 학습 단계에서는 선택한 머신러닝 알고리즘(예: 랜덤 포레스트)을 초기화하고, 훈련 데이터의 X와 y를 입력으로 주어 학습시킵니다. 그 결과 모델 내부에는 학습된 파라미터들이 세팅되고, 이 모델은 이후에 보지 못한 새로운 데이터에 대한 예측을 할 준비가 됩니다. 2. 해야 할 일:
앞서 준비한 훈련용 특징행렬 X (예: numpy 배열 또는 DataFrame)과 목표값 벡터 y (pIC50 값 시리즈)를 마련
사용할 머신러닝 모델 선택 (초보자는 RandomForestRegressor, 고급은 LightGBM Regressor 등)
모델 객체 초기화 (예: 트리 개수, 랜덤시드 등 하이퍼파라미터 설정)
모델 학습(fit): X와 y를 모델에 입력하여 학습시키기
(선택) 교차검증 등으로 모델의 성능을 검증하고 하이퍼파라미터 튜닝
최종적으로 학습된 모델 객체 확보 (이후 단계에서 예측에 사용)
3. 쉬운 코드 예시:
아래는 사이킷런(sklearn)의 랜덤 포레스트 회귀 모델을 사용하여 학습을 진행하는 간단한 코드입니다. (X_train은 훈련용 특징벡터, y_train은 그에 대응하는 pIC50 값이라고 가정합니다.)
from sklearn.ensemble import RandomForestRegressor

# 설명변수 X_train (예: 리스트의 리스트 또는 numpy 배열 형태)와 목표변수 y_train 준비
X_train = []  
for fp in train_df['fingerprint']:     # 이전 단계에서 train_df['fingerprint']에 저장했다고 가정
    X_train.append(list(fp))           # fingerprint (ExplicitBitVect)를 리스트로 변환 후 추가
X_train = np.array(X_train)            # 파이썬 리스트를 numpy 배열로 변환

y_train = train_df['pIC50'].values     # 목표값인 pIC50 배열

# 랜덤 포레스트 회귀 모델 초기화 (예: 트리 100개, 랜덤시드 고정)
model = RandomForestRegressor(n_estimators=100, random_state=42)
# 모델 학습 실행
model.fit(X_train, y_train)

# 학습이 완료되면, model 객체가 훈련된 모델을 나타냄
위 코드에서는 RandomForestRegressor를 사용했습니다. n_estimators=100은 결정트리 100개를 사용하겠다는 뜻이고, random_state=42는 결과 재현을 위한 시드 고정입니다. model.fit(X_train, y_train)을 호출하면 모델이 데이터를 학습합니다. 내부적으로는 100개의 결정트리가 각기 무작위 샘플의 데이터와 변수를 사용해 만들어지고
geeksforgeeks.org
, 평균을 내어 출력하도록 학습됩니다. 학습이 끝나면 model 객체는 훈련된 모델로서 이후 단계에서 사용됩니다. (만약 LightGBM을 쓴다면 lightgbm.LGBMRegressor를 비슷한 방식으로 사용하면 됩니다. LightGBM은 하이퍼파라미터 튜닝이 약간 필요할 수 있지만, 기본값으로도 대체로 잘 동작합니다.)
참고: 실제로 모델 성능을 높이려면 교차 검증(cross-validation)으로 검증하거나, 하이퍼파라미터(예: 트리 개수, 학습률 등)를 조정해야 합니다. 그러나 여기서는 초보자를 위한 기본 흐름 설명에 집중합니다.
5. 테스트 데이터 예측
1. 개념 설명:
모델이 훈련을 마쳤다면, 이제 새로운 데이터에 대해 예측을 할 차례입니다. 여기서 새로운 데이터란 테스트 데이터를 의미합니다. 테스트 데이터의 각 분자에 대해서는 실제 IC50 값이 알려져 있지 않으므로, 훈련된 모델로 pIC50을 예측하여 대회 제출용 결과를 만들어야 합니다. 이 단계는 마치 배운 내용을 적용해 보는 시험과 같으며, 모델의 일반화 능력을 확인하는 과정이기도 합니다. 예측 단계에서 중요한 것은, 훈련 때와 동일한 전처리 및 특징 생성 과정을 테스트 데이터에도 적용해야 한다는 것입니다. 즉, 훈련 시 SMILES를 어떻게 벡터로 바꾸었으면 테스트 시에도 같은 방식으로 벡터를 만들어야 합니다. 훈련 시 IC50을 pIC50으로 변환했다면, 예측한 pIC50을 제출 전에 다시 IC50으로 변환할지 여부도 고려해야 합니다 (대회 평가 기준에 따라 다름). 일반적으로 우리의 모델은 pIC50을 예측하도록 학습되었으므로, 출력도 pIC50 형태로 나옵니다. 만약 **제출이 IC50 값(nM)**으로 요구된다면, 예측한 pIC50를 다시 역변환해서 IC50(nM) 값으로 돌려놓아야 합니다. (pIC50 역변환: IC50 (M) = 10^(-pIC50), 그리고 단위 변환해 nM로) 한마디로, 이 단계에서는 훈련된 모델을 사용해 테스트 데이터에 대한 예측값(타깃) 을 얻는 단계이며, 얻은 예측값은 최종 제출물로 이어집니다. 2. 해야 할 일:
테스트 데이터의 SMILES 열을 훈련 때와 동일하게 특징 벡터로 변환 (예: RDKit로 Morgan 지문 생성)
변환한 테스트 특징행렬을 준비 (X_test)
훈련된 모델 (model)을 사용하여 X_test에 대한 예측값 계산 (model.predict)
예측 결과는 모델이 학습한 타깃 스케일(pIC50)일 것이므로, 필요 시 단위를 되돌림 (대회 요구사항에 따라 pIC50 그대로 제출하거나 IC50로 변환)
최종 제출을 위해 예측값과 해당 ID를 매칭
3. 쉬운 코드 예시:
아래는 앞서 학습한 model을 이용하여 test_df의 분자들에 대해 pIC50을 예측하는 과정의 예시입니다. (X_test는 테스트 SMILES를 벡터화한 결과이고, 제출은 IC50(nM)로 한다고 가정해 변환도 함께 보여줍니다.)
# 테스트 데이터의 SMILES를 훈련과 동일한 방식으로 벡터화
X_test = []
for smi in test_df['SMILES']:
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    X_test.append(list(fp))
X_test = np.array(X_test)

# 훈련된 모델로 예측 수행 (출력은 pIC50 값들의 배열)
predicted_pIC50 = model.predict(X_test)

# 필요하다면 pIC50 -> IC50(nM)로 변환 (예: 제출이 IC50(nM) 요구일 경우)
# pIC50 = -log10(IC50(M)) 이므로, IC50(M) = 10^(-pIC50)
# M 단위를 nM로 바꾸려면 *1e9
predicted_IC50_nM = 10**(-predicted_pIC50) * 1e9

# 예측 결과 상위 5개 출력 (pIC50과 IC50(nM) 나란히 확인)
for i in range(5):
    print(f"Pred pIC50: {predicted_pIC50[i]:.3f},  Pred IC50: {predicted_IC50_nM[i]:.2f} nM")
위 코드에서는 test_df['SMILES']에 대해 반복문을 돌며 RDKit을 이용해 Morgan 지문 (fp)을 계산하고, 리스트로 변환하여 X_test에 모았습니다. 그런 다음 model.predict(X_test)로 예측을 수행하면 각 분자의 예상 pIC50 값이 predicted_pIC50 배열로 얻어집니다. 이후 예시는 만약 제출 요구사항이 IC50 (나노몰) 값이라면, 예측된 pIC50을 IC50으로 되돌린 것입니다. 식을 통해 pIC50 -> IC50(M) -> IC50(nM) 변환을 했습니다. 변환 결과를 소수점으로 출력해 보면 예측된 pIC50가 7.00일 때 IC50 약 100 nM 수준으로 잘 변환되는 것을 확인할 수 있습니다. (대회에 따라 제출해야 하는 값이 pIC50인 경우도 있으므로, 해당 대회의 지시에 따라 이 변환은 생략하거나 수행해야 합니다.)
6. 제출 파일 생성
1. 개념 설명:
이제 최종적으로 제출 파일을 만들어야 합니다. 제출 파일은 대회 주최측에서 정한 양식에 맞게, 우리가 예측한 결과를 담은 파일입니다. 일반적으로 CSV 파일 형식으로, 각 테스트 샘플의 **식별자(ID)**와 예측한 값을 컬럼으로 포함합니다. 예를 들어 id, IC50 형태로 각 줄에 테스트 화합물의 ID와 우리가 예측한 IC50값을 써넣는 식입니다. 제출 파일을 만드는 이유는, 대회 시스템이 이 파일을 받아서 정답과 비교하여 점수를 계산하기 위해서입니다. 중요한 점은 제출 파일의 **형식(format)**을 정확히 지키는 것입니다. 열 이름, 순서, 파일명 등이 대회 규칙에 명시된 대로여야 하며, 특히 ID와 예측값이 올바르게 매칭되도록 주의해야 합니다. 예측 단계에서 구한 predicted_IC50_nM (또는 predicted_pIC50) 배열의 순서가 테스트 데이터의 순서와 정확히 같다고 가정하고, 해당 순서대로 ID와 연결해야 합니다. 보통 테스트 데이터에 ID 열이 있거나, 테스트 데이터가 원래 순서대로 평가되므로, 우리가 읽어왔던 test_df의 순서를 그대로 사용하면 됩니다. 마지막으로, 준비된 DataFrame을 to_csv() 함수 등을 사용하여 CSV 파일로 저장하면 제출 준비가 완료됩니다. 이 파일을 대회 플랫폼에 업로드하여 결과를 제출하면 됩니다. 2. 해야 할 일:
테스트 데이터의 식별자(ID) 열을 준비 (예: test_df['ID'] 또는 인덱스)
예측된 값 배열을 준비 (이전 단계에서 얻은 predicted_values)
두 정보를 하나의 데이터프레임으로 합치거나 컬럼으로 넣기 (ID와 예측값을 열로 갖는 데이터프레임 생성)
요구되는 컬럼 이름을 정확히 맞추기 (예: ID, IC50 혹은 Predicted 등 대회에서 지정한 이름)
DataFrame을 CSV로 저장 (to_csv), 인덱스는 포함하지 않도록 설정 (index=False)
CSV 파일 내용이 형식에 맞게 잘 작성되었는지 최종 확인 (헤더 존재 여부 등)
3. 쉬운 코드 예시:
아래는 예측 결과를 submission.csv 파일로 저장하는 예시입니다. test_df에 ID라는 컬럼이 있고 우리가 예측한 IC50(nM) 값이 있다고 가정합니다.
# test_df에 ID 컬럼과, predicted_IC50_nM 배열이 준비되어 있다고 가정
submission_df = pd.DataFrame({
    'ID': test_df['ID'],             # ID 열
    'IC50': predicted_IC50_nM       # 우리 모델이 예측한 IC50 값 (단위: nM)
})

# 지정된 형식에 맞게 CSV 파일로 저장 (index는 포함하지 않음)
submission_df.to_csv('submission.csv', index=False)

print("Submission file saved:", submission_df.shape)
print(submission_df.head(5))  # 제출 파일의 처음 몇 줄 미리보기
위 코드에서 submission_df는 두 개의 열을 가집니다: ID와 IC50. 각각 테스트 세트의 ID와 우리가 예측한 IC50 값을 담고 있습니다. to_csv('submission.csv', index=False)로 저장하면 제출 양식에 맞는 CSV 파일이 만들어집니다. 예를 들어 출력 미리보기를 보면 아래와 같을 것입니다 (예시):
ID,IC50  
1, 250.0  
2, 500.5  
3, 123.4  
...  
이런 형태로 각 행이 ID와 예측 IC50 값을 나열한 파일을 제출하면, 대회 서버에서 우리 예측값(pIC50을 다시 nM로 변환한 값)을 정답과 비교하여 점수를 계산하게 됩니다. 제출 전에 열 이름 철자나 대소문자 등이 요구사항과 일치하는지 반드시 확인하세요. 잘못된 형식으로 제출하면 채점이 안 될 수도 있습니다.
