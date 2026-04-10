# IMBK_Bank_Customer_Churn_ML

# 프로젝트 명: 고객 이탈 분류 ML 및 인사이트 분석

# 기간: 2026.04.10

# 기술스택:
import pandas as pd # pandas 사용
import numpy as np # numpy 사용

from sklearn.model_selection import train_test_split # train_test_split 사용
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier # RandomForestClassifier, GradientBoostingClassifier, StackingClassifier 사용
from sklearn.metrics import f1_score # f1_score 사용
from sklearn.preprocessing import LabelEncoder, StandardScaler # LabelEncoder, StandardScaler 사용

from sklearn.ensemble import RandomForestClassifier, StackingClassifier # RandomForestClassifier, StackingClassifier 사용
from xgboost import XGBClassifier # XGBClassifier 사용
from lightgbm import LGBMClassifier # LGBMClassifier 사용
from sklearn.linear_model import LogisticRegression # LogisticRegression 사용

import optuna # optuna 사용
import shap # shap 사용
import matplotlib.pyplot as plt # matplotlib.pyplot 사용
import seaborn as sns # seaborn 사용
import platform # platform 사용

# 데이터 출처: 캐글 Bank Customer Churn Dataset (row: 10000, col:12)

# 데이터 전처리:
# 전처리
customer_id를 제거했는데, 그 이유는 아이디는 단순한 일련번호라서 고객의 이탈 여부와는 아무런 상관이 없다고 판단했습니다. 만일 아이디를 지우지 않고 그대로 모델에 학습했다면 우연을 패턴으로 착각하여 Overfitting이 올 수 있기 때문에 제거하였습니다.

그리고 범주형 인코딩을 진행하였는데, 이는 머신러닝이 문자열 데이터를 계산할 수 없기 때문에 숫자형으로 변환하는 전처리 과정을 진행하였습니다.

또한 나이를 이용하여 이탈률을 확인하였는데, 그 이유는 나이가 연속형 변수라 꺽은선 그래프를 활용하여 전체적인 추세와 흐름을 파악할 수 있기 때문에 가장 적합하다고 판단하였습니다.

stratify=y를 적용하였는데, 이는 8:2로 쪼갤 때 비율을 똑같이 유지해서 나누어야 모델이 공정하게 학습할 수 있기 때문에 적용하는 게 좋다고 생각하였습니다.

마지막으로 스케일링을 사용한 이유는 모든 변수가 동일한 조건에서 진행을 해야 학습에 지장이 없기 때문이고, 또한 fit_transform을 통해 데이터의 누수를 방지할 수 있기에 스케일링을 적용해야 한다고 판단하였습니다.

# EDA 및 해석:
<img width="844" height="547" alt="image" src="https://github.com/user-attachments/assets/33e2da02-4e20-4c41-a2d4-b3e3106612a9" />

# EDA 해석
이 꺽은선 그래프를 보면, 20대인 청년층은 평균 이탈률이 적게 나타나고 있지만 40대에서 60대 사이인 중장년층은 평균 이탈률이 크게 나타나고 있다는 점을 알 수 있습니다.

아마 쳥년층 고객들은 주거래 은행을 잘 바꾸지 않고 이용하고 있지만, 50대 전후인 고객들은 은퇴 준비와 자녀 결혼 등 큰 자금이 필요할 수 있기 때문에 더 높은 예금 금리나 유리한 대출 조건을 찾아 주거래 은행을 바꿀 가능성이 크다고 볼 수 있다고 판단하였습니다.

그리고 80대에서 90대 사이에 튀는 구간이 보이는데, 이는 아마 표본 수가 적어 발생하는 데이터 노이즈라고 생각합니다.

# AutoML – Hyperparameter Tuning – Stacking Pipe – Shap value:

# 사용한 AutoML
Random Forest, LightGBM, XGBoost, Gradient Boosting :/
<img width="335" height="141" alt="image" src="https://github.com/user-attachments/assets/c41ef37e-0d43-424e-9600-36366d9348d0" />

# ML 선정 기준
먼저 Pycaret을 이용해, 수많은 머신러닝 모델 중에 어떤 모델을 사용하면 좋을지 1차적으로 선별을 하였습니다. 그 중에서 F1_score가 괜찮게 나온 상위의 모델 중에 제가 배우고 익숙한 4개의 모델을 선정하였습니다.

그 다음으로, 선정한 4개의 모델들을 성능을 올리기 위해 Optuna를 사용해 최적을 조합을 찾으려고 하였습니다.

이렇게 두 차례에 걸쳐 단일 모델 성능을 뽑아냈고, 그 결과 RandomForest는 0.5813, LightGBM는 0.5938, XGBoost는 0.5763, GradientBoosting는 0.6057라는 성능을 알 수 있었습니다.

# 인사이트 제안:
빨간색 점은 나이가 많은 고연령층을 뜻하고, 파란색 점은 나이가 적은 저연령층을 말합니다.

그리고 오른쪽인 (+) 방향으로 간다면 이탈을 높이는 요인이라는 뜻이고, (-)로 간다면 이탈을 막아주는 요인이라고 볼 수 있습니다.

위에 정보를 종합해 그래프를 분석하자면, 나이가 많을수록 이탈 할 확률이 높아지므로, 앞서 말한 가설과 일치하고 있습니다.

이와 같은 이탈을 방지하려면, 우선 청년층은 특별한 조치를 취하지 않아도 주거래 은행을 바꾸는 일이 거의 없기 때문에 괜찮지만 중장년층은 특히 조심해야 합니다.

중장년층 고객 유지를 위해 "은퇴 자금 우대 금리" 또는 건강을 타겟팅한 "건강 검진 제휴 혜택" 등 다양한 혜택과 상품을 활용해 고객 이탈을 방지하는 것이 좋은 제안이라고 생각합니다.

또한, 단순히 더 좋은 혜택이 있어서 이탈하는 게 아니라, 중장년층이 모바일 뱅킹 앱을 사용하기 어려워 하는 게 아닌지 확인하는 것도 좋은 방법일 수 있습니다.

최대한 글씨를 키우고, 메뉴를 단순화한 "중장년층 전용 모바일 앱" 이나 "중장년층 전용 모바일 기능"을 도입하는 것 또한 괜찮은 인사이트로 작용할 수 있습니다.
