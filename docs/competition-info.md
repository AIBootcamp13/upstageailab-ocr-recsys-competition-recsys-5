## 소개

이번 'Commerce Behavior Purchase Prediction' 대회는 사용자의 쇼핑 패턴을 분석하여 미래 (next one week)에 사용자가 구매할 상품을 추천하는 것을 목적으로 합니다. 수많은
제품 중에서 적합한 제품을 찾는 데 어려움을 겪는 사용자들에게, 추천 시스템은 개인의 쇼핑 습관, 관심사, 과거 구매 이력 등을 분석해 맞춤형 상품을 추천할 수 있습니다. 따라서, 이커머스 분야에서 추천 시스템은
사용자의 취향을 분석하여 알맞은 상품을 추천함으로써 사용자의 경험을 증진하고 기업의 매출 향상에 도움을 줄 수 있습니다.

이커머스 추천 시스템을 구축하는 것은 알맞은 데이터 전처리에서부터 시작하여, 목적에 맞는 모델을 선택하고, PyTorch 및 기존 라이브러리를 활용하여 모델을 구축하고, Feature Engineering 및 예측을
수행하는 전반적인 과정을 포함합니다. 추천 시스템 대회의 특성을 고려해서 평가 지표에 최적화된 파이프라인을 개발해보세요!

참고로 현업과 달리, 대회에서는 더 높은 점수를 얻기 위해서 실제 현업에서는 쓰기 어려운 구조를 채택하는 것도 고려할 수 있습니다! (e.g., 10개 이상의 모델의 앙상블)

## 평가방법

평가 지표는 Binary relevance 기반의 NDCG@10을 사용합니다. 해당 지표는 검색 엔진의 품질 평가에 사용되는 지표로, 최근 들어서는 추천 시스템에도 널리 활용되고 있습니다.

Relevance는 ground-truth set에 따라 1 (실제 구매) 또는 0 (구매하지 않음)으로 나뉩니다 (binary relevance).

즉, test set의 해당 user가 예측된 item을 구매했으면 1, 아니면 0을 할당합니다.

NDCG@10이 클 수록, NDCG@10 값이 동일하다면 제출 횟수가 적을 수록 높은 순위가 할당됩니다.

train 데이터의 모든 유저(638,257명)에게 10개씩 중복 없이 item을 추천해야만 채점이 진행됩니다.

| Predicted | Predicted | Predicted | True      |
|-----------|-----------|-----------|-----------|
| user_id   | item_id   | rank      | relevance |
| A         | 1         | 1         | 0         |
| A         | 2         | 2         | 1         |
| A         | 3         | 3         | 0         |

예시) user1에 대해 3개의 아이템 예측 및 nDCG@3 계산

DCG@3 = 0/log(1+1) + 1/log(2+1) + 0/log(3+1)
IDCG@3 = 1/log(1+1) + 1/log(2+1) + 1/log(3+1)
NDCG@3 = DCG@3/IDCG@3

추가자료:  https://en.wikipedia.org/wiki/Discounted_cumulative_gain

## 데이터

### 학습 데이터셋 개요

학습데이터셋에는 온라인 스토어 유저의 행동 데이터를 담고 있습니다.
기간은 약 4개월로 데이터엔 여러 유저와 아이템 아이디 뿐 아니라 이벤트 정보(구매, 카트에 넣기, 조회)와 아이템의 정보도 포함되어 있습니다.

- 2019년 11월 1일부터 2020년 2월 29일까지 4개월간의 데이터
- 사용자(user_id)는 쇼핑몰에 들어갈 때, 세션(user_session)를 할당 받습니다. 사용자의 행동은 다음과 같이 기록됩니다.
- 사용자는(user_id)는 특정 아이템(item_id)을 특정 시간(event_time)에 상품(product_id) 장바구니에 추가(event_type = 'cart')하거나 조회(event_type = '
  view')하거나 구매(event_type = 'purchase')합니다. 이때, 각 상품 별로 (해당 시점에 따른) 카테고리 코드(category_code)와 브랜드(brand), 가격(price)가
  주어집니다.

- 8,350,311개의 행으로 이루어져 있습니다.
- `user_id`: 유저 id
- `item_id`: 아이템 id
- `user_session`: 사용자의 세션 ID. 사용자가 오랜 일시 중지 후 온라인 스토어로 돌아올 때마다 변경됩니다.
- `event_time`: 이벤트가 일어난 시각(UTC기준)
- `category_code`: 아이템의 카테고리 분류입니다.
- `brand`: 아이템의 brand
- `price`: 아이템의 가격
- `event_type`: 이벤트의 종류

### 평가 데이터 개요

- 2020년 3월 1일부터 20년 3월 7일까지 일주일 간의 데이터입니다.
- 해당 기간 동안 유저가 구입한(event_type = 'purchase') 아이템 이력에 대한 데이터로 `user_id` 와 `item_id` 로 구성됩니다.
- 평가데이터는 무작위 (50:50 random split)로 public, private dataset으로 나눴습니다.
- 평가 데이터에는 학습데이터에 포함된 유저와 아이템으로만 이뤄져 있습니다.

### 평가과정

1) 학습데이터에 포함된 모든 유저(user_id)당 10개의 아이템(item_id)을 예측합니다.
2) 해당 데이터를 제출하면, 평가데이터에(public dataset, private dataset) 포함된 유저의 구매이력 데이터 기준으로 평가를 진행합니다.

##### sample_submission.csv

- 6,382,570개의 행으로 이루어져 있습니다.
- `user_id`: 예측하고자 하는 user의 id 입니다.
- `item_id`: user에게 추천할 item의 id 입니다.

- user 별 item_id의 순서는 compatibility score에 따라 내림차순 정렬되어야 합니다.
- 즉, predicted score가 가장 높은 item_id가 처음에 위치해야 합니다.
- sample_submission에서는 랜덤하게 10개의 item이 들어있습니다.