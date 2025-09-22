# Commerce Purchase Behavior Prediction | 커머스 상품 구매 예측

본 프로젝트는 docs/competition-info.md의 대회 내용을 기반으로,
이커머스 환경에서의 구매 예측 추천 시스템(다음 1주 내 구매 아이템 추천)을 구현·실험하기 위한 템플릿입니다. 
대회 개요, 평가 지표 및 제출 형식 등 핵심 안내는 아래 문서를 먼저 확인하세요.

- 대회 소개 및 평가: docs/competition-info.md
- uv 패키지 관리자 사용법: docs/how-to-use-uv.md

## Team

| 이름  | 역할 |
|:---:|:--:|
| 이나경 | 팀장 |
| 김두환 | 팀원 |
| 조의영 | 팀원 |
| 박성진 | 팀원 |
| 편아현 | 팀원 |

## 0. Overview

- 목적: 사용자별 상위 10개 추천 아이템을 산출하여 NDCG@10을 최대화
- 범위: 데이터 전처리 → 피처 엔지니어링 → 모델링(전통/딥러닝) → 추론/제출 파이프라인
- 산출물: 추천 결과 제출 파일, 분석 노트북, 실험 로그

### Environment

- Python 3.11.x (>=3.11,<3.12)
- 패키지 관리: uv
- OS: macOS/Linux/Windows

### Requirements

핵심 라이브러리

- numpy, pandas, scikit-learn, seaborn, matplotlib, torch
- 환경변수 관리: python-dotenv

개발 편의(옵션)

- ipykernel, ipywidgets, pre-commit, ruff

자세한 버전 범위는 pyproject.toml을 참고하세요.

## 1. Competition Info

- 목적: 사용자의 과거 행동(쇼핑 패턴, 관심사, 구매 이력 등)을 분석해 다음 1주 내 구매할 가능성이 높은 상품을 추천
- 과제: 모든 유저에 대해 중복 없이 정확히 10개 아이템을 추천해야 채점 대상이 됨
- 데이터 활용: 적절한 전처리와 피처 엔지니어링을 수행하고, PyTorch 및 기존 라이브러리를 활용해 모델을 구축
- 평가 지표: NDCG@10 (Binary relevance 기반)
    - Relevance는 실제 구매 여부에 따라 1(구매), 0(미구매)
    - DCG@K는 랭크가 낮을수록 할인되며, IDCG@K로 정규화해 NDCG@K 계산
    - 점수가 클수록 성능이 우수하며, 동점일 경우 제출 횟수가 적은 팀이 상위
- 제출 요건:
    - 모든 유저(예: 638,257명)에 대해 아이템 10개를 중복 없이 추천
    - 각 유저-아이템-랭크 형태의 제출 포맷을 유지하고 정렬·중복 여부를 검증
- 전략 참고:
    - 대회 특성상 점수 극대화를 위해 현업 대비 복잡한 구조(예: 다중 모델 앙상블)를 활용하는 것도 가능

더 자세한 사항은 아래 문서를 참고해주세요.

- [competition-info.md](docs/competition-info.md)

## 2. Components

### Directory

프로젝트 구조

```
.
├── README.md
├── data
│     └── fonts
│         └── NanumBarunGothic.ttf
├── docs
│     ├── competition-info.md
│     └── how-to-use-uv.md
├── notebooks
│     └── notebook_template.ipynb
├── pyproject.toml
├── ruff.toml
├── src
│     └── awesome-rec-sys
│         └── __init__.py
└── uv.lock
```

- src/awesome-rec-sys: 패키지 소스 루트
- notebooks/: 탐색/실험용 노트북 템플릿
- docs/: 대회 및 개발 환경 가이드
- data/fonts/NanumBarunGothic.ttf: 시각화 시 한글 폰트 설정용(예: matplotlib)

## 3. Quickstart

uv로 의존성 설치

```bash
# 프로젝트 루트에서
uv sync
# 개발용 의존성까지 설치(원한다면)
uv sync --extra dev
```

한글 폰트 설정(선택)

- 시각화에서 한글 깨짐이 있을 경우 data/fonts의 폰트를 matplotlib에 등록해 사용하세요.

환경 변수

- `.env` 를 사용한다면 `python-dotenv` 로 불러올 수 있습니다.

## 4. Data description

- 데이터 스키마/EDA/전처리 전략은 팀 논의 결과에 맞춰 notebooks/notebook_template.ipynb를 복제해 실험을 기록하세요.
- 전처리 체크리스트
    - 결측/이상치 처리
    - 사용자·아이템 단위 통계 피처화
    - 시계열 스플릿 또는 홀드아웃 전략 정의
    - 학습/검증 분리 로직 일관성 유지

## 5. Modeling

- 베이스라인 아이디어
    - 빈도/인기도 기반 추천
    - 협업 필터링(행렬분해, implicit 신호 활용)
    - 랭킹 학습(LTR) 또는 딥러닝(Embedding+MLP, 시퀀스 모델 등)
- 평가 루프
    - 검증 세트에서 NDCG@K 산출
    - 후보군 생성(retrieval) → 재정렬(rerank) 2단계 구조 고려
- 제출
    - 모든 유저에 대해 중복 없는 10개 추천 생성
    - 형식·정렬·중복 여부 확인 후 제출 파일 생성

## 6. Dev guide

코드 스타일 & 린팅

```bash
uv run ruff check .
uv run ruff format .
```

pre-commit(선택)

```bash
uv run pre-commit install
uv run pre-commit run -a
```

패키지 추가/제거

```bash
uv add 패키지명
uv remove 패키지명
```

## 7. Result

- 리더보드 스크린샷, 점수 및 실험 로그를 주기적으로 기록하세요.
- 발표 자료가 있을 경우 링크를 여기에 추가하세요.

## etc

- 미팅 로그/결정 사항은 팀 협업 도구(예: Notion/Google Docs) 링크로 정리하세요.
