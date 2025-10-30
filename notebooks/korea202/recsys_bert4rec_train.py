import pandas as pd
import os
import numpy as np
import math
import random
import torch
import json
from tqdm import tqdm
from collections import defaultdict

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import BERT4Rec, SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start.quick_start import load_data_and_model


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def get_config():
    config_dict = {
        'data_path': './data',  # 데이터셋 폴더가 들어있는 상위 경로입니다.
        'USER_ID_FIELD': 'user_idx',  # 사용자 ID가 저장된 컬럼명입니다.
        'ITEM_ID_FIELD': 'item_idx',  # 아이템(상품, 책 등) ID가 저장된 컬럼명입니다.
        'TIME_FIELD': 'event_time',  # 상호작용(클릭, 구매 등) 시각이 저장된 컬럼명입니다.
        'user_inter_num_interval': "[5,Inf)",  # 최소 5번 이상 상호작용한 사용자만 남깁니다.
        'item_inter_num_interval': "[5,Inf)",  # 최소 5번 이상 등장한 아이템만 남깁니다.
        'load_col': {'inter': ['user_idx', 'item_idx', 'user_session', 'event_time', 'event_type'],
                    'item': ['item_idx', 'category_code', 'brand', 'price']
                    },  # 불러올 컬럼을 명시합니다.

        'train_batch_size': 4096,  # 학습 시 한 번에 처리할 데이터 샘플 수입니다.
        'hidden_size': 256,  # 임베딩 및 내부 레이어의 차원 수입니다.
        'n_layers': 2,  # 모델의 레이어(층) 개수입니다.
        'n_heads': 4,  # Self-Attention에서 사용하는 헤드 개수입니다.
        'inner_size': 64,  # Feedforward 네트워크의 내부 차원입니다.
        'hidden_dropout_prob': 0.2,  # 은닉층에 적용할 드롭아웃 비율입니다.
        'attn_dropout_prob': 0.2,  # 어텐션 레이어에 적용할 드롭아웃 비율입니다.
        'hidden_act': 'gelu',  # 은닉층에서 사용할 활성화 함수입니다.
        'layer_norm_eps': 1e-12,  # Layer Normalization에서 사용하는 작은 상수입니다.
        'initializer_range': 0.02,  # 가중치 초기화 시 표준편차입니다.
        'pooling_mode': 'sum',  # 시퀀스 임베딩을 합산(sum) 방식으로 합칩니다.
        'loss_type': 'BPR',  # 학습에 사용할 손실 함수(BPR: 순위 기반 추천)입니다.
        'fusion_type': 'gate',  # 여러 정보를 결합할 때 게이트 방식을 사용합니다.
        'attribute_predictor': 'linear',  # 속성 예측에 사용할 방식을 지정합니다.
        #'epoch': 2,  # 전체 데이터셋을 몇 번 반복해서 학습할지 지정합니다.
        'epochs' : 10, 
        'stopping_step': 5,  # 검증 성능이 5번 연속 개선되지 않으면 학습을 멈춥니다.

        'MAX_ITEM_LIST_LENGTH': 50,  # 사용자별로 최대 50개까지의 시퀀스만 사용합니다.
        'eval_args': {
            'split': {'LS': 'valid_and_test'},  # Leave-Sequence 방식으로 검증/테스트 분할
            'group_by': 'user',  # 사용자별로 데이터를 그룹화해서 평가합니다.
            'order': 'TO',  # 시간순(Time Order)으로 정렬해서 분할합니다.
            'mode': 'uni100'  # 테스트 시 각 정답마다 100개의 negative 아이템을 샘플링합니다.
        },
        'metrics': ['Recall', 'NDCG'],  # 평가 지표로 Recall과 NDCG를 사용합니다.
        'topk': 10,  # 상위 10개 추천 결과만 평가에 사용합니다.
        'valid_metric': 'NDCG@10',  # 검증 기준으로 NDCG@10을 사용합니다.
        # 'checkpoint_dir': '/content'  # (주석 처리됨) 모델 체크포인트 저장 디렉토리입니다.
    }

    return Config(model='SASRec',
                    config_dict=config_dict,
                    dataset='sasrec_data')


def get_dataset(config):
    
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)

    # 데이터셋 전체 정보 출력
    print("Dataset Info:")
    print(dataset)
    print(f"\n데이터셋 이름: {dataset.dataset_name}")
    print(f"사용자 수: {dataset.user_num}")
    print(f"아이템 수: {dataset.item_num}")
    print(f"상호작용 수: {dataset.inter_num}")

    return data_preparation(config, dataset)

def train(train_data, valid_data, config):

    # model을 불러옵니다.
    model = SASRec(config, train_data.dataset).to(config['device'])
    print("model information : ", model)

    # trainer를 초기화합니다.
    trainer = Trainer(config, model)

    # model을 학습합니다.
       # model을 학습합니다.
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])

    print(best_valid_score, best_valid_result)

def main():
    
    # 시드 설정
    set_seed(42)
    # config 설정
    config = get_config()
    # 데이타셋 가져옴
    train_data, valid_data, _ = get_dataset(config)
    # 훈련
    train(train_data, valid_data, config)


if __name__ == "__main__":
    main()