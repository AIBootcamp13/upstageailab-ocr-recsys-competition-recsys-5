import pandas as pd
import os
import numpy as np
import math
import random
import torch
import json
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import BERT4Rec, SASRec
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, get_model
from recbole.utils.case_study import full_sort_topk
from recbole.data.interaction import Interaction
from recbole.quick_start.quick_start import load_data_and_model


def crack_load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file, weights_only=False)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data

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


def get_bpr_config():
    config_dict = {
        'data_path': './data',  # 데이터셋 폴더가 들어있는 상위 경로입니다.
        'USER_ID_FIELD': 'user_idx',  # 사용자 ID가 저장된 컬럼명입니다.
        'ITEM_ID_FIELD': 'item_idx',  # 아이템(상품, 책 등) ID가 저장된 컬럼명입니다.
        'TIME_FIELD': 'event_time',  # 상호작용(클릭, 구매 등) 시각이 저장된 컬럼명입니다.
        'user_inter_num_interval': "[5,Inf)",  # 최소 5번 이상 상호작용한 사용자만 남깁니다.
        'item_inter_num_interval': "[5,Inf)",  # 최소 5번 이상 등장한 아이템만 남깁니다.
        
        #'load_col': {'inter': ['user_idx', 'item_idx', 'user_session', 'event_time', 'event_type'],
        #            'item': ['item_idx', 'category_code', 'brand', 'price']
        #            },  # 불러올 컬럼을 명시합니다.
        'load_col': {'inter': ['user_idx', 'item_idx', 'event_time', 'view_cnt']
                    },  # 불러올 컬럼을 명시합니다.
        'train_batch_size': 4096,  # 학습 시 한 번에 처리할 데이터 샘플 수입니다.
        'epochs' : 1, 
        'stopping_step': 5,  # 검증 성능이 5번 연속 개선되지 않으면 학습을 멈춥니다.
        'MAX_ITEM_LIST_LENGTH': 50,  # 사용자별로 최대 50개까지의 시퀀스만 사용합니다.
        'eval_args': {
            'split': {'LS': 'valid_and_test'},  # Leave-Sequence 방식으로 검증/테스트 분할
            'group_by': 'user',  # 사용자별로 데이터를 그룹화해서 평가합니다.
            'order': 'TO',  # 시간순(Time Order)으로 정렬해서 분할합니다.
            'mode': 'full'  # 테스트 시 각 정답마다 100개의 negative 아이템을 샘플링합니다.
        },
        'metrics': ['Recall', 'NDCG'],  # 평가 지표로 Recall과 NDCG를 사용합니다.
        'topk': 10,  # 상위 10개 추천 결과만 평가에 사용합니다.
        'valid_metric': 'NDCG@10',  # 검증 기준으로 NDCG@10을 사용합니다.
        'loss_type': 'CE',
        # 'checkpoint_dir': '/content'  # (주석 처리됨) 모델 체크포인트 저장 디렉토리입니다.
        'saved_model_file': './saved/BPR.pth'
    }

    return Config(model='BPR',
                    config_dict=config_dict,
                    dataset='sasrec_data')


def get_config():
    config_dict = {
        'data_path': './data',  # 데이터셋 폴더가 들어있는 상위 경로입니다.
        'USER_ID_FIELD': 'user_idx',  # 사용자 ID가 저장된 컬럼명입니다.
        'ITEM_ID_FIELD': 'item_idx',  # 아이템(상품, 책 등) ID가 저장된 컬럼명입니다.
        'TIME_FIELD': 'event_time',  # 상호작용(클릭, 구매 등) 시각이 저장된 컬럼명입니다.
        'user_inter_num_interval': "[5,Inf)",  # 최소 5번 이상 상호작용한 사용자만 남깁니다.
        'item_inter_num_interval': "[5,Inf)",  # 최소 5번 이상 등장한 아이템만 남깁니다.
        
        #'load_col': {'inter': ['user_idx', 'item_idx', 'user_session', 'event_time', 'event_type'],
        #            'item': ['item_idx', 'category_code', 'brand', 'price']
        #            },  # 불러올 컬럼을 명시합니다.
        'load_col': {'inter': ['user_idx', 'item_idx', 'event_time', 'view_cnt']
                    },  # 불러올 컬럼을 명시합니다.
        'train_batch_size': 4096,  # 학습 시 한 번에 처리할 데이터 샘플 수입니다.
        'embedding_size': 128, 
        'hidden_size': 128,  # 임베딩 및 내부 레이어의 차원 수입니다.
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
        'epochs' : 1, 
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

    train_data, valid_data, test_data = data_preparation(config, dataset)
    return train_data, valid_data, test_data, dataset


def bpr_train(train_data, valid_data, config):

    # model을 불러옵니다.
    model = BPR(config, train_data.dataset).to(config['device'])
    print("model information : ", model)

    # trainer를 초기화합니다.
    trainer = Trainer(config, model)
    trainer.saved_model_file = config['saved_model_file']

    # model을 학습합니다.
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])
    
    print(best_valid_score, best_valid_result)

    return model

def train(train_data, valid_data, config):

    # model을 불러옵니다.
    model = SASRec(config, train_data.dataset).to(config['device'])
    print("model information : ", model)

    # trainer를 초기화합니다.
    trainer = Trainer(config, model)

    # model을 학습합니다.
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])

    print(best_valid_score, best_valid_result)

    return model


def get_dataframe():
    
    train_df = pd.read_csv('./data/prepared_df.csv', sep='\t')

    # 사용자(user)와 아이템(item)을 인덱스로 매핑하기 위한 딕셔너리 생성
    idx2user = {k: v for k, v in enumerate(train_df['user_id'].unique())}  # 각 인덱스를 사용자로 매핑
    idx2item = {k: v for k, v in enumerate(train_df['item_id'].unique())}  # 각 인덱스를 아이템으로 매핑

    print(train_df.head()) 

    return  train_df, idx2user, idx2item  

def bpr_inference(train_df, idx2user, idx2item, config, model, dataset, test_data):

    # 사용자별 시간별 아이템 정렬
    train_df = train_df.sort_values(by=['user_session','event_time'])
    users = defaultdict(list) # defaultdict은 dictionary의 key가 없을때 default 값을 value로 반환
    for u, i in zip(train_df['user_idx'], train_df['item_idx']):
        users[u].append(i)

    
    # 추천 상품 생성/저장 
    # cold-start user는 popular_top_10 items으로 make-up
    # groupby('item_idx') 가 쿼리의 인덱스가 된다.
    popular_top_10 = train_df.groupby('item_idx').count().rename(columns = {"user_idx": "user_counts"}).sort_values(by=['user_counts', 'item_idx'], ascending=[False, True])[:10].index
   

    user_predicted_items = defaultdict(list)   # user별 추천 아이템 목록용

    # short history user에 대해선 popular로 처리
    for uid in tqdm(users):
        if str(uid) in dataset.field2token_id['user_idx']:
            recbole_id = dataset.token2id(dataset.uid_field, str(uid))
            topk_score, topk_iid_list = full_sort_topk([recbole_id], model, test_data, k=100, device=config['device'])
            predicted_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
            predicted_item_list = predicted_item_list[-1]
            predicted_item_list = list(map(int,predicted_item_list))
        else: # cold-start users
            predicted_item_list = []

        
        user_predicted_items[uid] = predicted_item_list

    #print(f"user_predicted_items={user_predicted_items}")
    return user_predicted_items, popular_top_10

def inference(user_predicted_items, train_df, idx2user, idx2item, config, model, dataset, test_data, popular_top_10):

    print(f"popular_top_10=f{popular_top_10}")

    # 1. user_idx, event_time 기준으로 오름차순(과거→최신) 정렬
    train_df = train_df.sort_values(by=['user_idx', 'event_time'], ascending=[True, True])

    # 2. 사용자별로 마지막 51개(row)만 추출한 뒤, 마지막 1개를 제외(즉, 50개만 남김)
    def last_50_exclude_last(group):
        return group.tail(51).iloc[:-1]

    remove_last_df = train_df.groupby('user_idx', group_keys=False).apply(last_50_exclude_last).reset_index(drop=True)


    result = []

    count = 0

    for uid, item_list in tqdm(user_predicted_items.items()):

        count = count +1

        if item_list:
            print(f"item list={len(item_list)}", flush=True)
            item_scores = []
        
            for iid in item_list:

                user_df = remove_last_df[remove_last_df['user_idx'] == uid]
                item_idx_list = user_df['item_idx'].tolist()
                print(f"count={len(item_idx_list)}")
                print(item_idx_list)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                interaction_data = {
                    'user_idx': torch.tensor([uid]),
                    'item_idx': torch.tensor([iid]),
                    'item_idx_list' : torch.tensor([item_idx_list]),
                    'item_length' : torch.tensor([len(item_idx_list)]),

                }
                interaction = Interaction(interaction_data)    

                #score = model.predict(uid, iid)  # pseudo code
                score = model.predict(interaction.to(device))  # pseudo code
                item_scores.append((iid, score))

            # 점수 기준 내림차순 정렬
            item_scores.sort(key=lambda x: x[1], reverse=True)
            # 상위 10개만 사용
            top_10_items = item_scores[:10]
            # (user_id, item_id) 쌍을 result 에 저장
            for iid, score in top_10_items:
                result.append((idx2user[uid], idx2item[iid]))

        else:
            for iid in popular_top_10:
                result.append((idx2user[uid], idx2item[iid]))

              
    print(f"final count={count}") 
    print(f"result={len(result)}") 
    
    pd.DataFrame(result, columns=["user_id", "item_id"]).to_csv(f"./data/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)      

def main():
    
    # 시드 설정
    print("1. set_seed", flush=True)
    set_seed(42)

    # config 설정
    print("2. get_bpr_config", flush=True)
    bpr_config = get_bpr_config()

    if os.path.isfile(bpr_config['saved_model_file']):
     
        bpr_config, bpr_model, dataset, _ , _, test_data = crack_load_data_and_model(
            model_file= bpr_config['saved_model_file']
        )

    else:

        # 데이타셋 가져옴
        print("3. get_dataset", flush=True)
        train_data, valid_data, test_data, dataset = get_dataset(bpr_config)
        
        # 훈련
        print("4. train", flush=True)
        bpr_model = bpr_train(train_data, valid_data, bpr_config)


    # 데이터 로딩.
    print("5. get_dataframe", flush=True)
    train_df, idx2user, idx2item = get_dataframe()
    
    # 추론
    print("6. inference", flush=True)
    user_predicted_items, popular_top_10 = bpr_inference(train_df.copy(), idx2user, idx2item, bpr_config, bpr_model, dataset, test_data)

    print(len(user_predicted_items))
   
    # config 설정
    print("7. get_config", flush=True)
    config = get_config()

    # 데이타셋 가져옴
    print("8. get_dataset", flush=True)
    train_data, valid_data, test_data, dataset = get_dataset(config)
    
    # 훈련
    print("9. train", flush=True)
    #model = train(train_data, valid_data, config)

    _, model, _, _ , _, _ = crack_load_data_and_model(model_file= '/data/ephemeral/home/work/python/upstageailab-ocr-recsys-competition-recsys-5/notebooks/korea202/saved/SASRec-Oct-16-2025_15-01-47.pth')

    # 추론
    print("10. inference", flush=True)
    inference(user_predicted_items, train_df.copy(), idx2user, idx2item, config, model, dataset, test_data, popular_top_10)

    print("11. end", flush=True) 

if __name__ == "__main__":
    main()