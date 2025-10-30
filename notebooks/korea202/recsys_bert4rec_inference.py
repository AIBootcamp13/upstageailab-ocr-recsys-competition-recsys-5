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
from recbole.model.sequential_recommender import BERT4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, get_model
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start.quick_start import load_data_and_model

# load_data_and_model => crack_load_data_and_model로 대체 // pyTorch 2.6 부터 torch.load(model_file)  함수의 weights_only 파라미터 기본값이 False 에서 True 로 변경되어 저장된 모델의 호환이 안됨
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


def get_dataframe():
    
    train_df = pd.read_parquet('./data/train.parquet')

    # event_time을 datatime으로 변환
    train_df['event_time'] = pd.to_datetime(train_df['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
    train_df = train_df.sort_values(by=['event_time'])
    train_df['event_time'] = train_df['event_time'].values.astype(float)
    train_df = train_df[['user_id','item_id','user_session','event_time', 'event_type']]

    # 사용자(user)와 아이템(item)을 인덱스로 매핑하기 위한 딕셔너리 생성
    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}  # 각 사용자를 인덱스로 매핑
    idx2user = {k: v for k, v in enumerate(train_df['user_id'].unique())}  # 각 인덱스를 사용자로 매핑
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}  # 각 아이템을 인덱스로 매핑
    idx2item = {k: v for k, v in enumerate(train_df['item_id'].unique())}  # 각 인덱스를 아이템으로 매핑

    # 사용자와 아이템을 인덱스로 변환하여 새로운 열 추가
    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)

    train_df = train_df.dropna().reset_index(drop=True)
    
    print(train_df.head()) 

    return  train_df, idx2user, idx2item  

def inference(train_df, idx2user, idx2item):

    # 사용자별 시간별 아이템 정렬
    train_df = train_df.sort_values(by=['user_session','event_time'])
    users = defaultdict(list) # defaultdict은 dictionary의 key가 없을때 default 값을 value로 반환
    for u, i in zip(train_df['user_idx'], train_df['item_idx']):
        users[u].append(i)


    # 추천 상품 생성/저장 
    # 저장된 model명으로 변경하고 model과 데이터 불러오기
    config, model, dataset, _ , _, test_data = crack_load_data_and_model(
        model_file='./saved/SASRec-Oct-01-2025_18-19-50.pth'
    )
    print('Data and model load compelete')

    # cold-start user는 popular_top_10 items으로 make-up
    # groupby('item_idx') 가 쿼리의 인덱스가 된다.
    popular_top_10 = train_df.groupby('item_idx').count().rename(columns = {"user_idx": "user_counts"}).sort_values(by=['user_counts', 'item_idx'], ascending=[False, True])[:10].index
    result = []

    # short history user에 대해선 popular로 처리
    for uid in tqdm(users):
        if str(uid) in dataset.field2token_id['user_idx']:
            recbole_id = dataset.token2id(dataset.uid_field, str(uid))
            topk_score, topk_iid_list = full_sort_topk([recbole_id], model, test_data, k=10, device=config['device'])
            predicted_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
            predicted_item_list = predicted_item_list[-1]
            predicted_item_list = list(map(int,predicted_item_list))
        else: # cold-start users
            predicted_item_list = list(popular_top_10)

        for iid in predicted_item_list:
            result.append((idx2user[uid], idx2item[iid]))


    pd.DataFrame(result, columns=["user_id", "item_id"]).to_csv(f"./data/output_{datetime.now().strftime('%Y%m%d')}.csv", index=False)  


def main():
    
    # 시드 설정
    set_seed(42)

    # 데이터 로딩.
    train_df, idx2user, idx2item = get_dataframe()
    
    # 추론
    inference(train_df, idx2user, idx2item)


if __name__ == "__main__":
    main()