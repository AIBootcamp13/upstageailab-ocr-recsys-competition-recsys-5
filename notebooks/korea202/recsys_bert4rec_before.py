import pandas as pd
import os

def main():

    # 데이터를 로딩.
    train_df = pd.read_parquet('./data/train.parquet')
    
    # item 용 미리 복사
    item_df = train_df[['item_id', 'category_code', 'brand', 'price']].drop_duplicates(subset=['item_id'], keep='first')

    # event_time을 datatime으로 변환
    train_df['event_time'] = pd.to_datetime(train_df['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
    #  ********** 시간순으로 정렬함 **********
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

    # recbole 형식으로 컬럼명 변경
    recbole_df = train_df.rename(columns={'user_idx': 'user_idx:token', 'item_idx': 'item_idx:token', 'user_session': 'user_session:token', 'event_time': 'event_time:float', 'event_type': 'event_type:token'})

    # 디렉토리 생성
    os.makedirs('./data/sasrec_data', exist_ok=True)

    # 전처리된 df 저장
    train_df.to_csv('./data/prepared_df.csv', sep='\t',index=None)

    # 전처리 데이터 저장
    recbole_df[['user_idx:token', 'item_idx:token', 'user_session:token', 'event_time:float', 'event_type:token']].to_csv('./data/sasrec_data/sasrec_data.inter', sep='\t',index=None)

    # item 파일용 작업
    item_df['item_idx'] = item_df['item_id'].map(item2idx)
    item_df = item_df.dropna().reset_index(drop=True)
    # recbole 형식으로 컬럼명 변경
    item_df = item_df.rename(columns={'item_idx': 'item_idx:token', 'category_code': 'category_code:token', 'brand': 'brand:token', 'price': 'price:float'})
    item_df[['item_idx:token', 'category_code:token', 'brand:token', 'price:float']].to_csv('./data/sasrec_data/sasrec_data.item', sep='\t',index=None)

if __name__ == "__main__":
    main()