from cProfile import label
import re
import random
import pandas as pd
import numpy as np
import warnings
from os.path import join as pjoin
warnings.filterwarnings(action='ignore')

'''
Description
-----------
전처리에 필요한 파라미터 지정

    args.balanced: True일 때 get_balanced_dataset를 통한 분할,\
         False일 때 분할 비율에 따라 랜덤하게 분할

    args.use_valid: 검증 데이터 사용 여부, False일 때 validation data는 빈 값 저장
    args.test_ratio: 테스트 데이터 비율
    args.seed: random seed 고정 (19로 고정)
'''
def base_setting(args):
    args.balanced = getattr(args, 'balanced', True)
    args.test_ratio = getattr(args, 'test_ratio', 0.2)
    args.seed = getattr(args, 'seed', 19)

'''
Description
-----------
전처리 함수

    def del_newline(text : str)
        개행/탭 문자 공백 문자로 변경
    def del_special_char(text : str)
        느낌표, 물음표, 쉼표, 온점, 물결, @을 제외한 특수문자 삭제
    def repeat_normalize(text : str, num_repeats : int)
        반복 문자 개수 num_repeats으로 제한
    def del_nickname(text : str)
        @+문자 형식의 닉네임 삭제 (댓글 데이터에 닉네임 포함됨)
    def del_duplicated_space(text : str)
        중복 공백 삭제
'''
def del_newline(text : str):
    return re.sub('[\s\n\t]+', ' ', text)

def del_special_char(text : str):
    return re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ,.?!~@0-9a-zA-Z\s]+', '', text)

repeatchars_pattern = re.compile('(\D)\\1{2,}')
def repeat_normalize(text : str, num_repeats : int):
    if num_repeats > 0:
        text = repeatchars_pattern.sub('\\1' * num_repeats, text)
    return text

def del_duplicated_space(text : str):
    return re.sub('[\s]+', ' ', text)

def preprocess(text : str):
    proc_txt = del_newline(text)
    proc_txt = del_special_char(proc_txt)
    proc_txt = repeat_normalize(proc_txt, num_repeats=3)
    proc_txt = del_duplicated_space(proc_txt)
    return proc_txt.strip()

def processing(args, data):
    base_setting(args)

    # seed 고정
    random.seed(args.seed)
    np.random.seed(args.seed)

    # labeling
    labels = ['spam', 'ham']
    data['label'] = list(map(lambda x: labels.index(x), data['label']))
    
    # text processing
    data['proc_text'] = list(map(preprocess, data['text']))
    data.to_csv(pjoin(args.data_dir, 'proc_data.csv'), index=False)
    return data


'''
Description
-----------
테스트 시, 되도록 모든 클래스에 대한 정답률 파악을 위해 \
    5개 미만의 데이터를 보유한 클래스의 경우 임의로 할당

train, valid, test의 클래스 분포를 기존 data의 클래스 분포와 동일하게 유지  
'''
def get_balanced_dataset(data : pd.DataFrame, test_ratio : float, use_valid : bool):
    
    valid = train = test = pd.DataFrame()

    for idx in data.label.unique().tolist():
        sub_data = data[data.label==idx]
        num_valid = num_test = int(len(sub_data) * test_ratio)

        if num_test == 0:
            if len(sub_data) < 2:
                train = pd.concat([train, sub_data], ignore_index=True)
                continue
            elif len(sub_data) < 3:
                test = pd.concat([test, sub_data.iloc[:1]], ignore_index=True)
                train = pd.concat([train, sub_data.iloc[1:]], ignore_index=True)
                continue
            else: 
                # class 내 데이터가 3-4개 인 경우
                num_valid = num_test = int(len(sub_data)/ 3)

        test_idx = 2 * num_test if use_valid else num_test
        valid_idx = num_valid if use_valid else 0

        valid = pd.concat([valid, sub_data.iloc[:valid_idx]], ignore_index=True)
        test = pd.concat([test, sub_data.iloc[valid_idx:test_idx]], ignore_index=True)
        train = pd.concat([train, sub_data.iloc[test_idx:]], ignore_index=True)

        del sub_data

    return train, test, valid

'''
Description
-----------
전체 데이터를 train, valid, test로 분할하여 args.result_dir 내에 저장
'''
def split_dataset(args, data):
    if args.balanced:
        train, test, valid = get_balanced_dataset(data=data, test_ratio=args.test_ratio, \
            use_valid=args.use_valid)
    else:
        data = data.sample(frac=1, random_state=args.seed)
        num_test = int(len(data) * args.test_ratio)

        test_idx = 2 * num_test if args.use_valid else num_test
        valid_idx = num_test if args.use_valid else 0

        valid = data.iloc[:valid_idx]
        test = data.iloc[valid_idx:test_idx]
        train = data.iloc[test_idx:]

    print(f"Train Distribution : \n{train.label.value_counts()}")
    print(f"Valid Distribution : \n{valid.label.value_counts()}")
    print(f"Test Distribution : \n{test.label.value_counts()}")

    valid.to_csv(pjoin(args.result_dir, 'valid.csv'), index=False)
    test.to_csv(pjoin(args.result_dir, 'test.csv'), index=False)
    train.to_csv(pjoin(args.result_dir, 'train.csv'), index=False)

    print(f"Total Number of Data : {len(data)} -> {len(valid) + len(test) + len(train)}")
    return
