import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def load_data(df, shuffle=True, model_type='GKT', use_binary=True, res_len=2, use_cuda=True):
  
    df.dropna(subset=['skill_id'], inplace=True)

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id
    df['skill'], id_map = pd.factorize(df['skill_id'], sort=True)  # we can also use problem_id to represent exercises
    # Step 3 - Cross skill id with answer to form a synthetic feature
    # use_binary: (0,1); !use_binary: (1,2,3,4,5,6,7,8,9,10,11,12). Either way, the correct result index is guaranteed to be 1
    if use_binary:
        df['skill_with_answer'] = df['skill'] * 2 + df['correct']
    else:
        df['skill_with_answer'] = df['skill'] * res_len + df['correct'] - 1


    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    feature_list = []
    question_list = []
    seq_len_list = []

    def get_data(series):
        feature_list.append(series['skill_with_answer'].tolist())
        question_list.append(series['skill'].tolist())
        seq_len_list.append(series['correct'].shape[0])

    df.groupby('user_id').apply(get_data)

    # print('feature_dim:', feature_dim, 'res_len*question_dim:', res_len*question_dim)
    # assert feature_dim == res_len * question_dim

    data = KTTestDataset(feature_list, question_list)
    
    data_loader = DataLoader(data, batch_size=1, collate_fn=pad)
    return data_loader, {skill_id: idx for idx, skill_id in enumerate(id_map)}

def pad(batch):
    (features, questions) = zip(*batch)
    features = [torch.LongTensor(feat) for feat in features]
    questions = [torch.LongTensor(qt) for qt in questions]
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    return feature_pad, question_pad

class KTTestDataset(Dataset):
    def __init__(self, features, questions):
        super(KTTestDataset, self).__init__()
        self.features = features
        self.questions = questions

    def __getitem__(self, index):
        return self.features[index], self.questions[index]

    def __len__(self):
        return len(self.features)