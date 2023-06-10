import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle
import pandas
import emoji
import re
np.set_printoptions(threshold=np.inf)

stopwords = set([i.strip() for i in open(r'./datasets/stopwords.txt', 'r', encoding="utf-8").readlines()])

def tokenize(sentence):
    sentence = emoji.demojize(sentence)
    fileters = ['!', '"', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '\[', '\\', '\]', '^', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“']
    sentence = sentence.lower()  # 把大写转化为小写
    sentence = re.sub("<.*?>", " ", sentence)
    sentence = re.sub("<br />", " ", sentence)
    # 把表情去掉
    # sentence = re.sub("[^\\u0000-\\uFFFF]", ' ', sentence)
    # 把http替换为 url
    sentence = re.sub("(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", 'url', sentence)
    # 把@PramilaJayapal 替换为@xx
    sentence = re.sub("@[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", '@xxx', sentence)
    sentence = re.sub("|".join(fileters), " ", sentence)
    sentence = sentence.replace("’s", " is")
    sentence = sentence.replace("’m", " am")
    sentence = sentence.replace("n’t", " not")
    sentence = sentence.replace("n't", " not")
    sentence = sentence.replace("…", "")
    result = [i for i in sentence.split(" ") if len(i) > 0 and i not in stopwords]

    return result


class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Weibograph')):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))


def collate_fn(data):
    return data


class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):

        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))


class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, dataname, lower=2, upper=100000, droprate=0):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        # self.data_path = data_path
        self.data_path = './datasets/' + dataname + '_Interaction'
        self.droprate = droprate

        self.seq_len = 35
        self.ws = pickle.load(
            open('./datasets/' + dataname + '_ws.pkl', 'rb'))

        statuses = pandas.read_csv('./datasets/' + dataname + '/' + dataname + '_Statuses.csv',sep='\t',encoding='utf-8')

        statuses_dict = statuses.set_index('twitter_id')['20_status'].to_dict()
        self.statuses_dict = statuses_dict

        sources = pandas.read_csv('./datasets/' + dataname + '/' + 'source_tweets.txt', sep='\t', encoding='utf-8', names=['twitter_id', 'status'])
        sources_dict = sources.set_index('twitter_id')['status'].to_dict()
        self.source_dict = sources_dict


    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]


        text_real = eval(self.statuses_dict[int(id)])
        text_save = []
        for item in text_real:
            if type(item) != list:

                temp_text = tokenize(item)
                temp_text = self.ws.transform(temp_text, max_len=self.seq_len)
                text_save.append(temp_text)
            else:
                temp_text = [0 for i in range(self.seq_len)]
                text_save.append(temp_text)

        text_source = self.source_dict[int(id)]
        token_source = tokenize(text_source)
        token_source = self.ws.transform(token_source, max_len=self.seq_len)

        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]), tree_text_id=torch.LongTensor([int(id)]),
                    history_text=torch.LongTensor([text_save]), source=torch.LongTensor([token_source]))
