# 对数据进行预处理，训练word2vec模型
# 因为后续需要使用w2v,所以需要数据处理方法尽量函数化，方便后续调用
# 并进行数据预处理，如下：
# 去除停用词，分词，去除标点符号，去除数字，去除空格
import numpy as np
import pandas as pd
import jieba
import re
from gensim.models import Word2Vec
from torch.utils.data import Dataset
import torch


def preprocess_text(text, stopword=False):
    """
    数据预处理
    :param text:  待处理文本,例如：'我是一个好人'
    :param stopword:  是否去除停用词，默认不去除
    :return:  处理后的文本,例如：'我 是 一个 好 人'
    """
    text = re.sub(r'\d+', '', text)  # 去除数字
    text = ' '.join(jieba.cut(text))  # 分词
    # 是否停词
    if stopword:
        text = ' '.join([word for word in text.split() if word not in stopwords])  # 去除停用词
    else:
        pass
    return text


# 使用word2vec进行文本预训练，并保存模型
def train_word2vec(data, path):
    """
    训练word2vec模型
    :param data:  数据集 例如：pd.DataFrame(['我 是 一个 好人', '你 是 一个 坏人'])
    :param path:
    :return:
    """
    sentences = [text.split() for text in data]
    model = Word2Vec(sentences, vector_size=128, window=5, min_count=3, workers=4)  # 训练模型
    # vector_size 词向量的维度
    # window 窗口大小
    # min_count 词频阈值，小于该值的词将会被过滤掉
    # workers 训练的并行数
    model.save(path)


class HotelDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return len(self.x_train)


class collate:
    """
    定义自己的 collate_fn 方法, 用于处理变长数据.
    :param data_tuple: data_tuple是一个列表，列表中包含 batch_size 个元组，每个元组中包含数据和标签
    :return: 返回处理后的数据
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, data):
        X = []
        y = []
        print(len(data))
        print('yes')
        for text in data:
            vec = []
            for word in text[0]:
                try:
                    vec.append(self.model.wv[word])
                except:
                    pass
            if len(vec) == 0:
                vec = np.array([0] * 128)
                X.append(vec)
            else:
                X.append(sum(vec) / len(vec))
            y.append(text[1])
        return torch.tensor(np.array(X)).to(self.device), torch.tensor(np.array(y)).to(self.device)


if __name__ == '__main__':
    # 定义路径
    data_path = '../../data/ChineseNlpCorpus/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv'
    stopwords = '../../data/stopwords.txt'
    model_path = 'word2vec.pkl'

    # =============================== 数据预处理 =================================
    # 读取数据
    data = pd.read_csv(data_path)
    data = data[['label', 'review']]
    data = data.dropna()
    data = data.reset_index(drop=True)
    #  # 数据预处理:去除数字、分词
    data['review'] = data['review'].apply(preprocess_text)
    # 去除处理完后的review为空的行
    data = data[data['review'] != '']
    # data = data.reset_index(drop=True, inplace=True)
    # ============================== 训练word2vec ================================
    # 训练word2vec模型并保存
    train_word2vec(data['review'], model_path)
    # 保存预处理后的数据
    data.to_csv('../data/ChineseNlpCorpus_processed.csv', index=False)
