# 将文件夹中的ChnSentiCorp_htl_all.csv文件转换为train.txt, test.txt, dev.txt
# 并进行数据预处理，如下：
# 去除停用词，分词，去除标点符号，去除数字，去除空格

import pandas as pd
import jieba
import re
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from gensim.models import Word2Vec




def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # 去除数字
    text = ' '.join(jieba.cut(text))  # 分词
    text = ' '.join([word for word in text.split() if word not in stopwords])  # 去除停用词
    return text


# 使用word2vec进行文本预训练，并保存模型
def train_word2vec(data,path):
    sentences = [text.split() for text in data['review']]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)  # 训练模型
    # vector_size 词向量的维度
    # window 窗口大小
    # min_count 词频阈值，小于该值的词将会被过滤掉
    # workers 训练的并行数
    model.save(path)


if __name__ == '__main__':
    # 定义路径
    data_path = '../../data/ChineseNlpCorpus/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv'
    stopwords_path = '../../data/stopwords.txt'
    model_path = 'word2vec.model'
    # 读取数据
    data = pd.read_csv(data_path)
    data = data[['label', 'review']]
    data = data.dropna()
    data = data.reset_index(drop=True)
    #  # 数据预处理:去除数字、分词、去除停用词
    stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
    data['review'] = data['review'].apply(preprocess_text)
    # 去除处理完后的review为空的行
    data = data[data['review'] != '']
    # 训练word2vec模型并保存
    train_word2vec(data,model_path)
    # 保存预处理后的数据
    data.to_csv('../data/ChineseNlpCorpus_processed.csv',index=False)
