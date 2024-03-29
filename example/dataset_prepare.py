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

# 读取数据
data = pd.read_csv('../data/ChineseNlpCorpus/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv')
data = data[['label', 'review']]
data = data.dropna()
data = data.reset_index(drop=True)

# 去除停用词
stopwords = [line.strip() for line in open('../data/stopwords.txt', 'r', encoding='utf-8').readlines()]


def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # 去除数字
    text = ' '.join(jieba.cut(text))  # 分词
    text = ' '.join([word for word in text.split() if word not in stopwords])  # 去除停用词
    return text


data['review'] = data['review'].apply(preprocess_text)  # 数据预处理

# 划分数据集
X = data['review']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 保存数据集
if not os.path.exists('../example/data/'):
    os.makedirs('../example/data/')

with open('../example/data/train.txt', 'w', encoding='utf-8') as f:
    for i in range(len(X_train)):
        f.write(X_train.iloc[i] + '\t' + str(y_train.iloc[i]) + '\n')

with open('../example/data/test.txt', 'w', encoding='utf-8') as f:
    for i in range(len(X_test)):
        f.write(X_test.iloc[i] + '\t' + str(y_test.iloc[i]) + '\n')

with open('../example/data/dev.txt', 'w', encoding='utf-8') as f:
    for i in range(len(X_dev)):
        f.write(X_dev.iloc[i] + '\t' + str(y_dev.iloc[i]) + '\n')


