# 将文件夹中的ChnSentiCorp_htl_all.csv读取后，通过已经训练的word2vec模型进行文本向量化，如下：
import numpy as np
import pandas as pd
import jieba
import re
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from tqdm import tqdm
from gensim.models import Word2Vec

# 定义路径
data_path = 'data/ChineseNlpCorpus_processed.csv'
stopwords_path = '../data/stopwords.txt'
model_path = '../example/word2vec/word2vec.model'

# 读取数据
data = pd.read_csv(data_path)

# 导入训练好的word2vec模型
model = Word2Vec.load(model_path)

# 使用word2vec进行文本向量化
X = []
for text in tqdm(data['review']):
    vec = []
    for word in text: # 如果是英文，直接text.split()即可，因为英文单词之间是空格分隔，而中文是字之间没有空格
        try:
            vec.append(model.wv[word])  # 词向量化
        except:
            pass  # 词不在词典中,跳过
    if len(vec) == 0:  # 如果句子中的词都不在词典中，则用0向量代替
        vec = np.array([0] * 128)
        X.append(vec)
    else:
        X.append(sum(vec) / len(vec))  # 句子向量化,取平均值

X = pd.DataFrame(X)
y = data['label']
# word2XY(data, model)



# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 保存数据集
if not os.path.exists('../example/data/'):
    os.makedirs('../example/data/')

# 保存训练集、测试集、验证集
X_train.to_csv('../example/data/train.csv', index=False)
X_test.to_csv('../example/data/test.csv', index=False)
X_dev.to_csv('../example/data/dev.csv', index=False)
y_train.to_csv('../example/data/y_train.csv', index=False)
y_test.to_csv('../example/data/y_test.csv', index=False)
y_dev.to_csv('../example/data/y_dev.csv', index=False)
print('数据集保存成功！')

