# 预训练 word2vec(word 2 vec_model.py)

## 一、数据预处理函数
将一句话文本进行分词、去除停词等操作
```python
def preprocess_text(text, stopword=False):  
    """  
    数据预处理  
    :param text:  待处理文本,例如：'我是一个好人'  
    :param stopword:  是否去除停用词，默认不去除  
    :return:  处理后的文本,例如：'我 是 一个 好 人'  
    """    text = re.sub(r'\d+', '', text)  # 去除数字  
    text = ' '.join(jieba.cut(text))  # 分词  
    # 是否停词  
    if stopword:  
        text = ' '.join([word for word in text.split() if word not in stopwords])  # 去除停用词  
    else:  
        pass  
    return text
```

## 二、word 2vec 函数
将处理完的数据训练 word2vec 模型，并保存到相应位置
```python
def train_word2vec(data, path):  
    """  
    训练word2vec模型  
    :param data:  数据集 例如：pd.DataFrame(['我 是 一个 好人', '你 是 一个 坏人'])  
    :param path:    :return:  
    """    sentences = [text.split() for text in data]  
    model = Word2Vec(sentences, vector_size=128, window=5, min_count=3, workers=4)  # 训练模型  
    # vector_size 词向量的维度  
    # window 窗口大小  
    # min_count 词频阈值，小于该值的词将会被过滤掉  
    # workers 训练的并行数  
    model.save(path)
```

## 三、主函数
```python

import numpy as np  
import pandas as pd  
import jieba  
import re  
from gensim.models import Word2Vec  
from torch.utils.data import Dataset  
import torch

if __name__ == '__main__':  
    # 定义路径  
    data_path = '../../data/ChineseNlpCorpus/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv'  
    stopwords = '../../data/stopwords.txt'  
    model_path = 'word2vec.pkl'  
  
    # =============================== 数据预处理 =================================    # 读取数据  
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
```


# 利用训练好的 word2vec 模型完成下游任务
## 一、思路
- 1.导入训练 word2vec 前输入的文本数据
- 2.将文本数据根据 word2vec 网络转成向量
- 3. 划分训练集、测试集、验证集
- 4.建立 dataset、创建 dataloader
- 5.编写下游任务分类器 ANNmodel.py
- 6.训练模型
## 二、导入训练 word2vec 前输入的文本数据
```python
# 导入库
import numpy as np  
from model.ANNmodel import ANNmodel  
import torch  
import wandb  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from gensim.models import Word2Vec  
from torch.utils.data import DataLoader, Dataset  
from example.word2vec.word2vec_model import collate, HotelDataset  
import pytorch_lightning as pl  # 如果第六步不使用pytorchlightning方法的话可以注释 
from tqdm import tqdm  
import torch.nn as nn  
import torch.optim as optim
# 定义路径  
data_path = '../data/ChineseNlpCorpus_processed.csv'  
stopwords_path = '../data/stopwords.txt'  
model_path = '../word2vec/word2vec.pkl'  
# 调用gpu  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
# 导入数据  
data = pd.read_csv(data_path)
```

## 三、将文本数据根据已训练的 word2vec 模型转成向量
```python
# 导入训练好的word2vec模型  
data = pd.read_csv(data_path)  
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
  
X,y = pd.DataFrame(X),data['label']
```
## 四、数据切分、dataset、dataloader
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42)  
X_train.reset_index(drop=True, inplace=True)  
y_train.reset_index(drop=True, inplace=True)  
train_data2 = HotelDataset(np.array(X_train), np.array(y_train))  
train_loader2 = torch.utils.data.DataLoader(train_data2, batch_size=64, shuffle=True, drop_last=True, collate_fn=None)
```

## 五、编写下游任务分类器 (ANNmodel.py)
```python
import torch  
import torch.nn as nn  
import pytorch_lightning as pl  
  
  
class ANNmodel(pl.LightningModule):  
    def __init__(self, input_dim, output_dim):  
        super(ANNmodel, self).__init__()  
        self.fc1 = nn.Linear(input_dim, 128)  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(128, output_dim)  
        # softmax  
        self.softmax = nn.Softmax(dim=1)  
        self.loss = nn.CrossEntropyLoss()  
  
    def forward(self, x):  
        x = self.fc1(x)  
        x = self.relu1(x)  
        x = self.fc2(x)  
        # 分类  
        x = self.softmax(x)  
        return x  
  
    def configure_optimizers(self):  
        return torch.optim.Adam(self.parameters(), lr=0.001)  
  
    def training_step(self, batch, batch_idx):  
        x, y = batch  
        y_hat = self(x)  
        loss = self.loss(y_hat, y)  
        return loss
```


## 六、训练模型、
### 利用 pytorch_lightning 训练模型

```python
model = ANNmodel(input_dim=128, output_dim=2)  
trainer = pl.Trainer(max_epochs=100)  
trainer.fit(model, train_loader2)
```

### 正常训练模型
```python
# 定义模型  
model = ANNmodel(input_dim=128, output_dim=2)  
model.to(device)  
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数  
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器  
for epoch in range(100):  
    model.train()  
    for i, data in enumerate(train_loader2):  
        inputs, labels = data  
        optimizer.zero_grad()  # 梯度清零  
        output = model(inputs)  # 前向传播 input_dim=(128,1), output_dim=2        loss = criterion(output, labels.squeeze())  # 计算损失 labels.squeeze()将标签的维度从(64,1)变为(64,)  
        loss.backward()  # 反向传播  
        optimizer.step()  # 更新参数  
    if epoch % 100 == 0:  
        print('epoch: {}, loss: {}'.format(epoch, loss.item()))
```


