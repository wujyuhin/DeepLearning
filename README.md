# DeepLearning
My deep learning course teached by Prof Le Li

# word2vec

- 实现word2vec例子的数据来源是data文件夹中的ChnSentiCorp.csv

- 第一步运行example/word2vec/word2vec_model.py：
  - 对数据进行预处理
  - 指定csv文件训练word2vec模型（可以是其他文本文件）
  - 即可保存训练好的模型word2vec.pkl到本地文件夹example/word2vec/word2vec.pkl
  - 顺便将预处理后的数据也保存到example/data文件夹中（之后使用的数据均为预处理后的数据，因为如果修改预处理，那么word2vec模型自然需要重新训练）
  
  第二步运行example/word2vec/word2vec_model.py
  
- 第一步运行word2vec_model.py：
  - 对数据进行预处理
  - 指定csv文件训练word2vec模型（可以是其他文本文件）
  - 即可保存训练好的模型word2vec.model到本地文件夹example/word2vec
  - 顺便将预处理后的数据也保存到example/data文件夹中（之后使用的数据均为预处理后的数据，因为如果修改预处理，那么word2vec模型自然需要重新训练）
  
- 第二步运行dataset_prepare.py:

  - 对所有数据文本都通过已训练的word2vec模型映射到向量
  - 对映射的向量数据切分为train,dev,test三个数据集，并保存到example/data文件夹

思考：我认为是有必要单独出来，因为对管理数据比较清晰，算法可能需要运行很多遍，但切分数据集并不需要。

- 第三步运行word2vec_classify.py
  - 导入train、dev、test数据
  - 直接作为输入数据进行训练

上述模型可以调整的参数有word2vec的训练参数、分类器的训练参数

# LSTM（coding）
