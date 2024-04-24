from gensim.models import Word2Vec
model1 = Word2Vec([['我', '是', '我是','他是'], ['我是','他是', '我是']], vector_size=100, window=1, min_count=0, workers=4)
a = model1.wv['我']
b = model1.wv['是']
c = model1.wv['我是']
d = model1.wv['我','是','我是']

# e = model1.wv['他'] # c = model1.wv['他是'] # 此处报错，说明word2vec不支持没词转化


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

# 假设我们有一个数据集X和对应的标签y
X = np.random.rand(1000, 10)  # 这是一个1000个样本、10个特征的数据集
y = np.random.randint(2, size=1000)  # 这是对应的二分类标签

# 创建一个五折交叉验证器
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化神经网络模型
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

# 用于存储每折交叉验证的结果
cv_scores = []

for train_index, test_index in kf.split(X):
    # 分割数据
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 训练模型
    clf.fit(X_train, y_train)

    # 预测测试集
    y_pred = clf.predict(X_test)

    # 计算并记录准确率
    cv_scores.append(accuracy_score(y_test, y_pred))

# 输出平均准确率
print("Average Cross-Validation Accuracy: ", np.mean(cv_scores))