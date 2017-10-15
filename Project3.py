#这个文件实现程序的主过程

#自己的工具包
import Different_Nodes
import Operations

#标准工具包
import numpy as np
from sklearn.datasets import load_boston

#shuffle的功能是打乱数据原有的顺序
from sklearn.utils import shuffle, resample

"""
:数据的含义:
        - CRIM     per capita crime rate by town 
        （人均犯罪率）
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        （超过 25,000 sq.ft.的住宅用地比例）
        - INDUS    proportion of non-retail business acres per town
        （每个城镇非零售商业区）
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        （是否沿河）
        - NOX      nitric oxides concentration (parts per 10 million)
        （氮氧化物的浓度，空气污染）
        - RM       average number of rooms per dwelling
        （每住宅平均房间数）
        - AGE      proportion of owner-occupied units built prior to 1940
        （自1940年以来自住房屋比例）
        - DIS      weighted distances to five Boston employment centres
        （到波士顿就业中心的加权距离）
        - RAD      index of accessibility to radial highways
        （高速公路的可到达性）
        - TAX      full-value property-tax rate per $10,000
        （税率）
        - PTRATIO  pupil-teacher ratio by town
        （城镇的学生教师比）
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        （黑人的数量）
        - LSTAT    % lower status of the population
        （人口的较低水平）
        - MEDV     Median value of owner-occupied homes in $1000's
        （在1000美元以内的自住房屋中位数）
"""
data = load_boston()

#把训练的数据集和目标取出来
X_ = data['data']
y_ = data['target']

# 数据标准化
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

#取出训练集的种类个数
n_features = X_.shape[1]
#神经网络隐层的个数
n_hidden = 10

#初始化神经网络的权重
#这个是10隐层，1输出层神经网络
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Neural network
X = Different_Nodes.Input()
y = Different_Nodes.Input()
W1 = Different_Nodes.Input()
b1 = Different_Nodes.Input()
W2 = Different_Nodes.Input()
b2 = Different_Nodes.Input()

l1 = Different_Nodes.Linear(X, W1, b1)
s1 = Different_Nodes.Sigmoid(l1)
l2 = Different_Nodes.Linear(s1, W2, b2)
cost = Different_Nodes.MSE(y, l2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

#整体的循环次数
epochs = 10000
#整体的训练样本个数
m = X_.shape[0]
#每次训练的分组数
batch_size = 11
#一共循环epochs次，每次batch_size个，训练m // batch_size可以训练完
steps_per_epoch = m // batch_size

#确定节点的运算顺序
graph = Operations.topological_sort(feed_dict)
#设定需要训练的数据集
trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

#执行异步训练方法：计算导数以后，再对权值做训练。而不是计算完导数以后立即修改权值。
# Step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # 通过对原有数据重采样，生成训练数据
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # 这反映了原作的软件构架性问题，这里又重复给输入节点赋值了！
        # 当初一开始就不应该在节点运算顺序确定算法里面给输入节点赋值
        X.value = X_batch
        y.value = y_batch

        # Step 2
        Operations.forward_and_backward(graph)

        # Step 3
        # 根据导数修正
        Operations.sgd_update(trainables)

        loss += graph[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))













