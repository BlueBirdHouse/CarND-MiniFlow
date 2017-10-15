import numpy as np
import Base_Node

"""本文件从基础节点类当中派生出各种功能的节点"""

class Input(Base_Node.Node):
    """输入节点，作为网络的输入模块"""
    def __init__(self):
        """输入节点是一个没有inbound_nodes的节点"""
        Base_Node.Node.__init__(self,inbound_nodes=[])

    def forward(self,value = None):
        """将输入节点的前向传递函数设计为输入什么就是什么的"""
        if value is not None:
            self.value = value

    def backward(self):
        #输入节点的导数就是都加起来
        #输入节点由于没有前级节点，所以它的gradients的键值，不是前级的节点，而是自己！
        #那么，这个信息就是对本地节点的修正量
        self.gradients = {self: 0}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1

class Add(Base_Node.Node):
    """加法节点，做加法。"""
    def __init__(self, x,y):
        """加法是一个二元运算，有加法一定有两个节点作为输入与之关联"""
        Base_Node.Node.__init__(self,inbound_nodes=[x,y])

    def forward(self):
        """实现加法功能的函数"""
        ADD = 0
        for Printer in self.inbound_nodes:
            ADD = ADD + Printer.value

        self.value = ADD

class Linear(Base_Node.Node):
    """
    实现一个线性变换节点
    """
    def __init__(self, inputs, weights, bias):
        Base_Node.Node.__init__(self, [inputs, weights, bias])
        '''
        输入项目是输入节点类型对象。
        一个输入对应一个权重，最后只有一个偏移量
        注意：如果不是矩阵的话，inputs和weights一定都是行向量
        '''

    def forward(self):
        """
        计算线性结构的前向传递函数
        """
        #一般的请款改下,X都只有一行
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        B = self.inbound_nodes[2].value

        self.value = np.dot(X,W) + B

    def backward(self):
        """
        Calculates the gradient based on the output values.
        这里突出显示出向量方法的问题，提高了速度但是降低了可读性
        CS231n可是将每一个运算都抽象为二元函数来处理的
        """
        # 初始化每一个输入均对应一个导数0.分别是inputs, weights, bias。
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            # 获得后级反传的导数，注意，对于每一个输出，后级反传的导数只有一个。
            grad_cost = n.gradients[self]
            # 计算关于inputs的导数
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # 计算关于weights的导数
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # 计算关于bias的导数
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)

class Sigmoid(Base_Node.Node):
    """
    实现激活函数Sigmoid节点
    """
    def __init__(self, node):
        """这个节点仅支持一个输入"""
        Base_Node.Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used later with `backward` as well.

        `x`: A numpy array-like object.

        Return the result of the sigmoid function.
        """
        return 1./(1. + np.exp(-x))


    def forward(self):
        """
        Set the value of this node to the result of the
        sigmoid function, `_sigmoid`.
        """
        x = self.inbound_nodes[0].value
        self.value = self._sigmoid(x)


    def backward(self):
        #初始化每一个输入均对应一个导数0.当然实际中S函数仅有一个输入
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:
            #输出n传给输入的导数
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost

class MSE(Base_Node.Node):
    """计算误差的节点"""
    def __init__(self, y, a):
        """
        作为一个网络的终端节点使用。
        这个节点仅计算当前输入产生的误差，与历史输入无关。
        """
        Base_Node.Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        注意，一定要是y-a不可以反置，否则对应的导数存储位置发生错误！
        """
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        self.diff = y - a
        self.value = np.mean(np.square(self.diff))
    
        self.m = self.inbound_nodes[0].value.shape[0]

    def backward(self):
        """
        注意导数存储的对应性。
        由于在forward里面，y是存储在0位上的，所以，对应的导数也是存储在第0位上的。
        向字典当中逐个添加元素
        """
        #对y的导数
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        #对a的导数
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff

        
        
        
        
        
        
        
        
        