class Node(object):
    """提供每一个运算单元的基本数据结构。"""
    def __init__(self, inbound_nodes=[]):
        """由于每一个节点要和前面的节点做连接，所以当非初始节点的时候，应该提供输入节点的链接"""
        self.inbound_nodes = inbound_nodes

        self.outbound_nodes = []

        #自动与前面的节点挂接
        #如果不初始化输入节点，则自动默认为[]，这样这个循环就不会执行
        for u in self.inbound_nodes:
            u.outbound_nodes.append(self)
            
        self.value = None

        # 字典类型，为了防止导数错位
        # 对哪个输入做导数，那么字典的键值就是那个输入（的对象）
        # 对应的字典值就是应该传给前级的导数
        self.gradients = {}

    def forward(self):
        """前向传输函数"""
        raise NotImplementedError

    def backward(self):
        """后向传输函数"""
        """
        反向传输的诀窍有两条：
        1.对于下级的反传，要乘以本地的导数
        2.如果有多个下级，则每一个均乘以本地的导数以后要相加
        """
        raise NotImplementedError