"""这个文件定义操作网络的各种功能"""
import Different_Nodes

def topological_sort(feed_dict):
    """
    使用Kahn's Algorithm来判断网络内的单元运算顺序

    `feed_dict`: 是一个字典，其键值是输入节点，值是给这个输入节点输入的数据

    返回运算顺序配列的节点
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Different_Nodes.Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def forward_pass(output_node, sorted_nodes):
    """
    前向运算一个网络

    Arguments:

        `output_node`: 指定要输出的节点
        `sorted_nodes`: 被topological_sort重新排列的节点.

    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value

def forward_and_backward(sorted_nodes):
    """
    做一次前向计算，一次后向计算。
    `graph`: 是运行完`topological_sort`以后的结果.
    """
    # Forward pass
    for n in sorted_nodes:
        n.forward()

    for n in sorted_nodes[::-1]:
        n.backward()


def gradient_descent_update(x, gradx, learning_rate):
    """
    定义最速下降法的基础结构
    """
    x = x - learning_rate*gradx
    # Return the new value for x
    return x


def sgd_update(trainables, learning_rate=1e-2):
    """
    这里有叙述不清的问题，因为能够被训练的节点trainables一定是终端节点，
    终端节点的gradients存储的不是要传给前级的导数，为啥，因为它没有前级，
    所以它存储的就是本地的导数累计，就是对自己的修正量
    """
    for t in trainables:
        partial = t.gradients[t]
        t.value -= learning_rate * partial
