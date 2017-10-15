"""这个文件记录那些不可能成为主代码的测试代码"""
import Operations

def f(x):
    """
    Quadratic function.

    It's easy to see the minimum value of the function
    is 5 when is x=0.
    """
    return x**2 + 5


def df(x):
    """
    Derivative of `f` with respect to `x`.
    """
    return 2*x


# Random number better 0 and 10,000. Feel free to set x whatever you like.
#x = random.randint(0, 10000)
x = 1
# TODO: Set the learning rate
learning_rate = 0.01
epochs = 500

for i in range(epochs+1):
    cost = f(x)
    gradx = df(x)
    print("EPOCH {}: Cost = {:.5f}, x = {:.5f}".format(i, cost, gradx))
    x = Operations.gradient_descent_update(x, gradx, learning_rate)


