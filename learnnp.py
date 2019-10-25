import numpy as np
import matplotlib.pyplot as plt

def hello():
    my_array = np.array([1, 2, 3, 4, 5])
    print(my_array)
    print(my_array.shape)
    print(my_array[0])
    print(my_array[1])
    my_array[0] = -1
    print(my_array)

def hello2():
    # 圆括号和方括号都行
    my_new_array = np.zeros([5])
    print(my_new_array)
    my_new_array = np.random.random((5))
    print(my_new_array)
    my_2d_array = np.zeros([2, 3])
    print(my_2d_array)

    my_array = np.array([[4, 5], [6, 1]])
    print(my_array.shape)
    my_array_column_2 = my_array[:, 1]
    print(my_array_column_2)

def arr_ops():
    a = np.array([[1., 2.], [3., 4.]])
    b = np.array([[5., 6.], [7., 8.]])
    sum = a + b
    difference = a - b
    # 元素逐个计算
    product = a * b
    quotient = a / b
    print("Sum = \n", sum)
    print("Difference = \n", difference)
    print("Product = \n", product)
    print("Quotient = \n", quotient)
    # 矩阵计算
    matrix_product = a.dot(b)
    print("Matrix Product = \n", matrix_product)

def slicing():
    a = np.array([range(11, 16),
                  range(16, 21),
                  range(21, 26),
                  range(26, 31),
                  range(31, 36)])
    print('a = \n', a)
    print(a[0, 1:4])
    print(a[1:4, 0])
    print(a[::2, ::2])
    print(a[:, 1])

def properties():
    a = np.array([range(11, 16),
                  range(16, 21),
                  range(21, 26),
                  range(26, 31),
                  range(31, 36)])
    print(type(a))
    print(a.dtype)
    print(a.size)
    print(a.shape)
    print(a.itemsize)
    print(a.ndim)
    print(a.nbytes)

def basic_op():
    a = np.arange(25)
    print(a)
    a = a.reshape([5, 5])
    print(a)

    b = np.random.randint(0, 30, [25])
    print(b)
    b = b.reshape([5, 5])
    print(b)

    print('\n')
    print(a + b)
    print(a - b)
    print(a / b)
    print(a ** 2)
    print(a < b)
    print(a > b)
    print(a.dot(b))

def special():
    a = np.arange(10)
    print(a)
    print(a.sum())
    print(a.min())
    print(a.max())
    print(a.cumsum())

def indexing():
    a = np.arange(0, 100, 10)
    # print(a)
    indices = [1, 5, -1]
    b = a[indices]
    print(a)
    print(b)
    print(a[a >= 50])

def bool_masking():
    a = np.linspace(0, 2*np.pi, 50)
    b = np.sin(a)
    plt.plot(a, b)
    mask = b >= 0
    print(mask)
    plt.plot(a[mask], b[mask], 'bo')
    mask = (b >= 0) & (a <= np.pi/2)
    plt.plot(a[mask], b[mask], 'go')
    plt.show()

def Where():
    a = np.arange(0, 100, 10)
    b = np.where(a < 50)
    c = np.where(a >= 50)[0]
    print(b)
    print(c)

def from_buffer():
    s = b'Hello World'
    a = np.frombuffer(s, dtype='S1')
    print(a)

    list = range(5)
    it = iter(list)
    x = np.fromiter(it, dtype=np.float)
    print(x)
    
def log_create():
    a = np.logspace(1, 3, 3)
    print(a)
    b = np.logspace(1, 8, 4, 2)
    print(b)

def index():
    x = np.random.randint(0, 100, 100).reshape([10, 10])
    print('x is\n', x)
    print('sub matrix\n', x[np.ix_([1, 3], [2, 4])])

def array_ops():
    a = np.arange(12).reshape([3,4])
    print("原数组:\n", a, end='\n')
    print("对换数组\n", np.transpose(a), '\n')

def brdcst():
    x = np.array([[1],[2],[3]])
    y = np.array([[4,5,6]])
    b = np.broadcast(x,y)
    print(x + y)
    print(b.shape)

def joint():
    a = np.array([1,2,3]).reshape((3, 1))
    b = np.array([[4],[5],[6]])
    # 形状需要匹配
    print(np.concatenate((b, a), axis=1), '\n')
    # stack会增加维度
    # 改变axis看看会发生什么
    print(np.stack((b, a), axis=2))

    x = np.array([[1,2],[3,4]])
    y = np.array([[5,6],[7,8]])
    print('\n', np.concatenate((x, y), axis=0))

def test_plt():
    x = np.random.normal(0, 1, (1000000000))
    print(np.mean(x))
    print(np.std(x))
    # plt.hist(x, bins=np.linspace(-5, 5, 1000))
    # plt.show()

if __name__ == "__main__":
    # hello()
    # hello2()
    # arr_ops()
    # slicing()
    # properties()
    # basic_op()
    # print(np.array([1, 2, 3]).dot(np.array([4, 5, 6])))
    # special()
    # indexing()
    # bool_masking()
    # Where()
    # from_buffer()
    # log_create()
    # index()
    # array_ops()
    # brdcst()
    # joint()
    test_plt()