import numpy as np
import copy
np.set_printoptions(suppress=True)  # 禁用科学计数
import pandas as pd

class Value():
    def __init__(self, name, values):
        """
        name是属性名, values是属性具体取值
        """
        self.name = name
        self.values = values

class DecisionTree():
    """
    决策树算法
    """
    class Node():
        """
        节点类
        """
        def __init__(self, genre=None):
            self.next = {}
            self.is_genre = False

        def add_genre(self, genre):
            """
            genre是当该节点是根节点时的类别
            """
            self.genre = int(genre)

        def add_next_node(self, next_node, value, name):
            """
            next_node是下一个节点, value是到该节点的取值(条件), next_name是取决划分的属性名
            """
            self.next_name = name
            self.next[value] = next_node

    def __init__(self, pattern):
        """
        pattern是决策树划分准则
        """
        self.pattern = pattern
        self.pruning = {}
        self.pruning['none'] = self.none_pruning
        self.pruning['pre'] = self.pre_pruning
        self.pruning['after'] = self.after_pruning

    def fit(self, X, y, A=None, pruning='none'):
        """
        A为属性集, 可以手动传入也可以自动创建
        """
        self.X = X
        self.y = y
        self.head = self.Node()  # 定义头节点
        self.now_pruning = pruning
        if A is None:
            A = []
            for i in np.arange(X.shape[1]):  # 创建A
                A.append(Value(i, np.arange(np.unique(X[:, i]).shape[0])))
        temp_X = X.copy()
        temp_y = y.copy()
        self.X_to_temp_X = {}  # 映射集合
        for i in range(temp_X.shape[1]):
            name = np.unique(temp_X[:, i])
            for j in range(name.shape[0]):
                self.X_to_temp_X[name[j]] = j
                temp_X[temp_X[:, i] == name[j], i] = j
        name = np.unique(temp_y)
        self.temp_y_to_y = {}
        for i in range(name.shape[0]):
            temp_y[temp_y == name[i]] = i
            self.temp_y_to_y[i] = name[i]
        self.X_to_temp_X = pd.DataFrame(self.X_to_temp_X, index=[0])
        self.temp_y_to_y = pd.DataFrame(self.temp_y_to_y, index=[0])
        temp_y = temp_y.astype(int)
        temp_X = temp_X.astype(int)
        self.pruning[pruning](temp_X, temp_y, A, self.head)

    def after_pruning(self, X, y, A, now_node):
        """
        后剪枝
        """
        self.none_pruning(X, y, A, now_node)
        def recursive(now_node, front_node):  # 考察front_node是否要替换为叶节点
            if not now_node.is_genre:
                for name in now_node.next:
                    recursive(now_node.next[name], now_node)
            front_node.is_genre = True
            pruning_acc = self.accuracy(self.X, self.y)
            front_node.is_genre = False
            acc = self.accuracy(self.X, self.y)
            if pruning_acc >= acc:
                front_node.is_genre = True
            return
        for name in self.head.next:
            recursive(self.head.next[name], self.head)

    def pre_pruning(self, X, y, A, now_node):
        """
        预剪枝
        """
        self.none_pruning(X, y, A, now_node)
        def recursive(now_node): # 考察该节点是否要替换为叶节点
            if now_node.is_genre:
                return
            now_node.is_genre = True
            pruning_acc = self.accuracy(self.X, self.y)
            now_node.is_genre = False
            self.next_set_is_genre(now_node, True)
            acc = self.accuracy(self.X, self.y)
            self.next_set_is_genre(now_node, False)
            if pruning_acc >= acc:
                now_node.is_genre = True
                return
            else:
                for name in now_node.next:
                    recursive(now_node.next[name])
        recursive(self.head)

    def none_pruning(self, X, y, A, now_node):
        """
        A是属性集, front_node是前一个节点, 该函数寻找的是前一个节点的下一个最优节点
        """
        judge = np.unique(y)
        if judge.shape[0] == 1:  # 判断是否属于同一类别
            now_node.add_genre(judge)
            now_node.is_genre = True
            return
        now_node.add_genre(np.argmax(np.bincount(y.reshape(len(y)))))  # D中样本数最多的类
        if (len(A) == 0) or (np.unique(np.unique(X) == X[0])):
            now_node.is_genre = True
            return
        a = self.find_best(X, y, A, self.pattern)  # 返回最优的属性子集
        temp_A = copy.deepcopy(A)
        for i in range(len(temp_A)):
            if temp_A[i].name == a.name:
                break
        temp_A.pop(i)
        for a_v in a.values:
            temp = (X[:, a.name] == a_v)
            X_v = X[temp]
            y_v = y[temp]
            if X_v.shape[0] == 0:
                now_node.is_genre = True
                return
            else:
                next_node = self.Node()
                now_node.add_next_node(next_node, a_v, a.name)
                self.none_pruning(X_v, y_v, temp_A, next_node)

    def next_set_is_genre(self, node, setting):
        for name in node.next:
            if setting == True:
                if node.next[name].is_genre == True:
                    node.next[name].tag = True
                else:
                    node.next[name].tag = False
                node.next[name].is_genre = True
            else:
                if node.next[name].tag == True:
                    continue
                else:
                    node.next[name].is_genre = False

    def find_best(self, X, y, A, pattern):  # 寻找最好的属性集
        bigger = np.NINF
        for a in A:
            temp = pattern(X, y, a)
            if temp > bigger:
                bigger = temp
                greater_a = a
        return greater_a

    def predict(self, X):
        y = np.zeros(X.shape[0]).reshape(1, X.shape[0]).T.astype('object')
        i = 0
        for x in X:  # 对每个向量x进行决策树的预测
            x = np.array(self.X_to_temp_X[x]).reshape(x.shape)
            temp = self.head
            while not temp.is_genre:
                temp = temp.next[x[temp.next_name]]
            y[i] = self.temp_y_to_y[temp.genre][0]
            i += 1
        return y

    def accuracy(self, X, y):
        y_pre = self.predict(X)
        return (np.sum(y_pre == y) / y.shape[0])

def ent(y):
    res = 0
    num = y.shape[0]
    for k in np.unique(y):
        p_k = np.sum(y == k) / num
        res += p_k * np.log2(p_k)
    return -res

def gain(X, y, a):  # 信息增益
    res = 0
    num = y.shape[0]
    for value in a.values:
        label = (X[:, a.name] == value)
        res += (np.sum(label) / num) * ent(y[label])
    return ent(y) - res

def gini_index(X, y, a):
    res = 0
    num = y.shape[0]
    for value in a.values:
        label = (X[:, a.name] == value)
        res += (np.sum(label) / num) * gini(y[label])
    return -res

def gini(y):
    sum = 0
    num = y.shape[0]
    for k in np.unique(y):
        p_k = np.sum(y == k) / num
        sum += p_k ** 2
    return 1 - sum

path = 'cmc.data'
data = pd.read_csv(
    path,
    names=np.arange(10)
)
data.head()
data = data.drop(labels=[0, 3], axis=1)
X = np.array(data.iloc[:, :7])
y = np.array(data.iloc[:, 7]).reshape(1, data.shape[0]).T

# 基于基尼指数划分选择的决策树算法
decisiontree = DecisionTree(gini_index)
# 未剪枝
decisiontree.fit(X, y, pruning='none')
print('accuracy is {}%'.format(decisiontree.accuracy(X, y) * 100))
decisiontree.predict(X)

# 预剪枝
decisiontree.fit(X, y, pruning='pre')
print('accuracy is {}%'.format(decisiontree.accuracy(X, y) * 100))
decisiontree.predict(X)

# 后剪枝
decisiontree.fit(X, y, pruning='after')
print('accuracy is {}%'.format(decisiontree.accuracy(X, y) * 100))
decisiontree.predict(X)

# 基于信息熵划分选择的决策树算法
decisiontree2 = DecisionTree(gain)
# 未剪枝
decisiontree2.fit(X, y, pruning='none')
print('accuracy is {}%'.format(decisiontree2.accuracy(X, y) * 100))
decisiontree2.predict(X)

# 预剪枝
decisiontree2.fit(X, y, pruning='pre')
print('accuracy is {}%'.format(decisiontree2.accuracy(X, y) * 100))
decisiontree2.predict(X)

# 后剪枝
decisiontree2.fit(X, y, pruning='after')
print('accuracy is {}%'.format(decisiontree2.accuracy(X, y) * 100))
decisiontree2.predict(X)
