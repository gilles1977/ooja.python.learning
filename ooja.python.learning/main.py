from ooja import network
import numpy as np
import matplotlib.pyplot as plt
import netdraw as nd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

#data = np.array([[0, 1],
#     [1, 1],
#     [1, 0],
#     [0, 0]])
#target = np.array([1,
#     0,
#     1,
#     0])
#mapper = [1]
s = datasets.load_iris()
data = s.data
target = s.target
scaler = MinMaxScaler()
data = scaler.fit_transform(data, target)

onehot = OneHotEncoder(sparse=False, categories='auto')
target = onehot.fit_transform(target.reshape(len(target), 1))

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)

#mapper = np.unique(target)
#mapper = np.arange(3)
#ni = data.shape[1]
#no = len(mapper)
nsamples = len(data)



n = network.Network(lambda o: np.round(o, 2))
n.addlayer(17, ninput=4, transfer=network.Sigmoid())
n.addlayer(3, transfer=network.Sigmoid(), cost=network.Cost.MeanSquareError())
e = 1.
i = 0
minibatchsize = 20

for i in range(1, 5000):
    print('\r'+str(i)+'      '+str(e), sep='', end='', flush=True)
    #if e < 0.5:
    #   break
    e = 0.
    for j in range(0, X_train.shape[0], minibatchsize):
        X_train_mini = X_train[j:j + minibatchsize]
        y_train_mini = y_train[j:j + minibatchsize]

        y = n.forward(X_train_mini)
        n.backprop(y_train_mini, .01)
        e += n.E

    e = e / minibatchsize

    #y = n.forward(normalized)
    #n.backprop(tencoded, .001)
    #e = n.E

    #e = 0.5 * np.sqrt(e/nsamples)

for x in range(0, len(data)):
    print()
    o = n.forward(data[x])
    t = target[x]
    print("x={0} --> t={1} o={2}".format(data[x], t, o))

n.display(e, i)
