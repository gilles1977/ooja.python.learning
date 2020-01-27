from ooja import network
import numpy as np
import matplotlib.pyplot as plt
import netdraw as nd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split

#data = np.array([[0, 1],
#     [1, 1],
#     [1, 0],
#     [0, 0]])
#target = np.array([1,
#     0,
#     1,
#     0])
#mapper = [1]
s = datasets.load_digits()
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
n.addlayer(4, ninput=64, transfer=network.Sigmoid())
n.addlayer(10, transfer=network.Softmax(), cost=network.Cost.CrossEntropy())
e = 1.
i = 0
minibatchsize = 20

for i in range(1, 5000):
    print('\r'+str(i)+'      '+str(e), sep='', end='', flush=True)
    if e < 0.1:
      break
    e = n.sgd(X_train, y_train)
    #e = n.minibatch(data, target, minibatchsize)
    #e = n.batch(data, target)

    #y = n.forward(normalized)
    #n.backprop(tencoded, .001)
    #e = n.E

    #e = 0.5 * np.sqrt(e/nsamples)

for x in range(0, len(X_test)):
    print()
    o = n.forward(X_test[x])
    print("x={0} --> t={1} o={2}".format(X_test[x], y_test[x], o))

n.display(e, i)
