from ooja import network
import numpy as np
import matplotlib.pyplot as plt
import netdraw as nd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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
mapper = np.unique(target)
#mapper = np.arange(3)
ni = data.shape[1]
no = len(mapper)
nsamples = len(data)

scaler = MinMaxScaler()
normalized = scaler.fit_transform(data, target)

onehot = OneHotEncoder(sparse=False, categories='auto')
tencoded = onehot.fit_transform(target.reshape(len(target), 1))

n = network.Network(lambda o: np.round(o, 2))
n.addlayer(7, ninput=4, transfer=network.Sigmoid())
n.addlayer(3, transfer=network.Sigmoid())

e = 1.
i = 0
for i in range(1, 5000):
    print('\r'+str(i)+'      '+str(e), sep='', end='', flush=True)
    if e < 0.01:
       break
    e = 0.
    #for x in range(0, len(normalized)):
    y = n.forward(normalized)
        
    #t = (target == mapper) * 1
    n.backprop(tencoded, .1)
    e = n.E / nsamples
    #e = 0.5 * np.sqrt(e/nsamples)

for x in range(0, len(normalized)):
    print()
    o = n.forward(normalized[x])
    t = (target[x] == mapper) * 1
    print("x={0} --> t={1} o={2}".format(normalized[x], t, o))

n.display(e, i)
