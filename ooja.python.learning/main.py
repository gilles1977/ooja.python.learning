from ooja import network
import numpy as np
import matplotlib.pyplot as plt
import netdraw as nd
from sklearn import datasets

data = np.array([[0, 1],
     [1, 1],
     [1, 0],
     [0, 0]])
target = np.array([1,
     0,
     1,
     0])
mapper = [1]
#s = datasets.load_iris()
#data = s.data
#target = s.target
#mapper = np.arange(3)
ni = data.shape[1]
no = len(mapper)
nsamples = len(data)

shape = np.array([ni, 5, no])
n = network.Network(shape, network.Sigmoid(), lambda o: (o >= 0.5) * 1)


e = 1.
i = 0
for i in range(1, 5000):
    print('\r'+str(i)+'      '+str(e), sep='', end='', flush=True)
    if e < 0.06:
       break
    e = 0.
    for x in range(0, len(data)):
        o = n.forward(data[x])
        t = (target[x] == mapper) * 1
        n.backprop(t, e)
        e += np.mean(np.square(n.e))
    e = np.sqrt(e/nsamples)

for x in range(0, len(data)):
    print()
    o = n.forward(data[x])
    t = (target[x] == mapper) * 1
    print("x={0} --> t={1} o={2}".format(data[x], t, o))

n.display(e, i)
