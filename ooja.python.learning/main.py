from ooja import network
import numpy as np

shape = np.array([2, 2, 1])
n = network.Network(shape, network.Sigmoid(), lambda o: 0 if o < 0.5 else 1)
s = np.array([[0, 1, 1],
           [1, 1, 0],
           [1, 0, 1],
           [0, 0, 0]])
e = 1.
for l in n.layers:
        l.display()

for i in range(1, 5000):
    print('\r'+str(i)+'      '+str(e), sep='', end='', flush=True)
    if e < 0.1:
       break
    e = 0.
    for x in s:
        o = n.forward(x[:2].T)
        e += np.power(n.e, 2)
        n.backprop(x[2])
    e = 0.5 * np.sqrt(e / s.size)

for x in s:
    print()
    o = n.forward(x[:2].T)
    print("x={0}".format(x[:2]))
    print("o={0}".format(o))

for l in n.layers:
    l.display()
