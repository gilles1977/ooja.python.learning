from ooja import network
from numpy import random
from numpy import array

shape = array([2, 3, 2, 1])
n = network.Network(shape, network.Sigmoid())
x = array([0, 1])
o = n.forward(x)

for l in n.layers:
    l.display()
print("o={0}".format(o))

n.backprop(1)
for l in n.layers:
    l.display()

#n = network.Node(1, f, 0.1)
#l = network.Link()
#l.x = array([0, 1])
#l.w = random.rand(2)
#n.input = l
#s = n.output()

#print(l.x)
#print(l.w)
#print(s)