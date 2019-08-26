from ooja import network
from numpy import random
from numpy import array

f = lambda x: 0 if x < 0 else 1

n = network.Node(1, f, 0.1)
l = network.Link()
l.x = array([0, 1])
l.w = random.rand(2)
n.input = l
s = n.output()

print(l.x)
print(l.w)
print(s)