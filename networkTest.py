#Implementation of a neural network, drawing heavily from 
#http://outlace.com/Beginner-Tutorial-Theano/

import theano.tensor as T
import theano.tensor.nlinalg
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import theano
import matplotlib.pyplot as plt

def layer(x, w):
    bias = np.array([1], dtype=theano.config.floatX)
    x = T.concatenate([bias, x])
    s = T.dot(w.T, x)
    return T.tanh(s)

def gradient_descent(cost, weight):
    learn_rate = .01
    return weight - (learn_rate * T.grad(cost, wrt=weight))


x = T.dvector('x')
y = T.dscalar('y')

w1 = theano.shared(np.random.rand(3,3))
w2 = theano.shared(np.random.rand(4,1))

#hidden layer
hidden_1 = layer(x, w1)
#Output layer
out = T.sum(layer(hidden_1, w2))

#Squared error function (our cost function)
sq_err = (out - y) ** 2

#Training function
cost = function([x, y], outputs=sq_err, updates=[
    (w1, gradient_descent(sq_err, w1)),
    (w2, gradient_descent(sq_err, w2))])

run = function([x], out)


data = np.array([[1,0], [0,1], [0,0], [1,1]])
labels = np.array([1,1,0,0])

iterations = 10000
for i in range(iterations):
    for index in range(len(data)):
        err = cost(data[index], labels[index])
    if i % 1000 == 0:
        print(err)
        print(w1.get_value())

print(run(np.array([1,0])))
