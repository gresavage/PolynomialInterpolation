__author__ = 'Tom Gresavage'
'''
Polynomial Interpolation Code
'''

from numpy import *
from collections import defaultdict

a = [1, 2, 3, 4]
b = [2, 3, 2, 4]

# print "a[:]*b[:]= ", a[:]*b[:]
# print "a*b= ", a*b

def interpolate(x, y):
    c_ = list()
    for i in range(min(len(x), len(y))):
        c_.append(y[i])
        if i == 0: continue
        translate = [x[i]-x[j] for j in range(i)]
        translate.insert(0, 1)
        print translate
        c_[i] /= prod(translate)
        c_[i] -= sum([c_[k]*translate[k] for k in range(i)]/prod(translate))
    print i
    print c_
    def polynomial(t):
        COEFF = array([t - x[k] for k in range(i+1)])
        COEFF = concatenate((ones((1, shape(COEFF)[1])), COEFF), axis=0)
        z = list()
        for k in range(COEFF.shape[1]):
            print
            print k
            print array([prod(COEFF[:j, k]) for j in range(1, COEFF.shape[0])])
            z.append(array([prod(COEFF[:j, k]) for j in range(1, COEFF.shape[0])]).dot(array(c_)))
        print t
        print z
        return z
    return polynomial

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = [-2, -1, 0, 1, 2]
    y = [-5, -3, -15, 39, -9]
    print x, y
    print x.__len__(), y.__len__()
    interpolated = interpolate(x, y)
    t = linspace(-5, 5, 11)
    # u = interpolated(t)
    plt.plot(t, interpolated(t))
    plt.scatter(x, y)
    plt.show()