__author__ = 'Tom Gresavage'

import sys
import os
import math
import time
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def polynomial(coefficients, name="f"):
    # Creates a polynomial whose coefficients of increasing orders of x are given by "coefficients"
    def function(x):
        y = 0
        for i in range(len(coefficients)):
            y += coefficients[i]*np.power(x, i)
        return y
    function.__name__ = name
    function.coefficients = coefficients

    return function

def polyderiv(polynomial, d=1):
    # Calculates the "d"th derivative of "polynomial"
    __ = polynomial.coefficients
    for j in range(d):
        coefficients = []
        for i in range(len(__)-1):
            coefficients.append(__[i+1]*(i+1))
        __ = coefficients

    def function(x):
        y = 0
        for i in range(len(coefficients)):
            y += coefficients[i]*np.power(x, i)
        return y
    function.__name__ = polynomial.__name__ + "p"*d
    function.coefficients = coefficients
    return function

def make_splines(x, y, units='units'):
    '''
    Creates a set of natural cubic splines from x and y and calculates the length of the spline
    Returns x's and y's which are the coordinates of the splines, length of the spline, spline polynomial functions

    Inputs:
            x:  vector of x's to be used as spline endpoints
            y:  vector of y's to be used as spline endpoints
        units:  string denoting units of length, 'units' by default

    Outputs:
            _x: vector of x points on splines
            _y: vector of y points on splines
         s_int: length of spline
             s: list of polynomial spline functions
    '''

    n = len(x)-1
    S = np.zeros((4*n,4*n))
    C = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 2, 3], [0, 0, 1, 3]])
    b = np.zeros(4*n)

    '''
    Create a matrix whose (i,j)th entry is x[i]^j
    '''
    X = np.array([list(x) for i in range(4)]).transpose()
    for i in range(4):
        X[:, i] = X[:, i]**i

    '''
    Create a coefficient and solution matrix for the cubic spline system
    Assume naturual spline condition
    '''
    for i in range(n):
        S[4*i:4*i+2, 4*i:4*(i+1)] = C[:2, :]*X[i:i+2, :]
        b[4*i:4*i+2] = y[i:i+2]
        if i != n-1:
            for j in range(2):
                S[4*i+2+j, 4*i+1+j:4*(i+1)] = C[2+j, (1+j):]*X[i+1, :-1-j]
                S[4*i+2+j, (4*(i+1)+1+j):4*(i+2)] = -C[2+j, (1+j):]*X[i+1, :-1-j]
        elif i == n-1:
            S[4*i+2, 4*i+1:4*(i+1)] = C[3, 1:]*X[i+1, :-1]
            S[4*i+2, 1:4] = -C[2, 1:]*X[i+1, :-1]
            S[4*i+3, 4*i+2:4*(i+1)] = C[3, 2:]*X[i+1, :-2]
    splines = la.solve(S,b) # Coefficients of cubic splines
    print splines
    '''
    Create polynomials from the splines' coefficients
    Plot the results with a unique color for each spline
    '''
    s = [polynomial(splines[4*i:4*i+4]) for i in range(n)]
    t = [np.linspace(x[i], x[i+1]) for i in range(len(x)-1)]
    _x = list()
    _y = list()
    fig = plt.figure(1)
    fig.suptitle('The Ant\'s Path')
    ax = fig.add_subplot(111)

    s_der = [polyderiv(s[i]) for i in range(n)]
    quad_x = [0., np.sqrt(5.-2.*np.sqrt(10./7.))/3., -np.sqrt(5.-2.*np.sqrt(10./7.))/3., np.sqrt(5.+2.*np.sqrt(10./7.))/3., -np.sqrt(5.+2.*np.sqrt(10./7.))/3.]
    quad_w = [128./225., (322.+13.*np.sqrt(70))/900., (322.+13.*np.sqrt(70))/900., (322.-13.*np.sqrt(70))/900., (322.-13.*np.sqrt(70))/900.]
    s_int = 0

    for i in range(n):
        _x.extend(list(t[i]))
        _y.extend(list(s[i](t[i])))
        ax.plot(t[i], s[i](t[i]), label='Spline %s' %i)
        '''
        Create derivatives of each spline
        Calculate the curve length using 5-point Gaussian Quadrature
        Print the result
        '''
        ax.text(x[i], y[i], '%s %s' %("{:.2f}".format(s_int), units))
        s_int += ((x[i+1]-x[i])/2.)*np.sum([quad_w[j]*np.sqrt(1.+(s_der[i](((x[i+1]-x[i])/2.)*quad_x[j]+((x[i+1]+x[i])/2.)))**2.) for j in range(5)])
    ax.text(x[-1], y[-1], '%s %s' %("{:.2f}".format(s_int), units))
    ax.scatter(x,y)
    ax.legend(loc=4)
    ax.set_xlabel('Location (cm)')
    ax.set_ylabel('Location (cm)')
    plt.show()

    print "The length of the curve is %s %s" %('{:.2f}'.format(s_int), units)

    return _x, _y, s_int, s


x = np.array([0, 2, 4, 6, 8])
y = np.array([2, -3, -2, 3, 6])

print make_splines(x, y, 'cm')