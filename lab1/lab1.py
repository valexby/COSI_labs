#!/usr/bin/env python

import math, cmath, pdb, sys
import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return math.cos(2 * x) + math.sin(5 * x)


def get_vector(func, size):
    step = (math.pi * 2) / size
    return [ func(x * step) for x in range(size) ]


def dft(vect):
    N = len(vect)
    W = cmath.exp(complex(0, ( -2 * math.pi)) / N)
    out = []
    for k in range( N ):
        out.append( sum( [ vect[i] * ( W **( k * i ) ) for i in range(N)] ) / N )
    return out


def rdft(C):
    N = len(C)
    W = cmath.exp(complex(0, (2 * math.pi)) / N)
    out = []
    for i in range(N):
        out.append( sum ( [C[k] * ( W ** ( k * i ) ) for k in range(N) ] ) )
    return out


def rft(C, x):
    C = C[:27]
    N = len(C)
    out = []
    for t in x:
        out.append( sum ( [ C[k] * cmath.exp(1j * k*t) for k in range(N)] ).real )
    return out;


def plot(figure, x, y, color):

    ax = figure.add_subplot(1, 1, 1)
    ax.plot(x, y, '-', color=color)


def fft(a, N):
    if len(a) == 1:
        return a
    a_even = a[::2]
    a_odd = a[1::2]

    b_even = fft(a_even, N // 2)
    b_odd = fft(a_odd, N // 2)

    W = cmath.exp((-2j * math.pi) / N)

    w = 1
    y = [0 for x in range(N)]
    for i in range ( N // 2 ):
       y[i] = (b_even[i] + w * b_odd[i])
       y[i + N // 2] = (b_even[i] - w * b_odd[i])
       w *= W

    return y


def main():
    N = 32
    diag = plt.figure()

    vect = get_vector(func, N)
    #C = fft(vect, N)
    #C = dft(vect)
    C = np.fft.fft(vect)
    
    x_org = np.arange(0, 2 * math.pi, 0.01)
    y_org = [func(i) for i in x_org]
    x_trans = [ (2 * i * math.pi) / N for i in range(N)]
    y_trans = [x.real/32 for x in rdft(C)]
    #y_trans = rft(C, x_org)
    #pdb.set_trace()
    plot(diag, x_org, y_org, 'blue')
    plot(diag, x_trans, y_trans, 'green')
    #plot(diag, x_org, y_trans, 'green')
    plt.savefig('out.png', fmt='png')

    return 0

if (__name__ == '__main__'):
    sys.exit(main())
