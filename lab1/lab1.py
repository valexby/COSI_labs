#!/usr/bin/env python

import math, cmath, pdb, sys
import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return [math.cos(2 * i) + math.sin(5 * i) for i in x]


def get_vector(func, size):
    step = (math.pi * 2) / size
    return func([x * step for x in range(32)])


def dft(vect):
    N = len(vect)
    C = []
    for k in range(N):
        C.append(0)
        for i in range(N):
            W = cmath.exp( -2j * math.pi * k * i / N)
            C[k] += vect[i] * W
        C[k] /= N
    return C 


def idft(C):
    N = len(C)
    f = []
    for i in range(N):
        f.append(0)
        for k in range(N):
            W = cmath.exp(2j * math.pi * i * k / N)
            f[i] += C[k] * W
    return f


def ift(C, x):
    C = C[:(len(C)//2)+1]
    N = len(C)
    f = []
    for t in x:
        W = cmath.exp(1j * t) 
        f.append( sum ( [ C[k] * (W ** k) for k in range(N)] ).real*2 )
    return f;


def fft(a, direction = 1, N = 32):
    if len(a) == 1:
        return a
    a_even = a[::2]
    a_odd = a[1::2]

    b_even = fft(a_even, direction, N // 2)
    b_odd = fft(a_odd, direction, N // 2)

    W = cmath.exp((-2j * direction * math.pi) / N)

    w = 1
    y = [0 for x in range(N)]
    for i in range ( N // 2 ):
       y[i] = (b_even[i] + w * b_odd[i])
       y[i + N // 2] = (b_even[i] - w * b_odd[i])
       w *= W

    return y


def rademaher(k, t):
    sin_out = math.sin(2 ** k * math.pi * t)
    if (sin_out > 0): return 1
    else : return -1


def walsh(k, t):
    bins = [int(i) for i in bin(k)[2:]]
    N = len(bins)
    bins = [0] + bins
    powers = np.zeros(N)
    for i in range(N):
        powers[i] = bins[N - i] ^ bins[N - (i + 1)]

    out = 1
    for i in range(N):
        out *= rademaher(i+1, t) ** powers[i]
    return out


def fut(a, N = 32):
    if N == 1:
        return [a[0] / 8]
    out = [0 for x in range(N)]
    for i in range(N // 2):
        out[i] = a[i] + a[i + N // 2]
        out[i + N // 2] = a[i] - a[i + N // 2]
    return fut(out[:N // 2], N // 2) + fut(out[N // 2:], N // 2)


def rut(C, t):
    out = 0
    for i in range(len(C)):
        out +=sum([C[i] * walsh(k, t) for k in range(len(C))])
    return out

def print_discret(vect, ax, color):
    N = len(vect)
    C = fft(vect)

    x = [ (2 * i * math.pi) / N for i in range(N)]
    y = [i.real/N for i in fft(C, direction=-1)]

    ax.plot(x, y, '-', color=color)


def print_func(ax, color):
    x = np.arange(0, 2 * math.pi, 0.01)
    y = func(x)

    ax.plot(x, y, '-', color=color)


def print_tans_func(vect, ax, color):
    N = len(vect)
    C = dft(vect)
    
    x = np.arange(0, 2 * math.pi, 0.01)
    #x = [ (2 * i * math.pi) / N for i in range(N)]
    #y = [i.real for i in idft(C)]
    y = ift(C, x)

    ax.plot(x, y, '-', color=color)


def print_components(vect, ax):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    N = len(vect)
    C = dft(vect)
    x = np.arange(0, 2 * math.pi, 0.01)
    
    for k in range(N // 2):
        if abs(C[k].real) > 0.1 or abs(C[k].imag) > 0.1:
            y = [(C[k] * cmath.exp(1j * t * k)).real * 2 for t in x]
            ax.plot(x, y, '-.', color=colors[0])
            colors = colors[1:]


def main():
    N = 32
    fig, ax = plt.subplots(2, 1)
    #fig, ax = plt.subplots(4, 1)
    vect = get_vector(func, N)
    C = fut(vect)
    #ax[4].stem([abs(x) for x in dft(vect)], linefmt='r-', color='m')
    x = np.arange(0, 1, 0.01)
    y = [rut(C, t) for t in x]
    pdb.set_trace()
    ax[0].plot(x, y, '-', color='b')
    x = np.arange(0, 2*math.pi, 0.01)
    ax[1].plot(x, func(x), '-', color='g')

    plt.savefig('out.png', fmt='png')

    return 0

if (__name__ == '__main__'):
    sys.exit(main())
