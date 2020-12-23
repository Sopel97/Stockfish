import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys
import glob

def read_stats(filename):
    ret = dict()
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            e, w, d, l = tuple(int(v) for v in line.split(';'))
            ret[e] = (w, d, l)
    return ret

def get_perf_binned(stats, bin_size=1):
    ret = dict()
    for e, v in stats.items():
        w, d, l = v
        e = e // bin_size * bin_size
        if e in ret:
            cur = ret[e]
            new = (ret[e][0] + w, ret[e][1] + d, ret[e][2] + l)
            ret[e] = new
        else:
            ret[e] = v
    return { k : (v[0] + v[1] * 0.5) / (v[0] + v[1] + v[2]) for k, v in ret.items() }

def get_counts_binned(stats, bin_size=1):
    ret = dict()
    for e, v in stats.items():
        c = sum(v)
        e = e // bin_size * bin_size
        if e in ret:
            ret[e] += c
        else:
            ret[e] = c
    return ret

def sigmoid(x, k):
    y = 1 / (1 + np.exp(-k*x))
    return (y)

def do_plot(filename):
    data = read_stats(filename)
    perfs = get_perf_binned(data, 16)
    counts = get_counts_binned(data, 16)

    fig, axs = plt.subplots(2)
    fig.tight_layout(pad=2.0)
    fig.suptitle(filename)
    x = list(counts.keys())
    y = [counts[k] for k in x]
    axs[0].plot(x, y)
    axs[0].set_ylabel('density')
    axs[0].set_xlabel('eval')

    x = list(perfs.keys())
    y = [perfs[k] for k in x]
    p0 = [1/361] # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, x, y, p0, method='dogbox')
    axs[1].scatter(x, y, label='perf')
    y = [sigmoid(xx, popt[0]) for xx in x]
    axs[1].scatter(x, y, label='sigmoid(x/{})'.format(1.0/popt[0]))
    axs[1].legend(loc="upper left")
    axs[1].set_ylabel('perf')
    axs[1].set_xlabel('eval')

    #plt.show()
    plt.savefig('.'.join(filename.split('.')[:-1]) + '.png')

for file in glob.glob('*_eval_stats.txt'):
    do_plot(file)
