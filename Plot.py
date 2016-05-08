#!/usr/bin/env python

import argparse
import sys
import re
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Parse mxnet output log')
parser.add_argument('logfile', nargs=1, type=str,
                    help = 'the log file for parsing')
parser.add_argument('--format', type=str, default='plot',
                    choices = ['markdown', 'none', 'plot'],
                    help = 'the format of the parsed outout')
args = parser.parse_args()

with open(args.logfile[0]) as f:
    lines = f.readlines()

res = [re.compile('.*Epoch\[(\d+)\] Train.*=([.\d]+)'),
       re.compile('.*Epoch\[(\d+)\] Valid.*=([.\d]+)'),
       re.compile('.*Epoch\[(\d+)\] Time.*=([.\d]+)')]

data = {}

epoch = 1
for l in lines:
    i = 0
    for r in res:
        m = r.match(l)
        if m is not None:
            break
        i += 1
    if m is None:
        continue

    assert len(m.groups()) == 2
    #epoch = int(m.groups()[0])
    val = float(m.groups()[1])

    if epoch not in data:
        epoch = epoch + 1
        data[epoch] = [0] * len(res) * 2


    data[epoch][i*2] += val
    data[epoch][i*2+1] += 1
    
if args.format == 'markdown':
    print "| epoch | train-accuracy | valid-accuracy | time |"
    print "| --- | --- | --- | --- |"
    for k, v in data.items():
        print "| %2d | %f | %f | %.1f |" % (k+1, v[0]/v[1] if v[1] != 0 else 0, v[2]/v[3] if v[3] != 0 else 0, v[4]/v[5] if v[5] != 0 else 0)
elif args.format == 'none':
    print "epoch\ttrain-accuracy\tvalid-accuracy\ttime"
    for k, v in data.items():
        print "%2d\t%f\t%f\t%.1f" % (k+1, v[0]/v[1] if v[1] != 0 else 0, v[2]/v[3] if v[3] != 0 else 0, v[4]/v[5] if v[5] != 0 else 0)
elif args.format == 'plot':
    epochs = []
    train_score = []
    validation_score = []
    times = []

    for k, v in data.items():
        epochs.append(k+1)
        train_score.append(v[0]/v[1] if v[1] != 0 else 0)
        validation_score.append(v[2]/v[3] if v[3] != 0 else 0)
        times.append(v[4]/v[5] if v[5] != 0 else 0)

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.plot(epochs, train_score, 'r', label='train')
    plt.plot(epochs, validation_score, 'b', label='validation')
    plt.legend(loc='upper right')
    plt.show()
