#!/usr/bin/env python
import argparse
import numpy as np
import zmq
from rough_calibration import *
from os.path import exists

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--line_names', nargs='*', default=['ck', 'nk', 'ok', 'fela', 'nila', 'cula'])
parser.add_argument('-n', '--npulses', default=300000, type=int)
parser.add_argument('-f', '--filename', default='cal_dict.pkl')
parser.add_argument('--overwrite', action='store_true')

args = parser.parse_args()

address = "10.66.48.41"
sub_port= 5504

ctx = zmq.Context()
socket = ctx.socket(zmq.SUB)
socket.connect(f"tcp://{address}:{sub_port}")
socket.subscribe('')

dt1 = np.dtype([('channum', np.uint16), ('header version', np.uint16), ('presamples', np.uint32), ('length', np.uint32), ('preTrigMean', np.float32), ('peakValue', np.float32), ('pulseRMS', np.float32), ('pulseAverage', np.float32), ('residualStdDev', np.float32), ('trigTime', np.uint64), ('trigFrame', np.uint64)])

def take_data(npulses):
    data = {}
    for n in range(npulses):
        msg = socket.recv_multipart()
        summary = np.frombuffer(msg[0], dtype=dt1)
        channum = summary['channum'][0]
        if channum not in data:
            data[channum] = []
        data[channum].append(summary['pulseRMS'][0])
    return data

data = take_data(args.npulses)
energies = get_line_energies(args.line_names)
cal_dict = make_cal_dict(data, len(energies))
cal_dict['energies'] = energies

if not exists(args.filename) or args.overwrite:
    with open(args.filename, 'wb') as f:
        pickle.dump(cal_dict, f)
