"""
export EPICS_CA_AUTO_ADDR_LIST=no
export EPICS_CAS_AUTO_BEACON_ADDR_LIST=no
export EPICS_CAS_BEACON_ADDR_LIST=10.66.51.255
export EPICS_CA_ADDR_LIST=10.66.51.255
"""


from caproto.server import PVGroup, ioc_arg_parser, pvproperty, run
from caproto import ChannelType
import asyncio
import zmq.asyncio
import zmq
from textwrap import dedent
import time
import json
import numpy as np
import pickle
from os.path import exists
from rough_calibration import *

class MCA(PVGroup):
    """
    A class to read ZMQ pulse info from TES
    """
    MAXBINS = 10000
    DEFAULT_LLIM = 200
    DEFAULT_ULIM = 1000
    DEFAULT_NBINS = 800
    COUNTS = pvproperty(value=0, record='ai', dtype=int, doc="ROI Counts")
    SPECTRUM = pvproperty(value=np.zeros(MAXBINS, dtype=int), dtype=int, doc="ROI Histogram")
    LLIM = pvproperty(value=DEFAULT_LLIM, record='ai', doc="ROI lower limit")
    ULIM = pvproperty(value=DEFAULT_ULIM, record='ai', doc='ROI upper limit')
    NBINS = pvproperty(value=DEFAULT_NBINS, record='ai', doc="ROI resolution")
    CENTERS = pvproperty(value=np.zeros(MAXBINS, dtype=float), dtype=float)
    COUNT_TIME = pvproperty(value=1.0, record='ai', doc='ROI Count Time')
    ACQUIRE = pvproperty(value=0, doc="ACQUIRE")
    LOAD_CAL = pvproperty(value=0)
    MAKE_CAL = pvproperty(value=0)

    def __init__(self, *args, address="10.66.48.41", sub_port=5504,  **kwargs):
        self.address = address
        self.sub_port = sub_port
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.connect(f"tcp://{self.address}:{self.sub_port}")
        self.socket.subscribe("")
        self._buffer = []
        self._start_ts = time.time()
        self._dt1 = np.dtype([('channum', np.uint16), ('header version', np.uint16), ('presamples', np.uint32), ('length', np.uint32), ('preTrigMean', np.float32), ('peakValue', np.float32), ('pulseRMS', np.float32), ('pulseAverage', np.float32), ('residualStdDev', np.float32), ('trigTime', np.uint64), ('trigFrame', np.uint64)])
        self._dt2 = np.float64
        self._cal_file_name = "cal_dict.pkl"
        self._cal_loaded = False
        self._poly_dict = {}
        self._cal_data = {}
        self._acquire_cal = 0
        
        self._bins = np.linspace(self.DEFAULT_LLIM, self.DEFAULT_ULIM, self.DEFAULT_NBINS + 1)
        super().__init__(*args, **kwargs)

    @ACQUIRE.putter
    async def ACQUIRE(self, instance, value):
        if value != 0:
            self._buffer = []
            self._start_ts = time.time()
        return value

    @LLIM.putter
    async def LLIM(self, instance, value):
        self._bins = np.linspace(value, self.ULIM.value, self.NBINS.value + 1)
        centers = (self._bins[1:] + self._bins[:-1])*0.5
        await self.CENTERS.write(centers)

    @ULIM.putter
    async def ULIM(self, instance, value):
        self._bins = np.linspace(self.LLIM.value, value, self.NBINS.value + 1)
        centers = (self._bins[1:] + self._bins[:-1])*0.5
        await self.CENTERS.write(centers)

    @NBINS.putter
    async def NBINS(self, instance, value):
        if value > self.MAXBINS:
            value = self.MAXBINS
        self._bins = np.linspace(self.LLIM.value, self.ULIM.value, value + 1)
        centers = (self._bins[1:] + self._bins[:-1])*0.5
        await self.CENTERS.write(centers)

    def load_cal_file(self, cal_file_name):
        with open(cal_file_name, 'rb') as f:
            self._cal_dict = pickle.load(f)
        self._poly_dict = {}
        self._cal_energies = self._cal_dict.pop('energies')
        polydegree = min(4, len(self._cal_energies) - 1)
        y = np.insert(self._cal_energies, 0, 0.0)
        for k, peaks in self._cal_dict.items():
            x = np.insert(peaks, 0, 0.0)
            self._poly_dict[k] = np.poly1d(np.polyfit(x, y, polydegree))
        self._cal_loaded = True
        
    @LOAD_CAL.putter
    async def LOAD_CAL(self, instance, value):
        if value != 0:
            self.load_cal_file(self._cal_file_name)

    @MAKE_CAL.putter
    async def MAKE_CAL(self, instance, value):
        self._cal_data = {}
        self._cal_pulses = 0
        self._acquire_cal = 1
        await instance.write(value, verify_value=False)
        print("In make cal")
        while self._cal_pulses < value:
            await asyncio.sleep(5)
            print(self._cal_pulses)
        self._acquire_cal = 0
        energies = get_line_energies(['ck', 'nk', 'ok', 'fela', 'nila', 'cula'])
        cal_dict = make_cal_dict(self._cal_data, len(energies))
        cal_dict['energies'] = energies
        with open(self._cal_file_name, 'wb') as f:
            pickle.dump(cal_dict, f)
        self.load_cal_file(self._cal_file_name)
        return 0

            
    @CENTERS.startup
    async def CENTERS(self, instance, async_lib):
        centers = (self._bins[1:] + self._bins[:-1])*0.5
        await self.CENTERS.write(centers)
        
        
    @ACQUIRE.startup
    async def ACQUIRE(self, instance, async_lib):
        self._start_ts = time.time()
        self._buffer = []

        while True:
            if self.ACQUIRE.value != 0 and self._start_ts + self.COUNT_TIME.value < time.time():
                data = np.array(self._buffer)
                await self.COUNTS.write(len(self._buffer))
                if self._cal_loaded:
                    counts, _ = np.histogram(data, bins=self._bins)
                    #await self.PFY1.write(np.sum((self._buffer < self.PFY1_ULIM.value) & (self._buffer > self.PFY1_LLIM.value)))
                    await self.SPECTRUM.write(counts)
                self._buffer = []
                self._start_ts = time.time()
                if self.ACQUIRE.value > 0:
                    await self.ACQUIRE.write(self.ACQUIRE.value - 1)
            await async_lib.sleep(.05)

    async def __ainit__(self, async_lib):
        print('* `__ainit__` startup hook called')
        if exists(self._cal_file_name):
            await self.LOAD_CAL.write(1)

        while True:
            msg = await self.socket.recv_multipart()
            data = self.decode_msg(msg)
            if self.ACQUIRE.value != 0:
                #if data['channum'] == 1:
                e = self.convert_to_energy(data)
                self._buffer.append(e)
            if self.MAKE_CAL.value != 0 and self._acquire_cal != 0:
                channum = data['channum'][0]
                if channum not in self._cal_data:
                    self._cal_data[channum] = []
                self._cal_data[channum].append(data['pulseRMS'][0])
                self._cal_pulses += 1

    def decode_msg(self, msg):
        summaries = np.frombuffer(msg[0], dtype=self._dt1)
        return summaries

    def convert_to_energy(self, data):
        if self._cal_loaded:
            channum = data['channum'][0]
            try:
                return self._poly_dict[channum](data['pulseRMS'][0])
            except:
                return -1
        else:
            return -1    

if __name__ == "__main__":
    ioc_options, run_options = ioc_arg_parser(default_prefix="XF:07ID-ES{{UCAL:ROIS}}:",
                                              desc = dedent(MCA.__doc__),
                                              supported_async_libs=('asyncio',))
    ioc = MCA(**ioc_options)
    run(ioc.pvdb, startup_hook=ioc.__ainit__, **run_options)
