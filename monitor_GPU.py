import time
from pynvml import *

NVML_DEVICE_INDEX = 3
DT = 0.02   # 20ms
T  = 120.0   # seconds

nvmlInit()
h = nvmlDeviceGetHandleByIndex(NVML_DEVICE_INDEX)

vals = []
t0 = time.time()
while time.time() - t0 < T:
    util = nvmlDeviceGetUtilizationRates(h).gpu  # %
    vals.append(util)
    time.sleep(DT)

print("avg gpu util:", sum(vals)/len(vals))
nvmlShutdown()