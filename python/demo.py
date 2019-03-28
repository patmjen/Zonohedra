import numpy as np

import pygorpho

vol = np.zeros((100,101,102), dtype=np.float32)
vol[50,50,50] = 1

strelSizes = [5, 9, 7]
strelSteps = [[1,0,0],[0,1,0],[0,0,1]]

strel = np.zeros((5,9,7))

print(vol[44:56,44:56,50])

res1 = pygorpho.flat_linear_dilate(vol, strelSizes, strelSteps, doPrint=True)
res2 = pygorpho.gen_dilate(vol, strel, doPrint=True)

print("res1:")
print(res1[44:56,44:56,50])
print("res2:")
print(res1[44:56,44:56,50])