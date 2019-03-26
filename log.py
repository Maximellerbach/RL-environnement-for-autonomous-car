import numpy as np
import matplotlib.pyplot as plt

log = np.load('log.npy')

plt.plot(log)

plt.ylabel('score logs')
plt.show()