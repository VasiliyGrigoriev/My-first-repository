import matplotlib.pyplot as plt
import numpy as np

plt.figure(1)
plt.plot(np.load("td.npy"), np.load("cd.npy"))
plt.xlabel("$t$")
plt.ylabel("$Concentration$")
plt.show()
