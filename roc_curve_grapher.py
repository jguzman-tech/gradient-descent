import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
from datetime import datetime

from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
matplotlib.rc('font', size=24)

linear = np.linspace(0, 1, 1000)
plt.plot(linear, linear, linestyle='--', color="black")

for i in range(1, len(sys.argv)):
    data = np.load(sys.argv[i])
    fpr = data[:, 0]
    tpr = data[:, 1]
    label_name = sys.argv[i]
    label_name = label_name.replace('./roc-data/', '')
    label_name = label_name.replace('.npy', '')
    plt.plot(fpr, tpr, label=label_name, linewidth=3)

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
fname = "./figures/"
fname += datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
fname += "_roc_curve.png"
plt.savefig(fname)

print("Figure created:")
print(fname)
