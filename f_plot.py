# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-07-17 18:09:08
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-07-18 10:14:20

import numpy as np
import matplotlib.pyplot as plt


t = np.arange(1.0, 10.0, 0.5)
s = 1-1/t
plt.plot(t, s, label='1-1/T')
s2 = 1-np.exp(-t/4)
plt.plot(t, s2, label='1-exp(-T/4)')

plt.xlim(10, 1)  # decreasing time

plt.xlabel('predicted VLT')
plt.ylabel('RDC reserve ratio')
# plt.title('Smaller VLT, smaller reserve ratio')
plt.grid(True)
plt.legend()
# plt.show()
plt.savefig('f_plot.png')