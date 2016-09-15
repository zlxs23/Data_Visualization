# _*_coding:utf-8_*_
# # y = (ax + b*1/s)^3
# import numpy as np
# import matplotlib.pyplot as plt

# x = np.linspace(0, 1, 50)
# y = np.cos(x) + 0.1 * np.random.rand(50)

# cof = np.polyfit(x, y, 3)

# p = np.poly1d(cof)

# plt.plot(x, y, 'o', x, p(x), lw=2)
from itertools import product

for m, n in product('abc', [1, 2]):
	print m, n

# product can get 笛卡尔积 (直积)
# for m in product('abc', repeat=4):
	# print m[0]

for m in product('a', repeat=4):
	print m