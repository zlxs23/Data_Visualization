'''
3.2.2
定义图表类型 -- 柱状图，线形图 AND 堆积柱状图
'''
plot([1, 2, 3, 3, 2, 3, 3, 2, 2, 1])
plot([4, 3, 2, 1], [1, 2, 3, 4])

from matplotlib.pyplot import *

# some simple data
x = [1, 2, 3, 4]
y = [5, 4, 3, 2]

# create new figure
figure()

# divide subplots into 2 to 3 grid
# select #1 线性图
subplot(231)
plot(x, y)

# select #2 竖向柱状图（默认）
subplot(232)
bar(x, y)

# select #3 横向柱状图
# horizontal bar-charts
subplot(233)
barh(x, y)

# select #4 堆叠柱状图
# stacked bar charts
subplot(234)
bar(x, y)
# more data for stacked bar charts
y1 = [7, 8, 5, 3]
bar(x, y1, bottom=y, color='r')    # 设置 argument bottom = y [底部为 y 则 y1 在上部]

# select #5 箱线图
# box plot
subplot(235)
boxplot(x)

# select #6 散点图
# scatter plot
subplot(236)
scatter(x, y)

'''
用统一数据来绘制比较箱线图 AND 直方图
'''

from pylab import *

dataset = [113, 115, 119, 121, 124,
           124, 125, 126, 126, 126,
           127, 127, 128, 129, 130,
           130, 131, 132, 133, 136]
subplot(121)
boxplot(dataset, vert=False)
subplot(122)
hist(dataset)

'''
3.3
简单的正弦图 AND 余弦图
From -pi to pi 具有相同的线性距离 の 256 个点来计算正弦值 AND 余弦值
'''
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
yc = np.cos(x)
ys = np.sin(x)

plt.plot(x, yc)
plt.plot(x, ys)

plt.show()
'''
以上述简单图表为基础 进一步定制化添加更多信息
同时让坐标轴 AND 边界更加精确
'''
from pylab import *

# generator uniformly distribute
# 256 points from -pi to pi,inclusive
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)

# these are bectorised versions of math.cos and math.sin in built-in python maths
# compute cos for every x
yc = np.cos(x)
# compute sin for every x
ys = np.sin(x)

# plot cos
plot(x, yc)
# plot sin
plot(x, ys)

# define plot title
title('Functions $\sin$ and $\cos$')
# set x,y limit
xlim(-3.0, 3.9)
ylim(-1.0, 1.0)

# format ticks at specific values
xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
       [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
yticks([-1, 0, +1],
       [r'$-1$', r'$0$', r'$1$'])
show()

'''
改变线 の 属性
'''
# way 1
plot(x, y, linewidth=1.5)
# way 2
line = plot(x, y)
line.set_linewidth(1.5)
# way 3
line = plot(x, y)
setp(line, 'linewidth', 1.5)    # OR
setp(line, linewidth=1.5)

'''
3.6.2
使用 matplotlib.pyplot.locator_params() 方法控制刻度容器定位器 の 行为
刻度位置通常会自动确定下来 我们还是可控刻度的数目 以及在 plot 比较小时使用一个 紧凑 试图 tight view
'''
from pylab import *
# get current axis
ax = gca()
# set view to tight and maxium number of tick intervals to 10
ax.locator_params(tight=True, nbins=10)
# generator 100 normal distribution values
ax.plot(np.random.normal(10, .1, 100))
show()

'''
使用 matplotlib.ticker.FormatStrFormatter 刻度格式器 规定 刻度 值(通常是 数字)的显示方式
Eg: 可方便指定   '%2.1f' OR '%1.1f cm' 格式字符串 作为刻度标签
这里使用 dates module 来 实现时间刻度
'''
from pylab import *
import matplotlib as mpl
import datetime

fig = figure()

# get current axis
ax = gca()

# set some daterange
start = datetime.datetime(2016, 1, 1)
stop = datetime.datetime(2016, 12, 31)
delta = datetime.timedelta(days=1)

# convert dates for matplotlib
dates = mpl.dates.drange(start, stop, delta)

# generator some random values
values = np.random.rand(len(dates))

ax = gca()

# create plot with dates
ax.plot_date(dates, values, linestyle='-', marker='')

# specify formater
date_format = mpl.dates.DateFormatter('%Y-%m-%d')

# apply formater
ax.xaxis.set_major_formatter(date_format)

# atuoformat date labels
# rotates labels by 30 degrees by default
# use rotate param to specify different rotation degree
# use bottom param to give more room to date labels
fig.autofmt_xdate()
'''
3.7.2
添加图例 AND 注释
'''
from matplotlib.pyplot import *

# generator different normal distributions
x1 = np.random.normal(30, 3, 100)
x2 = np.random.normal(20, 3, 100)
x3 = np.random.normal(10, 3, 100)

# plot them
plot(x1, label='plot')
plot(x2, label='2nd plot')
plot(x3, label='3rd plot')

# generaator a legend box
legend(bbox_to_anchor=(0., 1.02, 1., 102), loc=3,
       ncol=3, mode='expand', borderaxespad=0.)

# annotate an import value
annotate('Important value', (55, 20), xycoords='data',
         xytext=(5, 38), arrowprops=dict(arrowstyle='->'))
show()

'''
3.8.2
移动轴线到图中央
因为普通 图 都是以 方形 而不是 十字形
方形：上下左右都有线 刻度是在 下 AND 左
十字形： 只有 中间 有两线呈十字 其他地方无线
为了调整成 十字形 则将方形 上 右 两线不显示|再将下 左 两线移动到中间
'''
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 500, endpoint=True)
y = np.sin(x)

plt.plot(x, y)
ax = plt.gca()

# hide two spines
ax.spines['right'].set_color('none')  # 隐藏 方形图 右边线
ax.spines['top'].set_color('none')  # 隐藏 方形图 上边线

# move bottom and left spine to 0,0
ax.spines['bottom'].set_position(('data', 0))  # 将 方形图 下边线 移动到 0
ax.spines['left'].set_position(('data', 0))  # 将 方形图 左边线 移动到 0

# move ticks posittions
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.show()
# 可将轴线限制在 数据 结束的地方结束 可用 set_smart_bound(True)
# 这时 matplotlib 尝试以一种复杂的方式设置边界 在数据延伸出试图的情况 裁剪线条以适应视图
'''
3.9
绘制直方图
直方图 被用于 可视化的分布估计
表示一定间隔下数据点的垂直矩形称为 bin 以固定间隔创建 SO 直方图的总面积等于数据点的数量
直方图可显示数据的相对频率 而不是使用数据的绝对值
'''
import numpy as np
import matplotlib.pyplot as plt

mu = 100
sigma = 15
x = np.random.normal(mu, sigma, 10000)
ax = plt.gca()

# the histogram of the data
ax.hist(x, bins=35, normed=True, histtype=step, color='r')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.set_title(r'$\mathrm{Histogram:}\ \mu=%d,\ \sigma=%d$' % (mu, sigma))
plt.show()

'''
3.10
绘制 误差条形图
'''
import numpy as np
import matplotlib.pyplot as plt

# generate number of measurements
x = np.arange(0, 10, 1)

# valueds computed from 'measured'
y = np.log(x)

# add some error samples from standard normal distribution
xe = 0.1 * np.abs(np.random.randn(len(y)))

# draw and show errobar
plt.bar(x, y, yerr=xe, width=0.4, align='center',
        ecolor='r', color='cyan', label='experiment # 1')
# give some explaintions
plt.xlabel('# measurement')
plt.ylabel('Measured values')
plt.title('Measurements')
plt.legend(loc='upper left')
plt.show()

'''
绘制饼图
'''
from pylab import *

# make a square and axes
figure(1, figsize=(6, 6))
ax = axes([.1, .1, .8, .8])

# the slices will be ordered
# and plotted counter-clockwise
labels = 'Sqring', 'Summer', 'Autumn', 'Winter'

# fractions are either x/sum(x) or x if sum(x) <= 1
x = [155, 30, 45, 10]

# explode must be len(x) sequence or None
explode = (.1, .1, .1, .1)

pie(x, explode=explode, labels=labels, autopct='%1.1f%%', startangle=67)

title('Rainy days be season')
show()

'''
绘制填充区域的图表
'''

from matplotlib.pyplot import figure, show, gca
import numpy as np

x = np.arange(0.0, 2, .01)
# two diff signals are measured
y1 = np.sin(2*np.pi*x)
y2 = 1.2*np.sin(4*np.pi*x)

fig = figure()
ax = gca()

# plot and fill between y1 and y2 where a logical condition is met
ax.plot(x, y1, x, y2, color='black')

ax.fill_between(
    x, y1, y2, where=y2 >= y1, facecolor='darkblue', interpolate=True)
ax.fill_between(
    x, y1, y2, where=y2 <= y1, facecolor='deeppink', interpolate=True)

ax.set_title('filled between')
show()
'''
绘制带彩色标记的散点图
'''
from matplotlib.pyplot as plt
import numpy as np

# generate x va
x = np.random.randn(10000)

# random measurements no correlation
y1 = np.random.randn(len(x))

# stong correlation
y2 = 1.2 + np.exp(x)

ax1 = plt.subplot(121)
plt.scatter(x, y1, color='indigo', alpha=0.3,
            edgecolors='white', label='no correl')
plt.xlabel('no correlation')
plt.grid(True)
plt.legend()

ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
plt.scatter(x, y2, color='green', alpha=0.3, edgecolors='grey', label='correl')
plt.xlabel('strong correlation')
plt.grid(True)
plt.legend()

plt.show()