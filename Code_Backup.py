'''
2.6.2
读取并解析 Github 网站最近活动时间表
Hehe We have shut down API v1 and v2 as promised
https://github.com/blog/1160-github-api-v2-end-of-life
'''

import requests

# 指定 Github URL 来读取 JSON 格式数据
url = 'https://github.com/timeline.json'
# use requests module 访问指定的 URL 同时读取内容
r = requests.get(url)
# 读取内容并将之转换为 JSON 格式的对象
json_obj = r.json()

# 迭代访问 JSON 对象 对于其中每一项读取每个代码库的 URL 值
repos = set()
for entry in json_obj:
    try:
        repos.add(entry['repository']['url'])
    except KeyError as e:
        print('No key %s. Skipping...' % e)

from pprint import pprint
pprint(repos)

'''
2.7
导出数据到 JSON CSV Exce
'''

# 导入所需模块
import os
import sys
import argparse
try:
    import cStringIO as StringIO
except:
    import StringIO
import strcut
import json
import csv

# 定义合适的读写数据方法


def import_data(import_file):
    '''
    Imports data form import_file
    Expects to find fixed width row
    Sample row 16111222333 3333324 323
    '''
    mask = '9s14s5s'
    data = []
    with open(import_file, 'r') as f:
        for line in f:
            # unpack line to tuple
            fields = strcut.Struct(mask).unpack_from(line)
            # strip any whitespave for each field
            # pack everythinf in a list and add into full dataset
            data.append(list([f.strip(for f in fields)]))
    return data


def write_data(data, export_format):
    '''
    Dispatches call to a specific transformer and returns dta set
    Exception is xlsx where we have to save data in file
    '''
    if export_format == 'csv':
        return write_csv(data)
    elif export_format == 'json':
        return write_json(data)
    elif export_format == 'xlsx':
        return write_xlsx(data)
    else:
        raise Exception('Illegal format defined')

# 为每种数据格式分别制定各自实现方法


def write_csv(data):
    '''
    Transforms data into CSV
    Return CSV as string
    '''
    # Using this simulate file IO
    # as CSV can only write to files
    f = StringIO.StringIO()
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)
    # Get the content of the file-like object
    return f.getvalue()


def write_json(data):
    '''
    Transforms data into json
    Very straightforward
    '''
    j = json.dumps(data)
    return j


def write_xlsx(data):
    '''
    Writes data into xlsx file
    '''
    from xlwt import Workbook
    book = Workbook()
    sheet1 = book.add_sheet('Sheet 1')
    row = 0
    for line in data:
        col = 0
        for datum in line:
            print(datum)
            sheet1.write(row, col, datum)
            col += 1
        row += 1
        # We have hard limit here of 65535 rows
        # That we are able to save in spreadsheet
        if row > 65535:
            print(
                'Hit limit of # of rows in one sheet (65535)', file=sys.stderr)
            break
    # xlsx is special case where we have to save the file and just return 0
    f = StringIO.StringIO()
    book.save(f)
    return f.getvalue()

# 完成main入口点代码 解析命令行参数中传入的文件路径 导入数据并导出成要求的格式
if __name__ == '__main__':
    # parse input argument
    parser = argparse.ArgumentParser()
    parser.add_argumnet('import_file', help='Path to a fixed-width data files')
    parser.add_argumnet('export_format', help='Export fomat: json,csv,xlsx')
    args = parser.parse_args()

    if args.import_file is None:
        print('You must specify path to import from', file=sys.stderr)
        sys.exit(1)
    if args.export_format not in ('csv', 'json', 'xlsx'):
        print('You must provide vaild export file format')
        sys.exit(1)

    # verify given path is accesible file
    if not os.path.idfile(args.import_file):
        print('Given path is not a file:%s' %
              args.import_file, file=sys.stderr)

    # read from formated fixed-width file
    data = import_data(args.import_file)

    # export data to specified format to make this Unix-lixe pipe-able
    # we just print to stdout
    print(write_data(data, args.export_format))

'''
2.9.2.1
用 MAD 来检测数据中的异常值 Outlier
'''

import numpy as np
import matplotlib.pyplot as plt


def is_outlier(points, threshold=3.5):
    '''
    Return a boolean array with True id points asr outliers and False otherwise

    Data points with a modified z-score greater than this
    # value will be calssified as outlier
    '''
    # transform into vector
    if len(points.shape) == 1:
        points = points[:, None]
    # compute median value
    median = np.median(points, axis=0)
    # compute diff sums along the axis
    diff = np.sum((points - median) ** 2, axis=1)
    diff = np.sqrt(diff)
    # compute MAD
    med_abs_deviation = np.median(diff)

    # compute modified Z-score
    # http://www.itl.nist.gov/dic898/handbook/eda/section4/eda43.htm#
    # Inlewicz
    modified_z_score = 0.6745 * diff / med_abs_deviation

    # return a mask for each outlier
    return modified_z_score > threshold

# Random data
x = np.random.random(100)
# histogram buckets
buckets = 50

# Add in a few outliers
x = np.r_[x, -49, 95, 100, -100]

# Keep valid data points
# Note here that
# '~' is logical NOT on boolean numpy arrays
filtered = x[~is_outlier(x)]

# plot histograms
plt.figure()

plt.subplot(211)
plt.hist(x, buckets)
plt.xlabel('Raw')

plt.subplot(212)
plt.hist(filtered, buckets)
plt.xlabel('Cleaned')

plt.show()

'''
2.9.2.2
用人眼检查数据
[1] 创建散点图 即轻易看到偏离簇中心的值
[2] 绘制箱线图 将显示出中值 上四分位数 下四分位数 and 远离箱体的outlier
'''

from pylab import *

# fake up some data
spread = rand(50) * 100
center = ones(50) * 50

# generate some outliers high and low
flier_high = rand(10) * 100 + 100
flier_low = rand(10) - 100

# merge generated data set
data = concatenate((spread, center, flier_high, flier_low), 0)

subplot(311)
# basic plot
# 'gx' defining the outlier plotting properties
boxplot(data, 0, 'gx')
# compare this with similar scatter plot
subplot(312)
spread_1 = concatenate((spread, flier_high, flier_low), 0)
center_1 = ones(70) * 25
scatter(center_1, spread_1)
xlim([0, 50])

# and with another that is more appropriate fo scatter plot
subplot(313)
center_2 = rand(70) * 50
scatter(center_2, spread_1)
xlim([0, 50])
show()

'''
2.9.2.3
相同数据在不同情况下显示起来截然不同
'''

# generate uniform data points
x = 1e6 * rand(1000)
y = rand(1000)

figure()

subplot(211)
# make scatter plot
scatter(x, y)
# limit x axis
xlim(1e-6, 1e6)

subplot(212)
scatter(x, y)
# BUT make x axis logarithmic
xscale('log')
xlim(1e-6, 1e6)

show()
'''
2.12.2
Lena 图 数字信号处理
'''
import scipy.misc
import matplotlib.plot as plt

# load already prepared ndarray from scipy
'''
lena = scipy.misc.lean()
'''
# lena 被 face OR ascent 替代
# load already prepared ndarray from scipy
subplot(211)
ascent = scipy.misc.ascent()
plt.gray()
plt.imshow(ascent)
plt.colorbar()
plt.show()
subplot(212)
face = scipy.misc.face()
# set the default colormap to gray
plt.gray()
plt.imshow(face)
plt.colorbar()
plt.show()

# 更一步进行检查这个对象
print(face.shape, '\n', face.max(), '\n', face.dtype)
print(ascent.shape, '\n', ascent.max(), '\n', ascent.dtype)
# 利用 Python Image Library Read Image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

bug = Image.open('head_image_zhihu.jpg')
arr = np.array(bug.getdata(), numpy.uint8).reshape(bug.size[1], bug.size[0], 3)
plt.gray()
plt.imshow(arr)
plt.colorbar()
plt.show()

'''
2.12.3
利用 Python 操作并处理图像
加载一幅 RGB 通道的真实图像 将之转换成单通道的 ndarray 然后利用数组切片的方法来放大大部分图像
'''

import matplotlib.pyplot as plt
import scipy
import numpy

bug = scipy.misc.imread('head_image_zhihu.jpg')

# if want to inspect the shape of the loaded image
# uncomment following line
# print bug.shape

# the original image is RGB having values for all three
# channels separately need to convert that to freyscale image
# by picking up just one channel

# covert to gray
bug = bug[:, :, 0]
# show original image
plt.figure()
plt.gray()

plt.subplot(211)
plt.imshow(bug)

# show 'zoomed' region
zbug = bug[250:750, 250:750]
plt.subplot(212)
plt.imshow(zbug)

plt.show()

'''
2.13.2
利用 Python 的 random 模块生成一个简单的随机数样本
'''

import pylab
import random

SAMPLE_SIZE = 100

# seed random generator
# if no argument provided
# uses system current time
random.seed()

# store generator random values here
real_rand_vars = []

# pick some random values
real_rand_vars = [random.random() for val in range(SAMPLE_SIZE)]
# create histogram form data in 10 buckets
pylab.hist(real_rand_vars, 10)
# define x and labels
pylab.xlabel('Number range')
pylab.ylabel('Count')

# show figure
pylab.show()

'''
用相似的方法 生产虚拟价格增长数据的时序图 and 加上一些随机噪声
'''
# days to generate data for
duration = 100
# mean value
mean_inc = 0.2

# standard deviation
std_dev_inc = 1.2

# time series
x = range(duration)
y = []
price_today = 0

for i in x:
    next_delta = random.normalvariate(mean_inc, std_dev_inc)
    price_today += next_delta
    y.append(price_today)

pylab.plot(x, y)
pylab.xlabel('Time')
pylab.ylabel('Value')
pylab.show()

'''
为了更多控制 添加各种不同的分布
'''

import random
import matplotlib
import matplotlib.pyplot as plt

SAMPLE_SIZE = 1000
#  histogram buckets
buckets = 100

plt.figure()

# need to update font size just for this example
# matplotlib.rcParams.update({'font.size':7})

# Figure 1 在 [0,1] 之间分布的随机变量
plt.subplot(521)
plt.xlabel('random.random')
# Return the next random floating point number in the range [0.0,1.0]
res = [random.random() for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

# Figure 2 均匀分布的随机变量
plt.subplot(522)
plt.xlabel('random.uniform')
# Return a random floating point number N such that a <= N <= b for a <=b and b <= N for b < a
# The end_point value b may or may not included in the range depending on
# floating_point rounding in the equation a + (b - a) * random()
a = 1
b = SAMPLE_SIZE
res = [random.uniform(a, b) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

# Figure 3 三角形分布
plt.subplot(523)
plt.xlabel('random.triangular')
# Return a random floating point number N such that low <= N <= high and with the specified
# mode between those bounds The los and high bounds default to zero and one The mode
# argument defaults to the midpoint between the bounds giving a symmentric
# distribution
low = 1
high = SAMPLE_SIZE
res = [random.triangular(low, high) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

# Figure 4 beta 分布
plt.subplot(524)
plt.xlabel('random.betavariate')
alpha = 1
beta = 10
res = [random.betavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

# Figure 5 指数分布
plt.subplot(525)
plt.xlabel('random.expovariate')
lambd = 1.0 / ((SAMPLE_SIZE + 1) / 2.)
res = [random.expovariate(lambd) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

# Figure 6 gamma 分布
plt.subplot(526)
plt.xlabel('random.gammavariate')
alpha = 1
beta = 10
res = [random.gammavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

# Figure 7 对数正态分布
plt.subplot(527)
plt.xlabel('random.lognormvariate')
mu = 1
sigma = 0.5
res = [random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

# Figure 8 正态分布
plt.subplot(528)
plt.xlabel('random.normalvariate')
mu = 1
sigma = 0.5
res = [random.normalvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

# Figure 9 帕累托分布
plt.subplot(529)
plt.xlabel('random.paretovariate')
alpha = 1
res = [random.paretovariate(alpha) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

plt.tight_layout()
plt.show()

'''
2.14.2
真实数据的噪声平滑处理
'''

from pylab import *
from numpy import *


def moving_average(interval, window_size):
    '''
    Compute convoluted window for given size
    '''
    window = ones(int(window_size)) / float(window_size)
    return convolve(interval, window, 'same')

t = linspace(-4, 4, 100)
y = sin(t) + randn(len(t)) * 0.1

plot(t, y, 'k.')

# compute moving average
y_av = moving_average(y, 10)
plot(t, y_av, 'r')
# xlim(0,100)
xlabel('Time')
ylabel('Value')
grid(True)
show()

'''
基于信号(数据点)窗口的卷积(函数的总和)
让窗口平滑处理达到更好的效果
'''

import numpy
from numpy import *
from pylab import *

# possiable window type

WINDOWS = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
# want to see just two window type comment previoys line
# and uncomment the following one
# WINDOWS = ['flat','hanning']


def smooth(x, window_len=11, window='hanning'):
    '''
    Smooth the data using a window with requested size
    Return smothed signal

    x -- input signal
    window_len -- length of smoothing window
    window -- type of window: 'flat','hanning','hamming','bartlett','blackman'
    flat window will produce a moving average smoothing
    '''
    if x.ndim != 1:
        raise ValueError('Smooth only accepts 1 dimension arrays')
    if x.size < window_len:
        raise ValueError('Input vector needs to be bigger than window size')
    if window_len < 3:
        return x
    if not window in WINDOWS:
        raise ValueError(
            'Window is one of [flat][hanning][hamming][bartlett][blackman]')
    # adding reflected window in front and at the end
    s = numpy.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    # pick windows type and da averaging
    if window == 'flat':    # moving average
        w = numpy.ones(window_len, 'd')
    else:
        # call appropriate func in numpy
        w = eval('numpy.' + window + '(window_len)')
        # NOTE: length(output) != length(input), to correct this:
        # return y[(window_len/2-1):-(window_len/2)] instead of just y.
        y = numpy.convolve(w/w.sum(), s, mode='vaild')
        return y

# Get some evently spaced numbers over a specified interval
t = linspace(-4, 4, 100)
# Make some noisy sinusoidal
x = sin(t)
xn = x + randn(len(t))*0.1

# Smooth it
y = smooth(x)

# windows
ws = 31

subplot(211)
plot(ones(ws))

# draws on the same axes
hold(True)

# plot for every windows
for w in WINDOWS[1:]:
    eval('plot('+w+'(ws))')
# configure axis properties
axis([0, 30, 0, 1.1])

# add legend for every window
legend(WINDOWS)

title('Smoothing window')

# add second plot
subplot(212)

# draw original signal
plot(x)

# and signal with added noise
plot(xn)

# smooth signal with noise for every possiable windowing algorithm
for w in WINDOWS:
    plot(smooth(xn, 10, w))
# add legend for every graph
l = ['original signal', 'signal with noise']
l.extend(WINDOWS)
legend(l)

title('Smoothed signal')

show()

'''
利用 中值滤波 算法 进行信号滤波
'''

import numpy as np
import pylab as p
import scipy.signal as signal

# get some linear data
x = np.linspace(0, 1, 101)

# add some noisy signal
x[3::10] = 1.5

p.plot(x)
p.plot(signal.medfilt(x, 3))
p.plot(signal.medfilt(x, 5))

p.legend(['original signal', 'length 3', 'length 5'])
p.show()

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

show()
