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
