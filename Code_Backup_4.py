'''
设置坐标轴标签 透明度大小
'''
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np

data = np.random.randn(70)

fontsize = 18
plt.plot(data)
title = 'This is figure title'
x_label = 'This is x axis label'
y_label = 'This is y axis label'

title_text_obj = plt.title(
    title, fontsize=fontsize, verticalalignment='bottom')

title_text_obj.set_path_effects([patheffects.withSimplePatchShadow()])

# offset_xy -- set the 'angele' of the shadow
# shadow_rgbFace -- set the color of the shadow
# patch_alpha -- setup the transparency of shadow

offset_xy = (1, -1)
rgbFace = (1.0, 0.0, 0.0) # Red
alpha = 0.8

# customize shadow properties
pe = patheffects.withSimplePatchShadow(
    offset=offset_xy, shadow_rgbFace=rgbFace, alpha=alpha)

# apply them to the xaxis and yaxis labels
xlabel_obj = plt.xlabel(x_label, fontsize=fontsize, alpha=0.5)
xlabel_obj.set_path_effects([pe])

ylabel_obj = plt.ylabel(y_label, fontsize=fontsize, alpha=0.5)
ylabel_obj.set_path_effects([pe])

plt.show()

'''
设置坐标轴标签 の 透明度
'''
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np

data = np.random.randn(70)

fontsize = 18
plt.plot(data)

title = 'This is figure title'
x_label = 'This is x axis label'
y_label = 'This is y axis label'

title_text_obj = plt.title(title, fontsize=fontsize, verticalalignment='bottom')
title_text_obj.set_path_effects([patheffects, withSimplePatchShadow()])

# offset_xy -- set the 'angle' of the shadow
# shadow_rgbFace -- set the color of the shadow
# patch_alpha -- setup the transparency of th shadow

offset_xy = (1, -1)
rgbFace = (1.0, 0.0, 0.0)
alpha = 0.8

# customzie shadow properties
pe = patheffects.withSimplePatchShadow(offset_xy=offset_xy, shadow_rgbFace=rgbFace, patch_alpha=alpha)

# apply them to the x, y axis labels
xlabel_obj = plt.xlabel(x_label, fontsize=fontsize, alpha=0.5)
xlabel_obj.set_path_effects([pe])

ylabel_obj = plt.ylabel(y_label, fontsize=fontsize, alpha=0.5)
ylabel_obj.set_path_effects([pe])

plt.show()

'''
为图表添加 阴影
'''
# 这里利用到 matplotlib 中 transformation Frame
# transformation know howto 将给定的坐标系与显示坐标系之间切换
# This Frame can 将现有对象 转化成 一个 偏移对象

import matplotlib.transforms as transforms

def setup(layout=None):
	assert layout is not None
	fig = plt.figure()
	ax = fig.add_subplot(layout)
	return fig, ax

def get_signal():
	t = np.arange(0., 2.5, 0.01)
	s = np.sin(5 * np.pi * t)
	return t, s

def plot_signal(t, s):
	line, = axes.plot(t, s, linewidth=5, color='magenta')
	return line,

def make_shadow(fig, axes, line, t, s):
	delta = 2/72. # how many points to move the shadow
	offset = transforms.ScaledTranslation(delta, -delta, fig.dpi_scale_trans)
	offset_transform = axes.transData + offset

	# plot the same data, but now, using, offset, transform
	# zorder -- to render it below the line
	axes.plot(t, s, linewidth=5, color='gray', transform=offset_transform, zorder=0.5*line.get_zorder())

if __name__ == '__main__':
	fig, axes = setup(111)
	t, s = get_signal()
	line, = plot_signal(t, s)
	make_shadow(fig, axes, line, t, s)
	axes.set_title('Shadow effect using an offset transform')
	plt.show()

'''
向图表添加数据表
'''

plt.figure()
ax = plt.gca()
y = np.random.randn(9)
col_labels = ['col1', 'col2', 'col3']
row_labels = ['row1', 'row2', 'row3']
table_vals = [[11, 12, 13], [21, 22, 23], [28, 29, 30]]
row_colors = ['red', 'gold', 'green']

my_tables = plt.table(cellText=table_vals,
					colWidths=[0.1] * 3,
					rowLabels=row_labels,
					colLabels=col_labels,
					rowColours=row_colors,
					loc='upper right')
plt.plot(y)
plt.show()

table(cellText=None, cellColours=None,
	cellLoc='right', colWidths=None,
	rowLabels=None, rowColours=None, rowLoc='left',
	colLabels=None, colColours=None, colLoc='center',
	loc='bottom', bbox=None)

'''
使用 subplots
'''

# subplots' BaseClass is matplotlib.axes.SubplotBase
# subplots is matplotlib.axes.Axes' Instance
# matplotlib.pyplot.subplots 用以方便 穿就爱你 普通 布局 子区 可指定 大小
# AND create share x, y labels' subplots 可使用 sharex, sharey kwargs to finish
# plot.subplot() -- based on 1
# plt.subplot2grid() -- based on 0
import matplotlib.pyplot as plt
plt.figure(0)

axes1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
axes2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
axes3 = plt.subplot2grid((3, 3), (1, 2))
axes4 = plt.subplot2grid((3, 3), (2, 0))
axes5 = plt.subplot2grid((3, 3), (2, 1), colspan=2)

# tidy up tick labels size
all_axes = plt.gcf().axes
for ax in all_axes:
	for ticklabel in ax.get_xticklabels() + ax.get_yticklabels():
		ticklabel.set_fontsize(10)

plt.suptitle('Demo of subplot2grid')
plt.show()

'''
定制化 网格
'''
plt.plot([1,2,3,3.5,4,5,4,3.5,3,2,1])
plt.grid()
# control 主刻度 or 次刻度
# args which 'major' 'minor' 'both', axis 'x', 'y' 'both'
# others args by kwargs to get syb a matplotlib.lines.Line2D Instance
ax.grid(color='g', linestyle='--', linewidth=1)

'''
定制化 matplotlib AND mpl_toolkits
简单且可管理的方式 create axes 网格 の  AxesGrid module
'''

from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.cbook import get_sample_data

def get_demo_image():
	f = get_sample_data('axes_grid/bivariate_normal.npy', asfileobj=False)
	# z is a numpy array of 15X15
	Z = np.load(f)
	return Z, (-3, 4, -4, 3)

def get_grid(fig=None, layout=None, nrows_ncols=None):
	assert fig is not None
	assert layout is not None
	assert nrows_ncols is not None

	grid = ImageGrid(fig, layout, nrows_ncols=nrows_ncols, axes_pad=0.05, add_all=True, label_mode='L')
	return grid

def load_images_to_grid(grid, Z, *images):
	min, max = Z.min(), Z.max()
	for i, image in enumerate(images):
		axes = grid[i]
		axes.imshow(image, origin='lower', vmin=min, vmax=max, interpolation='nearest')

if __name__ == '__main__':
	fig = plt.figure(1, (8, 5))
	grid = get_grid(fig, 111, (1, 3))
	Z, extent = get_demo_image()

	# slice iamge
	image1 = Z
	image2 = Z[:, :10]
	image3 = Z[:, 10:]

	load_images_to_grid(grid, Z, image1, image2, image3)

	plt.draw()
	plt.show()