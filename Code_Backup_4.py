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
