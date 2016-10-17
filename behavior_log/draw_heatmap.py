#!/user/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

import numpy as np
import json
from PIL import Image
from PIL import ImageDraw
from scipy.signal import convolve2d
from matplotlib.colors import hsv_to_rgb

width = 50
sigma = 1000.0
kernel = np.zeros((2*width+1, 2*width+1))
for x in range(2*width+1):
    for y in range(2*width+1):
        diff = (x-width)**2 + (y-width)**2
        kernel[x, y] = np.exp(-diff/sigma)
'''
l_map = (kernel * 255).astype('uint8')
l_img = Image.fromarray(l_map, mode='L')
l_img.show()
'''


def draw_bounding_box(page, image_file):
    img = Image.open(image_file)

    draw = ImageDraw.Draw(img)
    visible_elements = json.loads(page.visible_elements)
    for e in visible_elements:
        draw.rectangle([e['left'], e['top'], e['right'], e['bottom']], outline='Blue')

    img.save('.'.join(image_file.split('.')[:-1]) + '.box.png')


def compute_heat_map(page, image_file):
    fixations = []
    for vp in page.viewports:
        fixations += vp.fixations

    img = Image.open(image_file)

    print img.size

    heat = np.zeros(img.size)
    for f in fixations:
        x0 = f.x_on_page - width
        y0 = f.y_on_page - width
        for x in range(0, 2*width+1):
            for y in range(0, 2*width+1):
                if 0 <= x0+x < img.size[0] and 0 <= y0+y < img.size[1]:
                    heat[x0+x, y0+y] += kernel[x, y] * f.duration
    heat /= np.max(heat)
    heat = heat.transpose()

    shape = heat.shape
    hsv = np.ndarray(list(shape)+[3])
    hsv[..., 0] = 0.667 - heat * 0.667
    hsv[..., 1] = np.tile(1.0, shape)
    hsv[..., 2] = np.tile(1.0, shape)

    rgb = hsv_to_rgb(hsv)
    rgba = np.ndarray(list(shape)+[4], dtype='uint8')
    rgba[..., 0:3] = (rgb * 255).astype('uint8')
    rgba[..., 3] = (heat * 255).astype('uint8')

    heat_img = Image.fromarray(rgba, mode='RGBA')
    #heat_img.show()

    bg = Image.new('RGBA', img.size, color=0)
    bg.paste(img, box=(0, 0))
    bg.paste(heat_img, box=(0, 0), mask=heat_img)
    bg.save('.'.join(image_file.split('.')[:-1]) + '.heat.png')









