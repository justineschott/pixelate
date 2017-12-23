#!/usr/bin/env python

# This script will pixelate most jpg and png images
# It will both show you the result and save it

import sys

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import scipy.misc as misc

import webcolors
import pandas as pd
import matplotlib.patches as patches
import matplotlib.colors as colors
import math

from PIL import Image
import glob
import re


def load_img(filename):
    # boilerplate code to open an image and make it editable
    img = Image.open(filename)
    data = np.array(img)
    return data

def all_square_pixels(row, col, square_h, square_w):
    # Every pixel for a single "square" (superpixel)
    # Note that different squares might have different dimensions in order to
    # not have extra pixels at the edge not in a square. Hence: int(round())
    for y in range(int(round(row*square_h)), int(round((row+1)*square_h))):
        for x in range(int(round(col*square_w)), int(round((col+1)*square_w))):
            yield y, x

def make_one_square(img, row, col, square_h, square_w):
    # Sets all the pixels in img for the square given by (row, col) to that
    # square's average color
    pixels = []

    # get all pixels
    for y, x in all_square_pixels(row, col, square_h, square_w):
        pixels.append(img[y][x])

    # get the average color
    av_r = 0
    av_g = 0
    av_b = 0
    for r, g, b in pixels:
        av_r += r
        av_g += g
        av_b += b
    av_r /= len(pixels)
    av_g /= len(pixels)
    av_b /= len(pixels)

    # set all pixels to that average color
    for y, x in all_square_pixels(row, col, square_h, square_w):
        img[y][x] = (av_r, av_g, av_b)
        

def unique_rgb(image):
    #reshape to list of rgbs
    shaped = image[0]
    for i in range(1,image.shape[0]):
        shaped = np.concatenate((shaped, image[i]), axis=0)
        
    # unique RGBs
    unique = np.unique(shaped,axis=0)
    return unique

def distance(c1, c2):
    r1 = c1[0]
    b1 = c1[1]
    g1 = c1[2]
    r2 = c2[0]
    b2 = c2[1]
    g2 = c2[2]
    return math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
    
##import csv of file
df = pd.read_csv("C:/Users/Justine/Documents/GitHub/pixelate/data/rug_herrschners_purchased_manual.csv")
rug_herrschners = df.as_matrix()

## create list of unique herrschners colors in image  
unique_herrschners = np.unique(rug_herrschners)
df = pd.DataFrame(unique_herrschners)    
df.to_csv("C:/Users/Justine/Documents/GitHub/pixelate/data/colors_purchased_manual.csv")

##count colors
for color in unique_herrschners:
    print(color, ": ", np.argwhere(rug_herrschners == color).shape[0], " strings")

## create array of herrschners RBGs in image
#herrschner color name conversions
rug_herrschners_rgb = np.empty([rug_herrschners.shape[0],rug_herrschners.shape[1], 3], dtype="<U10")
for x in range(0, rug_herrschners_rgb.shape[0]):
    for y in range(0, rug_herrschners_rgb.shape[1]):
        rug_herrschners_rgb[x,y] = herrschners_name[rug_herrschners[x,y]]

rug_herrschners_rgb = rug_herrschners_rgb.astype(np.uint8)

## plot image in herrschners colors 
plt.axis('on')
plt.imshow(rug_herrschners_rgb)
ax = plt.gca()
ax.set_yticks(np.arange(0,50), minor=False)
ax.set_xticks(np.arange(0,50), minor=False)
ax.grid(color='w', linestyle='-', linewidth=.1)
ax.set_yticklabels([])
ax.set_xticklabels([])

plt.show()

plt.savefig('C:/Users/Justine/Documents/GitHub/pixelate/fig/translation_purchased_manual.png')