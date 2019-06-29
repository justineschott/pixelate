##imports
import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image
import scipy.misc
import webcolors
import pandas as pd
import matplotlib.patches as patches
import matplotlib.colors as colors
import math
import glob
import re
import csv

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
github_filepath = os.environ.get("github_filepath")

# 3 strings inputted
colors = sys.argv[1:3]


herrschners_name = {}
with open(github_filepath+'/pixelate/data/herrschners_name.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    herrschners_name = dict(reader)
    
for k, v in herrschners_name.items():
    herrschners_name[k] = [int(i) for i in re.sub(r' +', ' ', v).replace('[', '').replace(']', '').lstrip().split(' ')]

herrschners_name_limited = { key: herrschners_name[key] for key in colors }

# create closest color with this dict
def closest_herrschners_name_limited(requested_rgb):
    min_colours = {}
    for name, array in herrschners_name_limited.items():
        dist = distance(array, requested_rgb)
        min_colours[dist] = name
    return min_colours[min(min_colours.keys())]


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
        
    return (av_r, av_g, av_b)

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
    

    
## import rgb rug and most common strings file
rug_small = np.load(github_filepath+'/pixelate/data/'+rug_name+'/rug_rgb_'+rug_name+'.npy')

## create array of herrschners colors in image
rug_herrschners_limited = np.empty([rug_small.shape[0],rug_small.shape[1]], dtype="<U32")
for p in range(rug_small.shape[0]):
    array_herrschners = []
    for array in rug_small[p]:
        array_herrschners = np.append(array_herrschners, closest_herrschners_name_limited(array.tolist()))   
    for q in range(array_herrschners.shape[0]):
        rug_herrschners_limited[p,q] = array_herrschners[q]

df = pd.DataFrame(rug_herrschners_limited)
df.to_csv(github_filepath+'/pixelate/data/'+rug_name+'/rug_herrschners_limited.csv', index=False, columns=None)  

## create array of herrschners RBGs in image
#herrschner color name conversions
rug_herrschners_rgb_limited = np.empty([rug_herrschners_limited.shape[0],rug_herrschners_limited.shape[1], 3], dtype="<U10")
for x in range(0, rug_herrschners_rgb_limited.shape[0]):
    for y in range(0, rug_herrschners_rgb_limited.shape[1]):
        rug_herrschners_rgb_limited[x,y] = herrschners_name_limited[rug_herrschners_limited[x,y]]

rug_herrschners_rgb_limited = rug_herrschners_rgb_limited.astype(np.uint8)


## plot image in herrschners colors 
plt.axis('on')
plt.imshow(rug_herrschners_rgb_limited)
ax = plt.gca()
ax.set_yticks(np.arange(0,num_rows), minor=False)
ax.set_xticks(np.arange(0,num_cols), minor=False)
ax.grid(color='w', linestyle='-', linewidth=.1)
ax.set_yticklabels([])
ax.set_xticklabels([])

#plt.show()
plt.savefig(github_filepath+'/pixelate/fig/'+rug_name+'/translation_limited.jpg')

##count colors
buy_color_limited = []
buy_count_limited = []

for color in unique_herrschners:
    buy_color_limited.append(color)
    buy_count_limited.append(np.argwhere(rug_herrschners_limited == color).shape[0])

df2 = pd.DataFrame({'color': buy_color_limited, 'strings': buy_count_limited})

df2['packages'] = df2['strings']/320

df2.to_csv(github_filepath+'/pixelate/data/'+rug_name+'/buy_strings_limited.csv', index=False)  