##imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.misc as misc
import webcolors
import pandas as pd
import matplotlib.patches as patches
import matplotlib.colors as colors
import math
import glob
import re
import csv

import os
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())
GITHUB_FILEPATH = os.environ.get("GITHUB_FILEPATH")

#import classes
os.chdir(GITHUB_FILEPATH+'/pixelate/src') 
from classes import all_square_pixels, make_one_square, unique_rgb, distance

    
## import herrschners yarn images

herrschners_file = glob.glob(GITHUB_FILEPATH+'/pixelate/fig/herrschners/*.jpg') #assuming jpg
herrschners_file = np.asarray(herrschners_file)
herrschners_img = []
for filename in herrschners_file: 
    im=np.array(Image.open(filename))
    herrschners_img.append(im)

## list of herrschners rgbs
rgbs = []
for img in herrschners_img:
    # overwrite each square with the average color, one by one
    empty_var = make_one_square(img, 0, 0, float(img.shape[0]), float(img.shape[1]))
    rgbs.append(unique_rgb(img)[0])

## list of herrschners names
names = []
for filename in herrschners_file:
    names.append(re.search('130001P_(.+?).jpg', filename).group(1))

##dictionary of herrschners rgbs and names
names_and_rgbs= zip(rgbs, names)
herrschners_name = {}
for rgb, name in names_and_rgbs:
    herrschners_name[name] = rgb

#write dict to import later
with open(GITHUB_FILEPATH+'/pixelate/data/herrschners_name.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in herrschners_name.items():
       writer.writerow([key, value])


## plot all herrchners colors
color_plts = []
for rgb in rgbs:
    color_plts.append(webcolors.rgb_to_rgb_percent(rgb))

color_flts = []
for color_plt in color_plts:
    color_flts.append(tuple(float("." + x.replace(".", "").replace("%", "").zfill(4)) for x in color_plt))
        
fig = plt.figure()
ax = fig.add_subplot(111)

ratio = float(1.0 / 3.0)
count = math.ceil(math.sqrt(len(color_flts)))
x_count = count * ratio
y_count = count / ratio
x = 0
y = 0
w = 1 / x_count
h = 1 / y_count

c=0
for color_flt in color_flts:
    pos = (x / x_count, y / y_count)
    ax.add_patch(patches.Rectangle(pos, w, h, color= color_flt))
    ax.annotate(names[c], xy=pos)
    if y >= y_count-1:
        x += 1
        y = 0
        c += 1
    else:
        y += 1
        c += 1
plt.savefig(GITHUB_FILEPATH+'/pixelate/fig/all_colors.png')