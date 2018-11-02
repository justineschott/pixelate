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
    
    
## import herrschners yarn images

herrschners_file = glob.glob('C:/Users/Justine/Documents/GitHub/pixelate/fig/herrschners_purchased_11.29.17/*.jpg') #assuming jpg
herrschners_file = np.asarray(herrschners_file)
herrschners_img = []
for filename in herrschners_file: 
    im=load_img(filename)
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
    
## create closest color with this dict
def closest_herrschners_name(requested_rgb):
    min_colours = {}
    for name, array in herrschners_name.items():
        dist = distance(array, requested_rgb)
        min_colours[dist] = name
    return min_colours[min(min_colours.keys())]

## plot all herrchners colors
color_plts = []
for rgb in rgbs:
    color_plts.append(webcolors.rgb_to_rgb_percent(rgb))

color_flts = []
for color_plt in color_plts:
    color_flts.append(tuple(float("." + x.replace(".", "").replace("%", "").zfill(4)) for x in color_plt))
    
    
fig = plt.figure()
ax = fig.add_subplot(111)

ratio = 1.0 / 3.0
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
plt.show()


## pixelate rug image

#load image
rug = load_img('C:/Users/Justine/Documents/GitHub/pixelate/fig/hmb.rolling_stones_tongue.coaster.jpg')

# Figure out the dimensions of each square
# We want:
# 1. Square width and height should be about the same
# 2. No leftover pixels at the edges
# This means that some squares might have one more or one less pixel
# depending on rounding
num_cols = 50
square_w = float(rug.shape[1]) / num_cols
num_rows = int(round(rug.shape[0] / square_w))
square_h = float(rug.shape[0]) / num_rows

# overwrite each square with the average color, one by one
# also create new smaller matrix with the average color
rug_small = np.zeros([50,50,3], dtype='uint8')
for row in range(num_rows):
    for col in range(num_cols):
        rug_small[row,col] = make_one_square(rug, row, col, square_h, square_w)
        

# show the image
plt.axis('on')
plt.imshow(rug)
ax = plt.gca()
ax.set_yticks(np.arange(0,340,340/50), minor=False)
ax.set_xticks(np.arange(0,340,340/50), minor=False)
ax.grid(color='w', linestyle='-', linewidth=.1)
ax.set_yticklabels([])
ax.set_xticklabels([])

plt.show()

# save the image with gridlines
plt.savefig('C:/Users/Justine/Documents/GitHub/pixelate/fig/hmb.rolling_stones_tongue.coaster.gridlines.png')

# save the image pixelated
misc.imsave('C:/Users/Justine/Documents/GitHub/pixelate/fig/hmb.rolling_stones_tongue.coaster.pixelated.jpg', rug)

## show the small image
plt.axis('on')
plt.imshow(rug_small)
ax.set_yticklabels([])
ax.set_xticklabels([])

plt.show()

## create list of unique herrschners colors in image
unique_arrays = unique_rgb(rug_small)

#herrschner color name conversions
unique_herrschners = []
for unique_array in unique_arrays:
    unique_herrschners = np.append(unique_herrschners, closest_herrschners_name(unique_array.tolist()))   
unique_herrschners = np.unique(unique_herrschners)
df = pd.DataFrame(unique_herrschners)    
df.to_csv("C:/Users/Justine/Documents/GitHub/pixelate/data/colors_purchased.csv")

## plot all herrschners colors in image
color_plts = []
for unique_herrschner in unique_herrschners:
    color_plts.append(webcolors.rgb_to_rgb_percent(herrschners_name[unique_herrschner]))

color_flts = []
for color_plt in color_plts:
    color_flts.append(tuple(float("." + x.replace(".", "").replace("%", "").zfill(4)) for x in color_plt))
    
    
fig = plt.figure()
ax = fig.add_subplot(111)

ratio = 1.0 / 3.0
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
    ax.annotate(unique_herrschners[c], xy=pos)
    if y >= y_count-1:
        x += 1
        y = 0
        c += 1
    else:
        y += 1
        c += 1
plt.show()
plt.savefig('C:/Users/Justine/Documents/GitHub/pixelate/fig/string_purchased.png')


## plot all actual colors in image
color_plts = []
for array in unique_arrays:
    color_plts.append(webcolors.rgb_to_rgb_percent(array))
    
color_flts = []
for color_plt in color_plts:
    color_flts.append(tuple(float("." + x.replace(".", "").replace("%", "").zfill(4)) for x in color_plt))
    
fig = plt.figure()
ax = fig.add_subplot(111)

ratio = 1.0 / 3.0
count = math.ceil(math.sqrt(len(color_flts)))
x_count = count * ratio
y_count = count / ratio
x = 0
y = 0
w = 1 / x_count
h = 1 / y_count

for color_flt in color_flts:
    pos = (x / x_count, y / y_count)
    ax.add_patch(patches.Rectangle(pos, w, h, color= color_flt))
    if y >= y_count-1:
        x += 1
        y = 0
    else:
        y += 1

plt.show() 

## create array of herrschners colors in image
#herrschner color name conversions
rug_herrschners = np.empty([rug_small.shape[0],rug_small.shape[1]], dtype="<U32")
for p in range(rug_small.shape[0]):
    array_herrschners = []
    for array in rug_small[p]:
        array_herrschners = np.append(array_herrschners, closest_herrschners_name(array.tolist()))   
    for q in range(array_herrschners.shape[0]):
        rug_herrschners[p,q] = array_herrschners[q]

df = pd.DataFrame(rug_herrschners)
df.to_csv("C:/Users/Justine/Documents/GitHub/pixelate/data/rug_herrschners_purchased.csv", index=False)  

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

plt.savefig('C:/Users/Justine/Documents/GitHub/pixelate/fig/translation_purchased.png')

##count colors
for color in unique_herrschners:
    print(color, ": ", np.argwhere(rug_herrschners == color).shape[0], " strings")

