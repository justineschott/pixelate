This page is a work in progress.

# Custom Hook Rug Grids

This GitHub repository contains project materials for a personal project which makes images into grids for creating latch hook rugs.

## About

Latch hooking is a craft that creates a small shag rug to display or make into a throw pillow. Latch hook kits are available to buy online, but the design can't be customized.

The Custom Hook Rug Grid tool creates a latch hook kit out of any JPG or PNG that you choose.

## Project Directory

Here is the structure for the repo. 

```
├── pixelate
|   ├── data
|   |   ├──rug_name
|   |   |   |──buy_strings.csv
|   |   |   |──rug_herrschners.xlsx
|   |   ├──herrschners_name.csv
|   ├── fig
|   |   ├──all_colors.jpg
|   |   ├──herrschners
|   |   |   |──130001P_antique-brass.jpg
|   |   |   |──[remainder of herrschner's colors from website].jpg
|   |   ├──rug_name
|   |   |   |──rug_name.jpg
|   |   |   |──rug_name_gridlines.png
|   |   |   |──rug_name_pixelated.jpg
|   |   |   |──string.jpg
|   |   |   |──translation.jpg
|   ├── src
|   |   ├──01_herrschners.py
|   |   ├──02_pixelate.py
|   |   ├──03_limit_colors.py
|   |   ├──classes.py
└──
```

## How to run this code

1. Add an environemnt variable, `GITHUB_FILEPATH`, which is the filepath to the folder which contains the `pixelate` folder 
2. Create folder named the same name as your image within the `data` folder
3. Create folder named the same name as your image within the `fig` folder and save the JPG or PNG file in that folder folder
4. If `herrschners.csv` is not already saved in the `data` folder, run the Herrschners translation file: `pixelate/src/01_herrschners.py` 
5. Run the pixelation file with the proper arguments: : `pixelate/src/02_pixelate.py`. Arguments are: `rug_name jpg/png num_columns`

Note: `03_limit_colors.py` is still a work in progress.