import numpy as np
import pandas as pd
from math import inf
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.colors import LinearSegmentedColormap
import sqlite3,pickle,zstd

rawMaps = pd.read_sql('SELECT * FROM RawMaps', sqlite3.connect('rawMaps.db'))
rawMaps['map'] = rawMaps['map'].apply(lambda x: pickle.loads(zstd.decompress(x)))
MAPS = {}
for m in rawMaps['name']:
    MAPS[m] = rawMaps[rawMaps['name']==m]['map'].iloc[0]

def get_map_img(map_data:list, size:tuple, crop=False, padding=30):
    '''
    Returns the conversion of the OccupancyGrid data field *map_data* 
    into the corresponding matrix of size *size*

    Parameters
    ----------
    map_data: list[int]
        The data field of nav_msgs/OccupancyGrid.msg
    size: tuple[int, int]
        Tuple containing height and width of the map
    cut: bool
        If True the map will be cropped removing the unscanned (-1) cells
        in the map outline
    padding: int
        The padding added starting from the cells different from -1
    
    Returns
    -------
    occ_matrix: list[list[int]]
        The matrix corresponding to the data field of the OccupancyGrid
    '''
    height, width = size
    img = np.reshape(map_data,[height,width])
    if crop:
        min_x = width
        max_x = 0
        min_y = height
        max_y = 0
        for i in range(height):
            for j in range(width):
                if img[i][j]!=-1 and j<min_x:
                    min_x = j
                if img[i][width-j-1]!=-1 and width-j-1>max_x:
                    max_x = width-j-1
        for j in range(width):
            for i in range(height):
                if img[i][j]!=-1 and i<min_y:
                    min_y = i
                if img[height-i-1][j]!=-1 and height-i-1>max_y:
                    max_y = height-i-1
        if min_y<padding:
            min_y=padding
        if min_x<padding:
            min_x=padding
        cut_img = img[min_y-padding:max_y+padding,min_x-padding:max_x+padding]
    else:
        cut_img = img
    return cut_img

def plot_map_img(img:list, ax:Axes, bg_color="#607372") -> None:
    '''
    Plots the map *img* into the matplotlib ax *ax*

    Parameters
    ----------
    img: list[list[int]]
        Matrix of int representing the data field of the OccupancyGrid
    ax: Axes
        matplotlib ax in which the map will be plotted
    bg_color: str
        Color of the unscanned cells (-1) of the map; the format is the
        same of matplotlib. The default color is the color used by RViz
    '''
    cmap = LinearSegmentedColormap.from_list('my_gradient', (
        (0.000, (0.745, 0.757, 0.741)),
        (0.330, (0.745, 0.757, 0.741)),
        (0.560, (0.580, 0.580, 0.580)),
        (0.640, (0.000, 0.000, 0.000)),
        (0.770, (0.000, 0.000, 0.000)),
        (1.000, (0.000, 0.000, 0.000))))
    cmap.set_bad(color=bg_color)
    masked = np.ma.masked_where(img==-1,img)
    ax.imshow(masked,cmap=cmap,origin='lower')

def merge_maps(maps_data:list, maps_resolution:list, maps_origin:list, maps_size:list, threshold=70):
    '''
    Returns the map resulting from the merging of the maps *maps_data*.

    Parameters
    ----------
    maps_data: list[list[int]]
        Ordered list of data fields of nav_msgs/OccupancyGrid.msg
    maps_resolution: list[float]
        Ordered list of map resolutions
    maps_origin: list[tuple[float,float]]
        Ordered list of map origins; each origin is a tuple containing its x,y
    maps_size: list[tuple[int,int]]
        Ordered list of map sizes; each size is a tuple containing its height and width
    threshold: int
        Threshold above which a cell is considered an obstacle
    
    Returns
    -------
    merged_map: list[list[int]]
        The map resulting from the merging of the maps *maps_data*. The merging
        is an overlapping of the maps based on their origins.
    '''
    grid = {}
    min_x = inf 
    min_y = inf
    max_x = -inf
    max_y = -inf
    for index in range(len(maps_data)):
        occupancy = maps_data[index]
        resolution = maps_resolution[index]
        origin_x, origin_y = maps_origin[index]
        height, width = maps_size[index]
        for i in range(width):
            for j in range(height):
                if occupancy[i+j * width] == -1:
                    continue
                if occupancy[i+j * width] > threshold:
                    x = (i * resolution) + origin_x
                    y = (j * resolution) + origin_y
                    p = (x, y)
                    grid[p] = True
                elif occupancy[i+j * width] <= threshold:
                    x = (i * resolution) + origin_x
                    y = (j * resolution) + origin_y
                    p = (x, y)
                    if p not in grid:
                        grid[p] = False
                if x<min_x:
                    min_x=x
                elif x>max_x:
                    max_x=x
                if y<min_y:
                    min_y=y
                elif y>max_y:
                    max_y=y
    h,w = round(abs(max_y-min_y)/resolution),round(abs(max_x-min_x)/resolution)
    o_x,o_y = min_x,min_y
    img = np.zeros((h+1,w+1))-1
    for p in grid:
        x,y = p
        x,y = round((x-o_x)/resolution),round((y-o_y)/resolution)
        img[y,x] = 100 if grid[p] else 0
    return img