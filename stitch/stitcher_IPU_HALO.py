from glob import glob
import os
import math
from concurrent.futures import ThreadPoolExecutor, wait
import time

import tifffile
import numpy as np
import zarr
from tqdm import tqdm

folders = [el for el in os.listdir('.') if os.path.isdir(el)]

print(f"Found {len(folders)} cases")

def stitch_case(folder_path):
    
    print(f'Started stitching {folder_path}')
    
    file_names = []
    x_positions = []
    y_positions = []
    channel_names = []

    IMAGE_X_RESOLUTION = None
    IMAGE_Y_RESOLUTION = None
    
    def get_xy(file_name):
        nonlocal IMAGE_X_RESOLUTION
        nonlocal IMAGE_Y_RESOLUTION
        nonlocal channel_names
        with tifffile.TiffFile(file_name) as file:
            if not channel_names:
                channel_names = [file.pages[el].tags['ImageDescription'].value.split('Name>')[3].split('<')[0] for el in range(8)]            
            
            xres = file.pages[0].tags['XResolution'].value
            IMAGE_X_RESOLUTION = xres
            xres = xres[0] / xres[1]
            
            yres = file.pages[0].tags['YResolution'].value
            IMAGE_Y_RESOLUTION = yres
            yres = yres[0] / yres[1]
            
            xpos = file.pages[0].tags['XPosition'].value
            xpos = xpos[0] / xpos[1]
            
            ypos = file.pages[0].tags['YPosition'].value
            ypos = ypos[0] / ypos[1]
            
            x = int(xres * xpos)
            y = int(yres * ypos)
            
            return x, y 
    
    for file in glob(os.path.join(folder_path, '*.tif')):
        file_names.append(file)
        x, y = get_xy(file)
        x_positions.append(x)
        y_positions.append(y)
    
    max_x = max(x_positions) + 1862
    max_y = max(y_positions) + 1396
    min_x = min(x_positions)
    min_y = min(y_positions)
    image_x = max_x - min_x
    image_y = max_y - min_y
    
    canvas = zarr.zeros((8, image_y, image_x), dtype=np.float32)
    
    def write_to_canvas(arr):
        f, x, y = arr
        nonlocal canvas
        nonlocal min_x
        nonlocal min_y
        
        x = x - min_x
        y = y - min_y
        canvas[:, y:y+1396, x:x+1860] = tifffile.imread(f)
        
    
    print('Reading images and stitching...')
    st = time.monotonic()
    with ThreadPoolExecutor(len(file_names)) as t:
        fts = [t.submit(write_to_canvas, [file, x, y]) for file, x, y in zip(file_names, x_positions, y_positions)]
        wait(fts)
    print(f"Finished reading {len(file_names)} images and stitching in {time.monotonic() - st:.2f} seconds")
    
    def get_num_of_subresolutions():
        nonlocal image_x
        nonlocal image_y
        return int(math.log(max(image_x, image_y), 2)) - 10 # So that the smallest image has dimensions between 1024-2048
    
    pixel_metadata = []

    for i in range(get_num_of_subresolutions()):
        for j in range(8):
            pixel_metadata.append(f'<dimension sizeX="{math.ceil((image_x) / (2**i))}" sizeY="{math.ceil((image_y) / (2**i))}" ifd="{i*8 + j}" channel="{j}" level="{i}"/>\n')
    
    
    with tifffile.TiffWriter(f'{os.path.basename(folder_path)}.tif', bigtiff=True) as tif:
        options = dict(
            photometric=tifffile.TIFF.PHOTOMETRIC.MINISBLACK,
            tile=(800, 800),
            compression=tifffile.TIFF.COMPRESSION.ADOBE_DEFLATE,
            planarconfig = tifffile.TIFF.PLANARCONFIG.CONTIG,
            metadata=None,
            extratags= [
                        (274, tifffile.TIFF.DATATYPES.SHORT, 1, tifffile.TIFF.ORIENTATION.TOPLEFT, False)
                ]
        )
        
        tif.write(
            canvas[0, :, :],
            resolution = (IMAGE_X_RESOLUTION, IMAGE_Y_RESOLUTION, 'CENTIMETER'),
            software='IndicaLabsImageWriter v1.2.1',
            description= f"""<?xml version="1.0" encoding="utf-8"?>
                                        <indica>
                                            <post_proc type="0"/>
                                            <image>
                                                <pixels>
                                                    {
                                                        '                                                    '.join(pixel_metadata)
                                                    }
                                                </pixels>
                                                <channels>
                                                    <channel id="0" name="{channel_names[0]}" rgb="255" min="0.000000" max="{canvas[0, :, :].max():.6f}"/>
                                                    <channel id="1" name="{channel_names[1]}" rgb="65535" min="0.000000" max="{canvas[1, :, :].max():.6f}"/>
                                                    <channel id="2" name="{channel_names[2]}" rgb="65280" min="0.000000" max="{canvas[2, :, :].max():.6f}"/>
                                                    <channel id="3" name="{channel_names[3]}" rgb="16776960" min="0.000000" max="{canvas[3, :, :].max():.6f}"/>
                                                    <channel id="4" name="{channel_names[4]}" rgb="16744448" min="0.000000" max="{canvas[4, :, :].max():.6f}"/>
                                                    <channel id="5" name="{channel_names[5]}" rgb="16711680" min="0.000000" max="{canvas[5, :, :].max():.6f}"/>
                                                    <channel id="6" name="{channel_names[6]}" rgb="16777215" min="0.000000" max="{canvas[6, :, :].max():.6f}"/>
                                                    <channel id="7" name="{channel_names[7]}" rgb="0" min="0.000000" max="{canvas[7, :, :].max():.6f}"/>
                                                </channels>
                                                <objective value="10.000000"/>
                                            </image>
                                        </indica>""",
            **options
        )
        
        st = time.monotonic()
        
        print("Starting to write the full size image")
        for i in tqdm(range(7)):
            tif.write(
                canvas[i+1, :, :],
                resolution = (IMAGE_X_RESOLUTION, IMAGE_Y_RESOLUTION, 'CENTIMETER'),
                software='IndicaLabsImageWriter v1.2.1',
                **options
            )
        
        print("Starting to write the downsampled images in the pyramid")
        for level in tqdm(range(get_num_of_subresolutions())):
            mag = 2**(level + 1)
            subsampled_canvas = canvas[:, ::mag, ::mag]
            for j in range(8):
                tif.write(
                    subsampled_canvas[j, :, :],
                    resolution = (IMAGE_X_RESOLUTION, IMAGE_Y_RESOLUTION, 'CENTIMETER'),
                    software='IndicaLabsImageWriter v1.2.1',
                    subfiletype=1,
                    **options
                )
        
        print(f'Finished writing the output image in {time.monotonic() - st:.2f} seconds')

for folder in folders:
    stitch_case(folder)
