'''
Support for stitching of tiles coming from inform process for multiplex
immunofluorescence images to be processed with qupath software for
analysis within qupath.

Qupath internally depends on openmicroscopy standardisation and requires
specific tags related to the pixels and channels and the regular tiff tags.

#############################
Role of Parallelisation
#############################

ThreadPoolExecutor (from the `concurrent.futures` module) and `tqdm` is used in the code.
ThreadPoolExecutor` and `wait` (from `concurrent.futures`)**
**Purpose**: These tools are used for managing and executing tasks concurrently
(in parallel), which can significantly speed up operations that are independent
and can be run simultaneously.

#### How it works ####
- **`ThreadPoolExecutor`**:
  This class is used to create a pool of threads. Each thread can run a separate task simultaneously.
  You define how many threads you want to run concurrently by specifying the number of workers.
  For example, `ThreadPoolExecutor(len(file_names))` creates a pool with as many threads as
  there are files to process.


 Tiff tags
1. photometric=tifffile.TIFF.PHOTOMETRIC.MINISBLACK
Photometric Interpretation:

The photometric option defines how pixel data should be interpreted in terms of color.
MINISBLACK: This setting indicates that the image is grayscale, where the minimum pixel value
is black, and the maximum is white. This is typical for single-channel images such as grayscale images.

2. tile=(800, 800)
Tiling:

The tile option specifies the size of the tiles into which the image is divided.
In this case, the image is split into 800x800 pixel tiles.

Why Tile?: Tiling can improve performance, especially when working with
large images, because it allows you to read or write small portions of the
image independently rather than the entire image at once. This is particularly
useful for pyramidal images or when viewing large images in a viewer that only
needs to load part of the image at a time.


3. compression=tifffile.TIFF.COMPRESSION.ADOBE_DEFLATE
Compression:

The compression option specifies the method used to compress the image data, which can reduce file size.
ADOBE_DEFLATE: This is a lossless compression algorithm similar to ZIP compression.
It reduces the file size without losing any image data, making it a good choice when you
need to save space but want to preserve the original quality of the image.

4. planarconfig=tifffile.TIFF.PLANARCONFIG.CONTIG
Planar Configuration:

The planarconfig option determines how the image data for multiple channels (e.g., RGB) is organized in the file.
CONTIG (Contiguous): In this configuration, all channels for each pixel are stored together.
For example, in an RGB image, the red, green, and blue values for a single pixel are stored
together in a single location (e.g., R1G1B1, R2G2B2).
This contrasts with the SEPARATE planar configuration, where each channel is stored in a
separate plane (e.g., all red values are stored together, followed by all green, then blue).

5. metadata=None
Metadata Handling:

The metadata option allows you to specify additional metadata for the TIFF file. Setting it to
None means that no extra metadata is added beyond what is necessary.
Why None?: Sometimes, omitting metadata can simplify the file or avoid compatibility
issues with specific software that might misinterpret custom metadata.

6. extratags
Custom Tags:

The extratags option allows you to add custom TIFF tags to the file.
Each tag is specified as a tuple with the following components:
Tag ID: The identifier for the tag.
Data Type: The type of data the tag will store (e.g., SHORT, LONG, etc.).
Count: How many values the tag contains.
Value: The actual value(s) for the tag.
Write Once: A flag indicating if the tag should only be written once.

Summary
These options configure various aspects of how the TIFF file is written:

Photometric Interpretation: Defines how pixel values are interpreted (e.g., grayscale).
Tiling: Splits the image into smaller sections, improving performance for large images.
Compression: Reduces file size without losing image quality.
Planar Configuration: Determines how multi-channel data is stored.
Metadata Handling: Controls the inclusion of additional metadata.
Custom Tags: Allows for the addition of specific TIFF tags, such as image orientation.
These options are essential for creating optimized, compatible TIFF files, especially in
contexts like medical imaging, microscopy, or large-scale image analysis where specific configurations are required.

'''
from glob import glob
import os
import math
from concurrent.futures import ThreadPoolExecutor, wait
import time

import tifffile
import numpy as np
import zarr
from tqdm import tqdm

# Specify the directory path
directory_path = r'E:\Miki\for_stitching'
st_dir = r'E:\Miki\30_Aug_2024'


# List of folders in the specified directory, excluding the `.idea` directory
folders = [el for el in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, el)) and el != 'Run_1']

# # List of folders to process, excluding the `.idea` directory
# folders = [el for el in os.listdir('.') if os.path.isdir(el) and el != '.idea']

print(f"Found {len(folders)} cases")


def stitch_case(folder_path):
    print(f'Started stitching {folder_path}')
    # output_file = os.path.join(st_dir, f"{os.path.basename(folder_path)[:5]}_Stitched.ome.tif")
    # print(output_file)
    colors = [(0, 0, 255),
              (0, 255, 255),
              (0, 255, 0),
              (255, 255, 0),
              (255, 165, 0),
              (255, 0, 0),
              (255, 255, 255),
              (0, 0, 0)
              ]

    file_names = []
    x_positions = []
    y_positions = []
    channel_names = []

    IMAGE_X_RESOLUTION = None
    IMAGE_Y_RESOLUTION = None

    def get_xy(file_name):
        # print(file_name)
        nonlocal IMAGE_X_RESOLUTION
        nonlocal IMAGE_Y_RESOLUTION
        nonlocal channel_names
        with tifffile.TiffFile(file_name) as file:
            if not channel_names:
                channel_names = [file.pages[el].tags['ImageDescription'].value.split('Name>')[3].split('<')[0] for el in
                                 range(8)]

                # Extract the actual resolution value
            xres = file.pages[0].tags['XResolution'].value
            IMAGE_X_RESOLUTION = xres[0] / xres[1]

            yres = file.pages[0].tags['YResolution'].value
            IMAGE_Y_RESOLUTION = yres[0] / yres[1]
            inverse = 1 / IMAGE_Y_RESOLUTION
            # print(f'{inverse:.4e}')

            xpos = file.pages[0].tags['XPosition'].value
            xpos = xpos[0] / xpos[1]

            ypos = file.pages[0].tags['YPosition'].value
            ypos = ypos[0] / ypos[1]

            x = int(IMAGE_X_RESOLUTION * xpos)
            y = int(IMAGE_Y_RESOLUTION * ypos)

            return x, y

    for file in glob(os.path.join(directory_path, folder_path, '*.tif')):
        file_names.append(file)
        # print(file)
        x, y = get_xy(file)
        x_positions.append(x)
        y_positions.append(y)

    max_x = max(x_positions) + 1862
    max_y = max(y_positions) + 1396
    min_x = min(x_positions)
    min_y = min(y_positions)
    image_x = max_x - min_x
    image_y = max_y - min_y

    # Create a canvas with dimensions (image_x, image_y, 8) for 8 channels
    canvas = zarr.zeros((image_y, image_x, 8), dtype=np.float32)

    def write_to_canvas(arr):
        f, x, y = arr
        nonlocal canvas
        nonlocal min_x
        nonlocal min_y

        x = x - min_x
        y = y - min_y

        # Read the tile and transpose it from (8, y, x) to (x, y, 8) and change dtype to float32
        tile = tifffile.imread(f).astype(np.float32)
        tile_transposed = np.transpose(tile, (1, 2, 0))  # Shape now (y, x, 8)
        #
        # # Insert the transposed tile into the canvas
        # canvas[x:x + tile_transposed.shape[0], y:y + tile_transposed.shape[1], :] = tile_transposed
        canvas[y:y+1396, x:x+1860, :] = tile_transposed

    print('Reading images and stitching...')
    st = time.monotonic()
    with ThreadPoolExecutor(len(file_names)) as t:
        fts = [t.submit(write_to_canvas, [file, x, y]) for file, x, y in zip(file_names, x_positions, y_positions)]
        wait(fts)
    print(f"Finished reading {len(file_names)} images and stitching in {time.monotonic() - st:.2f} seconds")

    def get_num_of_subresolutions():
        nonlocal image_x
        nonlocal image_y
        return int(math.log(max(image_x, image_y), 2)) - 10  # Smallest image has dimensions between 1024-2048

    pixel_metadata = []

    for i in range(get_num_of_subresolutions()):
        for j in range(8):
            pixel_metadata.append(
                f'<dimension sizeX="{math.ceil((image_x) / (2 ** i))}" sizeY="{math.ceil((image_y) / (2 ** i))}" ifd="{i * 8 + j}" channel="{j}" level="{i}"/>\n')
    inverse = 0.4967
    ome_description = f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="{os.path.basename(folder_path)}">
    <Pixels DimensionOrder="XYZCT"
            ID="Pixels:0"
            Type="float"
            SizeX="{image_x}"
            SizeY="{image_y}"
            SizeZ="1"
            SizeC="8"
            SizeT="1"
            PhysicalSizeX="{inverse:4e}"
            PhysicalSizeY="{inverse:4e}"
            ResolutionUnit="CENTIMETER"
            Magnification="10"
            SignificantBits="16"
            BigEndian="false">
        {
    ''.join(pixel_metadata)
    }
        <Channel ID="Channel:0:0" Name="{channel_names[0]}" SamplesPerPixel="1" Color="65535">
          <LightPath/>
        </Channel>
        <Channel ID="Channel:0:1" Name="{channel_names[1]}" SamplesPerPixel="1" Color="2047">
          <LightPath/>
        </Channel>
        <Channel ID="Channel:0:2" Name="{channel_names[2]}" SamplesPerPixel="1" Color="2016">
          <LightPath/>
        </Channel>
        <Channel ID="Channel:0:3" Name="{channel_names[3]}" SamplesPerPixel="1" Color="65504">
          <LightPath/>
        </Channel>
        <Channel ID="Channel:0:4" Name="{channel_names[4]}" SamplesPerPixel="1" Color="64800"> 
          <LightPath/>
        </Channel>
        <Channel ID="Channel:0:5" Name="{channel_names[5]}" SamplesPerPixel="1" Color="63488">
          <LightPath/>
        </Channel>
        <Channel ID="Channel:0:6" Name="{channel_names[6]}" SamplesPerPixel="1" Color="64988">
          <LightPath/>
        </Channel>
        <Channel ID="Channel:0:7" Name="{channel_names[7]}" SamplesPerPixel="1" Color="0">
          <LightPath/>
        </Channel>
    </Pixels>
  </Image>
</OME>"""
    output_file = os.path.join(st_dir, f"{os.path.basename(folder_path)}_120MCF_Stitched.ome.tif")
    with tifffile.TiffWriter(output_file, bigtiff=True) as tif:
        options = dict(
            photometric=tifffile.TIFF.PHOTOMETRIC.MINISBLACK,
            tile=(800, 800),
            resolutionunit=tifffile.TIFF.RESUNIT.CENTIMETER,
            compression=tifffile.TIFF.COMPRESSION.ADOBE_DEFLATE,
            planarconfig=tifffile.TIFF.PLANARCONFIG.CONTIG,
            metadata=None,
            extratags=[
                (274, tifffile.TIFF.DATATYPES.SHORT, 1, tifffile.TIFF.ORIENTATION.TOPLEFT, False)
            ]
        )

        # Write the first channel of the full-size image with metadata
        tif.write(
            canvas[:, :, 0],
            resolution=(IMAGE_X_RESOLUTION, IMAGE_Y_RESOLUTION, 'CENTIMETER'),
            software='IPU_ICR',
            description=ome_description,
            **options
        )

        st = time.monotonic()

        print("Starting to write the remaining channels of the full-size image")
        for i in tqdm(range(1, 8)):
            tif.write(
                canvas[:, :, i],
                resolution=(IMAGE_X_RESOLUTION, IMAGE_Y_RESOLUTION, 'CENTIMETER'),
                software='IPU_ICR',
                **options
            )

        print("Starting to write the downsampled images in the pyramid")
        for level in tqdm(range(get_num_of_subresolutions())):
            mag = 2 ** (level + 1)
            subsampled_canvas = canvas[::mag, ::mag, :]
            for j in range(8):
                tif.write(
                    subsampled_canvas[:, :, j],
                    resolution=(IMAGE_X_RESOLUTION, IMAGE_Y_RESOLUTION, 'CENTIMETER'),
                    software='IPU_ICR',
                    subfiletype=1,
                    **options
                )

        print(f'Finished writing the output image in {time.monotonic() - st:.2f} seconds')


# Process each folder
for folder in folders:
    print(folder)
    stitch_case(folder)
