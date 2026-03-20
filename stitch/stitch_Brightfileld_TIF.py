import numpy as np
import pickle
import os
import glob
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import re
import tifffile



def natural_key(string_):

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='='): #chr(0x00A3)
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration / total)
    bar = fill * filledLength + '>' + '.' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end="")
    # Print New Line on Complete
    if iteration == total:
        print()

def get_stitched_dimension_image_from_tiles(cws_folder,annotated_dir,output_dir,scale):

    wsi_files = sorted(glob.glob(os.path.join(cws_folder, '*.ndpi')))

    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    for wsi in range(0, len(wsi_files)):

        filename = wsi_files[wsi]

        param = pickle.load(open(os.path.join(filename, 'param.p'), 'rb'))

        slide_dimension = np.array(param['slide_dimension']) / param['rescale']
        print(slide_dimension, scale)

        slide_w, slide_h = slide_dimension
        cws_w, cws_h = param['cws_read_size']

        divisor_w = np.ceil(slide_w / cws_w)
        divisor_h = np.ceil(slide_h / cws_h)

        w, h = int(slide_w / scale), int(slide_h / scale)
        print('%s, Ss1 size: %i,%i'%(os.path.basename(filename),w,h))
        img_all = np.zeros((h, w, 3))

        drivepath, imagename = os.path.split(wsi_files[wsi])
        annotated_dir_i = os.path.join(annotated_dir, imagename)
        images = sorted([fname for fname in os.listdir(annotated_dir_i) if fname.startswith('Da') and fname.endswith('.jpg') is True], key=natural_key)
        print(len(images))
        printProgressBar(0, len(images), prefix='Progress:', suffix='Complete', length=50)

        for i in images:
            cws_i = int(re.search(r'\d+', i).group())
            h_i = np.floor(cws_i / divisor_w) * cws_h
            w_i = (cws_i - h_i / cws_h * divisor_w) * cws_w

            h_i = int(h_i / scale)
            w_i = int(w_i / scale)
            # print(cws_i, w_i, h_i)

            img = cv2.imread(os.path.join(annotated_dir_i,i))

            img = cv2.resize(img, (int(img.shape[1]/1), int(img.shape[0]/1)))

            img_all[h_i : h_i + int(img.shape[0]), w_i : w_i + int(img.shape[1]),:] = img

            printProgressBar(cws_i, len(images), prefix='Progress:',
                             suffix='Completed for %s'%i, length=50)

            #if w_i + cws_w / scale > w:
                #cv2.imwrite(os.path.join(output_dir,imagename + "_out.png"), img_all)

        with tifffile.TiffWriter(f'{os.path.join(output_dir,imagename + "_out" )}.tif', bigtiff=True) as tif:
            options = dict(
                photometric=tifffile.TIFF.PHOTOMETRIC.MINISBLACK,
                tile=(800, 800),
                compression=tifffile.TIFF.COMPRESSION.ADOBE_DEFLATE,
                planarconfig=tifffile.TIFF.PLANARCONFIG.CONTIG,
                metadata=None)
        #tif.write(img_all)
        tif.write(img_all, photometric='rgb')
        print(img_all.shape)


if __name__ == '__main__':

    params = {
                'cws_folder' : os.path.normpath(r'C:\Projects\proj_4\os_image_package\tiles'),
                'annotated_dir' : r'C:\Projects\proj_4\os_image_package\classify',
                'output_dir' : r'C:\Projects\proj_4\os_image_package\lowres',
                'scale' : 1
               }
    # params = {
    #     'c'
    # }
    get_stitched_dimension_image_from_tiles(params['cws_folder'], params['annotated_dir'], params['output_dir'],
                                                params['scale'])
