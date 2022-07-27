# conda create -y -n napari_e python=3.9
# conda activate napari_e
# pip install napari[all] in a new conda environment
# pip install
import napari
from skimage import io

# Load small image to the napari viewer from python console
viewer=napari.Viewer()
img = io.imread(r'treated_img.jpg')
viewer = napari.view_image(data=img, rgb=True)

#Load Whole slide image in the napari

#
