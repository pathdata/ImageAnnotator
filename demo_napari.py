import napari
from skimage import io

# Load small image to the napari viewer from python console
viewer=napari.Viewer()
img = io.imread(r'treated_img.jpg')
viewer = napari.view_image(data=img, rgb=True)
