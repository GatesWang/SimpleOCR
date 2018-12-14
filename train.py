import numpy as np
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, morphology
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from get_stats import Stats

def train_on(filename,
             Features,
             label_tracking,
             total_features,
             number,
             stats,
             plot):
    total_features = [0] * len(total_features)
    #preprocess
    from preprocess import process
    img_binary, img_label = process(filename, plot)
    
    regions = regionprops(img_label)
    io.imshow(img_binary)
    ax = plt.gca()
    plt.title('Bounding boxes')

    from preprocess import extract_features
    extract_features(regions=regions,
                     number=number,
                     img_binary=img_binary,
                     ax=ax,
                     Features=Features,
                     bbox_centers = None,
                     label_tracking=label_tracking,
                     total_features=total_features,
                     stats=stats)

    #plot = True
    if plot:
        io.show()

    


















