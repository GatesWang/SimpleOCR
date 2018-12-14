import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, morphology
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import cv2
from threshold import *

#does median filter, thresholding, and morphology
#returns the binary image and labeled image
def process(filename, plot):
    #read file
    img = io.imread(filename)
    if plot:
        io.imshow(img)
        plt.title('Original Image')
        io.show()
        
    #apply median filter
    img = cv2.medianBlur(img,3)
    
    #image histogram
    hist = exposure.histogram(img)
    if plot:
        plt.title('Histogram')
        plt.bar(hist[1], hist[0])
        plt.title('Histogram')
        plt.show()
        
    #binarization
    th = 230
    img_binary = (img < th).astype(np.double)
    if plot:
        io.imshow(img_binary)
        plt.title('Binary Image')
        io.show()

    #morphology
    selem1 = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]],dtype=np.uint8)

    selem2 = np.array(
        [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]],dtype=np.uint8)
    
    #dilate twice then erode
    img_binary = morphology.binary_dilation(img_binary, selem1)
    img_binary = morphology.binary_erosion(img_binary, selem2)
    img_binary = morphology.binary_erosion(img_binary, selem2)

    #performs CCA to label each character
    img_label = label(img_binary, background=0)
    #print(np.amax(img_label))
    
    if plot:
        io.imshow(img_label)
        plt.title('Labeled Image')
        io.show()

    return img_binary, img_label

def extract_features(regions,
                    number,
                    img_binary,
                    ax,
                    Features,
                    bbox_centers,
                    label_tracking,
                    total_features,
                    stats):
    from features import RegionFeatures
    #go through each region of interest
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        width = maxc - minc
        height = maxr - minr

        #check for outliers
        if stats.is_outlier(width, height):
            #print width, height
            continue
        
        feats = RegionFeatures(img_binary, minr, minc, maxr, maxc)

        #these turn out to be bad features
        #longest_horizontal = feats.get_longest_horizontal()
        #longest_vertical = feats.get_longest_vertical()
        #percent_filled = feats.get_longest_vertical()
        
        x_center, y_center = feats.get_center()
        x_projections = feats.get_x_projections()
        y_projections = feats.get_y_projections()
        if not bbox_centers == None:
            bbox_centers.append(((maxc+minc)/2, (maxr+minr)/2))

        #plot box
        ax.add_patch(Rectangle((minc,minr), width, height,fill=False, edgecolor='red', linewidth=1))
        
        #computing Hu moments
        roi = img_binary[minr:maxr, minc:maxc]
        m = moments(roi)  
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        mu = moments_central(roi, cr, cc)
        nu = moments_normalized(mu)
        
        #append hu moments and other features to Feature list
        hu = moments_hu(nu)
        #hu = np.append(hu, np.array(longest_horizontal))
        #hu = np.append(hu, np.array(longest_vertical))
        #hu = np.append(hu, np.array(percent_filled))
        hu = np.append(hu, np.array(x_center))
        hu = np.append(hu, np.array(y_center))
        hu = np.append(hu, np.array(y_projections))
        hu = np.append(hu, np.array(x_projections))

        Features.append(hu)
        total_features = [sum(x) for x in zip(total_features, hu)]
        if number:
            label_tracking.append(number)
