import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, morphology
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

label_rev_mapping = {1:'a',2:'d', 3:'m', 4:'n', 5:'o', 6:'p', 7:'q', 8:'r', 9:'U', 10:'w'}

def test_on(filename,
            mean_arr,
            std_arr,
            classifier,
            stats,
            total_features,
            plot):
    total_features = [0] * len(total_features)
    #get groundtruth values
    pkl_file = open('test1_gt.pkl.txt', 'rb')
    mydict = pickle.load(pkl_file)
    pkl_file.close()
    classes = mydict['classes']
    locations = mydict['locations']
    
    #read file, preprocess, and make predictions
    from preprocess import process
    img_binary, img_label = process(filename, plot)

    regions = regionprops(img_label)
    io.imshow(img_binary)
    ax = plt.gca()
    plt.title('Bounding boxes')  
    
    Features = []
    bbox_centers = []
    
    from preprocess import extract_features
    extract_features(regions=regions,
                     number=None,
                     img_binary=img_binary,
                     ax=ax,
                     Features=Features,
                     bbox_centers=bbox_centers,
                     label_tracking=None,
                     total_features=total_features,
                     stats=stats)
    #plot = True
    if plot:
        io.show()
    
    #normalize Features with respct to training stats
    features_arr = np.array(Features)
    features_arr = np.subtract(features_arr, mean_arr)
    features_arr = np.divide(features_arr, std_arr)
    
    #make predictions
    y_pred = classifier.predict(features_arr)
    
    #find closest center
    D = cdist(bbox_centers, locations)
    D_index = np.argsort(D, axis=1)
    y_real = [classes[D_index[i][0]] for i in range(0,len(D_index))]
    
    y_pred = [label_rev_mapping[i] for i in y_pred]
    print "real"
    print y_real
    print "predicted"
    print y_pred
        
    results = [1 if pred == real else 0 for pred,real in zip(y_pred,y_real)]
    #print results
    accuracy = float(sum(results))/len(results)
    print "number detected " + str(len(y_pred))
    print "accur " + str(accuracy)
    

