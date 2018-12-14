import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage import io, exposure
import matplotlib.pyplot as plt
import pickle

plot = False


#get stats about entire test set
from get_stats import Stats
stats = Stats(label_mapping, sigma=3)
stats.calculate_stats()
stats.remove_outliers()

#train our data
train_filenames = ['a.bmp','d.bmp','m.bmp','n.bmp','o.bmp','p.bmp','q.bmp','r.bmp','U.bmp','w.bmp']
label_mapping = {'a':1,'d':2,'m':3,'n':4,'o':5,'p':6,'q':7,'r':8,'U':9,'w':10}
label_rev_mapping = {1:'a',2:'d', 3:'m', 4:'n', 5:'o', 6:'p', 7:'q', 8:'r', 9:'U', 10:'w'}
Features = []
total_features = [0]*19
label_tracking = []
from train import train_on
for filename in train_filenames:
    number = label_mapping[filename[0]]
    train_on(filename = filename,
             Features = Features,
             label_tracking = label_tracking,
             total_features = total_features,
             number = number,
             stats = stats,
             plot = plot)

#normalize Features
features_arr = np.array(Features)
num_samples = len(Features)
mean_list = [feat_total/num_samples for feat_total in total_features]
mean_arr = np.array(mean_list)
std_arr = np.std(features_arr, axis=0)
features_arr = np.subtract(features_arr,mean_arr)
features_arr = np.divide(features_arr, std_arr)

###use euclidean distance
##D = cdist(features_arr, features_arr)
##D_index = np.argsort(D, axis=1)
##y_pred = [label_tracking[D_index[i][1]] for i in range(0,num_samples)]
##if plot:
##    io.imshow(D)
##    plt.title('Distance Matrix')
##    io.show()

###use random-forest
##from sklearn.ensemble import RandomForestClassifier
##clf = RandomForestClassifier(n_estimators=75, max_depth=4, random_state=0)
##clf.fit(features_arr, label_tracking)
##y_pred = clf.predict(features_arr)

###use svm
##from sklearn import svm
##clf = svm.SVC(gamma='scale')
##clf.fit(features_arr, label_tracking)
##y_pred = clf.predict(features_arr)

#use k-neighbors
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=len(label_mapping.keys()))
clf.fit(features_arr, label_tracking)
y_pred = clf.predict(features_arr)

#confusion matrix
confM = confusion_matrix(label_tracking, y_pred)
plot = True
if plot:
    io.imshow(confM)
    plt.title('Confusion Matrix')
    io.show()
    
#calculate accuracy
print "training data"
for i in range(0, len(confM)):
    print "accu for " + label_rev_mapping[i+1] + " " + str(float(confM[i][i])*1.25) + "%"
        
#use classifier on test data
from test import test_on
test_filename = 'test1.bmp'
test_on(filename=test_filename,
        mean_arr=mean_arr,
        std_arr=std_arr,
        classifier=clf,
        stats=stats,
        total_features=total_features,
        plot=True)

 
