# SimpleOCR
Preprocessing
Thresholding
An implementation of Ostu Thresholding method yielded undesirable results so we just picked a relatively high threshold so that we can get as many pixels as possible
Binary Morphology
After thresholding we faced a problem of the characters not being connected. In particular the letter ‘O’ was almost never fully connected. To solve this problem we used binary morphology to dilate the image. A portion of the code is shown below. The rest can be found in preprocess.py.

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
    
    #dilate once then erode twice
    img_binary = morphology.binary_dilation(img_binary, selem1)
    img_binary = morphology.binary_erosion(img_binary, selem2)
    img_binary = morphology.binary_erosion(img_binary, selem2)

We first dilated mostly in the vertical direction because the O shapes were always vertically disconnected. The bounding boxes were on top of one another.

We used a erosion element to erode horizontally because a problem occurred as a result of the dilation. Some letters became too wide. They intersected with the white border, causing the connected component algorithm to mistaken two far apart letters as one giant letter. The specific structuring element that we used also shifted the bounding boxes to the left, further contributing to our solution. 


Outlier Elimination
After applying our binary morphology, the next step is to get rid of noise. It was found that the amount of erosion necessary to eliminate noise broke up the letters to the point where it was not possible to connect them. Furthermore, dilating to reverse the erosion turned the letters into blobs with no distinguishable features. Rather than using morphology again, which would distort the letters, a simpler solution would be to ignore the bounding boxes that are outliers. 

        minr, minc, maxr, maxc = props.bbox
        width = maxc - minc
        height = maxr - minr
        
        #check for outliers
        if stats.is_outlier(width, height):
            #print width, height
            continue

The above code checks each boundary box and tests to see if it is an outlier. 

To determine whether a bounding box is an outlier, we calculated the average and standard deviation for width, height, area, and perimeter for all bounding boxes in the test data. All of the code is included in get_stats.py 

    #recalculate after getting rid of outliers
    def remove_outliers(self):
        while True:
            outlier_count = 0
            for width,height in self.width_height:
                if self.is_outlier(width,height):
                    outlier_count += 1
                    #remove all instances of (w,h)
                    self.width_height.remove( (width,height) )
                    #everytime outliers are removed, recalculate stats
                    self.calculate_stats()
             if outlier_count == 0:
                break

The above code removes outliers using the backwards elimination method. The average and standard deviation were calculated excluding any outliers. This led to significant differences in standard deviation.






Changes due to backward elmination


avg before
avg after
std before
std after
Perimeter
61
61
22.07
11.18
Area
954
931
957.33
331.91
Width
27
27
8.54
7.07
Height
33
33
18.95
7.87


 def is_outlier(self, w,h):
       area = w * h
        perim = w + h
        temp_sigma = self.sigma
    
        increase = 1.3
        #we are more lenient for large objects
        if w - self.width_avg  > 0:
            temp_sigma += increase
        if abs(self.width_avg - w) > temp_sigma * self.width_std:
            #print "width"
            return True
        temp_sigma = self.sigma

The above code takes checks to see if the input width is within sigma standard deviations of the average width of all bounding boxes. Similarly, the code tests for outliers in terms of height, area, and perimeter. The code is omitted to save space.

Boxes that were slightly larger than the average were classified as outliers. To solve this problem, a greater amount of leniency was allowed for boxes that were considered too big. That was accomplished by using a large sigma=4.3 (for the rest of our cases we used sigma=3). This was necessary because our bounding boxes were left skewed. The majority of the boxes were small, therefore it was necessary to allow for more leniency when detecting for bounding boxes that are too big. 

Before adding leniency, the outliers detected had widths and heights of (52,45) (50,39) and (56,34). While these are large, they are not actually outliers. It was determined that noise is mostly small boxes, so it is ok to allow leniency for larger boxes. 
Training
Distance Matrix
We used the code provided to create a distance matrix. It is shown below.

Below is the confusion matrix using Euclidean Distance as our classifier.

The accuracy for the readings are not the best.

accu for a 45%
accu for d 72.5%
accu for m 37.5%
accu for n 41.25%
accu for o 85%
accu for p 50%
accu for q 20%
accu for r 56.25%
accu for u 26.25%
accu for w 33.75%
K-Nearest-Neighbors 

The accuracy did not improve by much compared to Euclidean Distance. 

accu for a 61.25%
accu for d 67.5%
accu for m 43.75%
accu for n 75.%
accu for o 91.25%
accu for p 51.25%
accu for q 21.25%
accu for r 68.75%
accu for u 31.25%
accu for w 41.25%



Enhancements
Center
We added additional features. Letters appear to have very unique centroids. Hence it is a good idea use this property as a features. Since each bounding box size is different, we divide the y coordinate of the center by the height and we divide the x coordinate of the center by the width. This fortunately yielded a good result. The accuracy went up for all but one letter.


accu for a 80%
accu for d 83.75%
accu for m 76.25%
accu for n 48.75%
accu for o 95%
accu for p 71.25%
accu for q 41.25%
accu for r 78.75%
accu for u 38.75%
accu for w 52.5%
Projections
We added additional features. Letters appear to have very unique projections. Hence it is a good idea use projections as a features. Since each bounding box size is different, the box is vertically divided into 5 intervals. We then add up the amount of pixels inside each of the ten intervals. The same is done horizontally. This yielded amazing results as the results went up exponentially for almost every character. 

accu for a 95.0%
accu for d 98.75%
accu for m 91.25%
accu for n 97.5%
accu for o 100.0%
accu for p 96.25%
accu for q 86.25%
accu for r 97.5%
accu for u 88.75%
accu for w 95.0%

Simple Vector Machine
Since letters tend to be similar except with a few differences, SVM might be a machine learning algorithm that is worth trying. In any case it does not hurt to try a different learning algorithm. As expected, SVM yielded pretty good results. The results however, are similar to that of k-nearest-neighbors. Hence, this method is not necessarily an improvement. 

accu for a 96.25%
accu for d 97.5%
accu for m 82.5%
accu for n 96.25%
accu for o 100.0%
accu for p 92.5%
accu for q 93.75%
accu for r 97.5%
accu for u 93.75%
accu for w 92.5%
Testing
K-Nearest-Neighbor 
Number of objects detected
Recognition Rate
Threshold
70
37.1428571429%
230

SVM with Center Feature
Number of objects detected
Recognition Rate
Threshold
70
51.4285714286%
230
K-Nearest-Neighbor with Center Feature
Number of objects detected
Recognition Rate
Threshold
70
54.2857142857%
230

K-Nearest-Neighbor with Projection Features
Number of objects detected
Recognition Rate
Threshold
number detected 70
81.4285714286%
230

SVM with Projection Features
Number of objects detected
Recognition Rate
Threshold
70
81.4285714286%
230

SVM with Projection and Center Features
Number of objects detected
Recognition Rate
Threshold
70
81.4285714286%
230


K-Nearest-Neighbor with Projection and Center Features
Number of objects detected
Recognition Rate
Threshold
70
81.4285714286%
230

 
It seems that for training data projections and center coordinates were the best features to use. In practice it seems that projection is the only feature that is needed. It seems that K-Nearest-Neighbors and SVM have the same performance. 



Running the Code
Changes only need to be made to RunMyOCRRecognition.py
Training Files


To train on different files, change the file names in the list train_filenames. The first letter of the file must correspond to the letter it contains. The dictionaries may also have to modified similarly.
Test Files

Modify test_filename to run the code on a different file. 
