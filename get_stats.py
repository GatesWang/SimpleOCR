import numpy as np
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from threshold import *

class Stats:
    def __init__(self, label_mapping, sigma):
        self.length = 0
        self.width_height = []
        self.sigma = sigma
        self.perim_avg = 0
        self.perim_std = 0
        self.area_avg = 0
        self.area_std = 0
        self.width_avg = 0
        self.width_std = 0
        self.height_avg = 0
        self.height_std = 0

        #for all test cases append width and height
        from preprocess import process
        for char in label_mapping.keys():
            filename = char + '.bmp'
            img_binary, img_label = process(filename, plot = False)
            regions = regionprops(img_label)
            for props in regions:
                minr, minc, maxr, maxc = props.bbox
                self.width_height.append( (maxc - minc, maxr - minr) )
            
    def calculate_stats(self):
        self.length = len(self.width_height)
        self.perim_avg, self.perim_std = self.calculate_perim_stats()
        self.area_avg, self.area_std = self.calculate_area_stats()
        self.width_avg, self.width_std = self.calculate_width_stats()
        self.height_avg, self.height_std = self.calculate_height_stats()
        
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
        
        if h - self.height_avg  > 0:
            temp_sigma += increase
        if abs(self.height_avg - h) > temp_sigma * self.height_std:
           #print "height"
           return True
        temp_sigma = self.sigma

        if area - self.area_avg  > 0:
            temp_sigma += increase
        if abs(self.area_avg - area) > temp_sigma * self.area_std:
            #print "area"
            return True
        temp_sigma = self.sigma
        
        if perim - self.perim_avg  > 0:
            temp_sigma += increase
        if abs(self.perim_avg - perim) > temp_sigma * self.perim_std:
            #print "perim"
            return True
        temp_sigma = self.sigma
        
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
        
    #returns mean and std for area
    def calculate_area_stats(self):
        area_avg = sum([w*h for w,h in self.width_height])/self.length
        area_std = (sum([((w*h)-area_avg)**2 for w,h in self.width_height])/self.length)**(.5)

        return area_avg, area_std

    #returns mean and std for perimeter
    def calculate_perim_stats(self):
        perim_avg = sum([w+h for w,h in self.width_height])/self.length
        perim_std = (sum([((w+h)-perim_avg)**2 for w,h in self.width_height])/self.length)**(.5)

        return perim_avg, perim_std
                      
    #returns mean and std for width
    def calculate_width_stats(self):
        width_avg = sum([w for w,h in self.width_height])/self.length
        width_std = (sum([(w-width_avg)**2 for w,h in self.width_height])/self.length)**(.5)

        return width_avg, width_std

    #returns mean and std for height
    def calculate_height_stats(self):
        height_avg = sum([h for w,h in self.width_height])/self.length
        height_std = (sum([(h-height_avg)**2 for w,h in self.width_height])/self.length)**(.5)

        return height_avg, height_std
    
    def print_stats(self):
        print  "perim " + str(self.perim_avg - self.sigma * self.perim_std) + " " + str(self.perim_avg + self.sigma * self.perim_std) 
        print  "area " + str(self.area_avg - self.sigma * self.area_std) + " " + str(self.area_avg + self.sigma * self.area_std) 
        print  "width " + str(self.width_avg - self.sigma * self.width_std) + " " + str(self.width_avg + self.sigma * self.width_std) 
        print  "height " + str(self.height_avg - self.sigma * self.height_std) + " " + str(self.height_avg + self.sigma * self.height_std) 
