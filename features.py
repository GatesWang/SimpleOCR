import numpy as np
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, morphology
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from get_stats import Stats

class RegionFeatures:
    def __init__(self, img_binary, minr, minc, maxr, maxc):
        self.img_binary = img_binary
        self.minr = minr
        self.minc = minc
        self.maxr = maxr
        self.maxc = maxc
        
    #given image and bbox get longest horizontal divided by width
    def get_longest_horizontal(self):
        max_count = 0
        count = 0
        width = self.maxc - self.minc
        height = self.maxr - self.minr
        for i in range(0,height-1):
            for j in range(0, width-1):
                if self.img_binary[self.minr + i][self.minc + j]:
                    count +=1
                else:
                    if count > max_count:
                        max_count = count
                    count = 0
            count = 0
                    
        return float(max_count)/(width)

    #given image and bbox get longest vertical divided by height
    def get_longest_vertical(self):
        max_count = 0
        count = 0
        width = self.maxc - self.minc
        height = self.maxr - self.minr
        for j in range(0,width-1):
            for i in range(0, height-1):
                if self.img_binary[self.minr + i][self.minc + j]:
                    count +=1
                else:
                    if count > max_count:
                        max_count = count
                    count = 0
            count = 0
                    
        return float(max_count)/(height)
    
    #find the percentage filled
    def get_percentage_filled(self):
        count = 0
        width = self.maxc - self.minc
        height = self.maxr - self.minr
        for j in range(0,width-1):
            for i in range(0, height-1):
                if self.img_binary[self.minr + i][self.minc + j]:
                    count +=1
                    
        return float(count)/(height*width)
    
    #find the center coordinates
    def get_center(self):
        x_center = 0
        y_center = 0
        count = 0
        width = self.maxc - self.minc
        height = self.maxr - self.minr
        for i in range(0, height-1):
            for j in range(0, width-1):
                if self.img_binary[self.minr + i][self.minc + j]:
                    y_center += i
                    x_center += j
                    count +=1
        x_center = x_center/count
        y_center = y_center/count
        
        x_center = float(x_center)/width
        y_center = float(y_center)/height
        return x_center, y_center

    #find the x projections
    def get_x_projections(self):
        count = 0
        width = self.maxc - self.minc
        height = self.maxr - self.minr
        x_intervals = np.linspace(0,width,5+1)
        interval_counts = [0]*(len(x_intervals)-1)
        #print x_intervals
        interval_index = 0
        for i in range(0,width-1):
            for j in range(0, height-1):
                count+=1
                if self.img_binary[self.minr + j][self.minc + i]:
                    interval_counts[interval_index]+=1
                if i > x_intervals[interval_index+1]:
                    interval_index+=1
                    #print i, x_intervals[interval_index]

        return [float(i_count)/count for i_count in interval_counts]
    
    #find the y projections
    def get_y_projections(self):
        count = 0
        width = self.maxc - self.minc
        height = self.maxr - self.minr
        y_intervals = np.linspace(0,height,5+1)
        interval_counts = [0]*(len(y_intervals)-1)
        #print y_intervals 
        interval_index = 0
        for j in range(0, height-1):
            for i in range(0,width-1):
                count+=1
                if self.img_binary[self.minr + j][self.minc + i]:
                    interval_counts[interval_index]+=1
                if j > y_intervals[interval_index+1]:
                    interval_index+=1
                    #print j, y_intervals[interval_index]
        
        return [float(i_count)/count for i_count in interval_counts]
    
