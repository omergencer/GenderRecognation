import cv2,re,os,csv
from skimage.morphology import skeletonize
from skimage import transform as tf
import numpy as np
#from skimage import img_as_ubyte
#from skimage import data, io
#from matplotlib import pyplot as plt
directory = os.getcwd()+"\Real"
with open('data_kaze.csv', mode='w', newline='') as d_file:#creates a csv file to save the data
    d_writer = csv.writer(d_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for file in os.listdir(directory):#size of dataset
        name = re.search("^(\d+)__", file).group(1)#this section is used for matching image names and categorising them
        gender = re.search("__(\w)", file).group(1)
        LoR = re.search("[R|L]", file).group(0)
        finger = re.search("_([a-z]\D{3,5})_", file).group(1)
        img = cv2.imread(os.path.join(directory, file),0)
        #print(file)

       
        ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)#tresholding
      
    
        thresh[thresh == 255] = 1
        skeleton = skeletonize(thresh)#skeleton
       
      
        
        #io.imshow(skeleton)
        #plt.show()
      
       

        alg = cv2.KAZE_create()#Kaze feature detection
        kps = alg.detect(img)
        data = [x.response for x in kps][:7]
        if len(data) < 7:#if key points are less than specified number, add zeroes till it is enough
            data = data+(7-len(data))*[0]
        
        d_writer.writerow([name]+[gender]+data)#writes data with id and gender
