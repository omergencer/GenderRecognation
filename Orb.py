import cv2,re,os,csv
from skimage.feature import ORB
from skimage.morphology import skeletonize
from skimage import transform as tf
import numpy as np

directory = os.getcwd()+"\Real"
with open('data_orb.csv', mode='w', newline='') as d_file:#creates a csv file to save the data
    d_writer = csv.writer(d_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for file in os.listdir(directory):#size of dataset
        name = re.search("^(\d+)__", file).group(1)#this section is used for matching image names and categorising them
        gender = re.search("__(\w)", file).group(1)
        LoR = re.search("[R|L]", file).group(0)
        finger = re.search("_([a-z]\D{3,5})_", file).group(1)
        img = cv2.imread(os.path.join(directory, file),0)
        print(file)
        
        ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)#tresholding
    
        thresh[thresh == 255] = 1#skeleton
        skeleton = skeletonize(thresh)

        descriptor_extractor = ORB(n_keypoints=10)#detecting key points
        descriptor_extractor.detect_and_extract(skeleton)
        keypoints = descriptor_extractor.keypoints
        data = keypoints.tolist()
        flat_list = [item for sublist in data for item in sublist]
        if len(flat_list) < 20:#if key points are less than specified number, add zeroes till it is enough
            flat_list = flat_list+(20-len(flat_list))*[0]
        
        d_writer.writerow([name]+[gender]+flat_list)#writes data with id and gender
        
        




