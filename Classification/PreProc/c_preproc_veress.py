'''
Justin Reynolds
Preprocessing the veress classification data.
- Import the images and labels pertaining to the layers fat, muscle, abdominal space, and intestine (skip skin). 
- Confirm the integrity of imports and mappings. 
- Export the data for modeling. 
'''
import os, sys, time
import re
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

if __name__ == '__main__':
    # import string
    path = "/Users/justinreynolds/Desktop/21summer/veress/classification/data/classification_images/"

    # export strings
    export_path = "/Users/justinreynolds/Desktop/21summer/veress/classification/data/preprocessed_data/"
    verdir =       'v20210803/'
    fn_images =    'c_images_1D_20210803.npy'
    fn_labels =    'c_labels_20210803.npy'
    fn_filenames = 'c_filenames_20210803.npy'
    fn_vid =       'c_vid_20210803.npy'

    # input
    alldirs = os.listdir( path )
    dirs=[]
    start_time=time.time()
    for d in alldirs:
        if re.match(r'V\d', d) and not re.match(r'V\d_*.zip', d):
            dirs.append(d)

    # get full path to directories with images
    fullpath = [] 
    count=0
    print("--- Directories to images ---")
    for item in dirs:
        _temp_dir = path + item
        if os.path.isdir(_temp_dir) and re.match(r'V\d', item):
            temp_dir = path + item
            subdirs = os.listdir(temp_dir)
            print(item)
            for subitem in subdirs:
                if re.match(r'V\d_', subitem) and not re.match(r'V\d_skin', subitem):
                    fullpath.append(temp_dir + "/" +subitem)
                    count+=1
                    print("\t", subitem)
                    
    # input the images                
    print("\n------- Input images --------")
    images_32k = []
    filenames_32k = []
    labels_32k = []
    vid_32k = []
    n_layers=4 # CHANGE TO 4 WHEN CHEN SENDS 'ABD_SPACE' IMAGES
    n_images_per_subject_per_layer=1000
    n=int(n_images_per_subject_per_layer*len(fullpath))
    count = 0

    # read and store the classification images
    for item in fullpath:
        file_dir_temp = os.listdir(item)
        #if not re.match(r'')
        for item_list in file_dir_temp:
            path_img = item + "/" + item_list # get the full path to the image
            img = mpimg.imread(path_img, format = "png") # read 3-channel image with matplotlib
            res=re.search(r'(?<=V)[0-9]+', item_list) # re filename for vid
            # extract labels from filename and store
            #if "skin" in item_list:
            #    labels_32k.append("skin")
            if "fat" in item_list:
                labels_32k.append("fat") # store label
                images_32k.append(img) # store image
                filenames_32k.append(item_list) # store filename
                vid_32k.append(int(res.group(0))) # store vid
            elif "muscle" in item_list:
                labels_32k.append("muscle") # store label
                images_32k.append(img) # store image
                filenames_32k.append(item_list) # store filename
                vid_32k.append(int(res.group(0))) # store vid
            elif "abdominal_space" in item_list:
                labels_32k.append("abdominal space") # store label
                images_32k.append(img) # store image
                filenames_32k.append(item_list) # store filename
                vid_32k.append(int(res.group(0))) # store vid
            elif "intestine" in item_list:
                labels_32k.append("intestine") # store label
                images_32k.append(img) # store image
                filenames_32k.append(item_list) # store filename
                vid_32k.append(int(res.group(0))) # store vid
            count+=1 
            percent_complete=int(100*(count/n)) # calculate progress
            print(str(percent_complete)+"%", end='\r') # display progress
            sys.stdout.flush() 
    mytime=time.time()-start_time

    label_count={"fat":0, "muscle":0, "abdominal space":0, "intestine":0}
    for i in labels_32k:
        if i == "fat":
            label_count["fat"]+=1
        elif i == "muscle":
            label_count["muscle"]+=1
        elif i == "abdominal space":
            label_count["abdominal space"]+=1
        elif i == "intestine":
            label_count["intestine"]+=1

    print("Done... "+str(len(images_32k))+ " total images processed")
    print("time to input images is %.2f mins" % (mytime/60.0))
    print("-----------------------------")
    print("unique subjects: "+str(np.unique(vid_32k)))
    print("ground truth breakdown:")
    print(label_count)
    print("shape of images before removing last channel: "+str(images_32k[3].shape))
    images_1D_32k=[i[:,:,0] for i in images_32k]
    print("shape of images after removing last channel: "+str(images_1D_32k[3].shape))

    # export
    with open(export_path+verdir+fn_images, 'wb') as f:
        np.save(f, np.array(images_1D_32k))
    with open(export_path+verdir+fn_labels, 'wb') as f:
        np.save(f, np.array(labels_32k))
    with open(export_path+verdir+fn_filenames, 'wb') as f:
        np.save(f, np.array(filenames_32k))
    with open(export_path+verdir+fn_vid, 'wb') as f:
        np.save(f, np.array(vid_32k))
