# USAGE
# python build_dataset.py --dataset /raid/datasets/flowers17/flowers17

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os
import re
import pickle


# this constant indicates how much of the data will be used as train data
TEST_DATA_RATIO = 0.75
AUDIO_DATA_NAME = "audioPower.npy"
PROCESSED_DATA_FOLDER = "processedData/"
dataset_datapath = "./dataset/" #Datapath for the dataset
dataset_raw = "./dataset/raw/"  #Path in which all raw video files are
dataset_config_filename = "config"

try:    #load the configuration file os the dataset
    with open(dataset_datapath+dataset_config_filename,"rb") as fp:
        dataset_config = pickle.load(fp)
except: #new dataset, config file does not exist
	print("The dataset path lacks of a config file. Please run dataset_rawVideoProcess before trying to build")
	exit()

#Creates folder for the processed data
try:
    if not os.path.exists(PROCESSED_DATA_FOLDER):
        os.makedirs(PROCESSED_DATA_FOLDER)
except OSError:
	print ('Error: Creating directory of data')
	exit ()

try:    #Trys loading the config file from the processed data folder
    with open(PROCESSED_DATA_FOLDER+"config","rb") as fp:
        processedData_config = pickle.load(fp)

    dataset_config = [item for item in dataset_config if item not in processedData_config]
    processedData_config += dataset_config
except:     #first time building dataset, config file does not exist
    processedData_config = dataset_config

# If dataset_config is empty, there are no new videos to be built
if not dataset_config:
    print("Built is already up to date")

for path in dataset_config:
	datapath = dataset_datapath + path.replace(".","")

	# grab all image paths and create and orders it
	imagePaths = list(paths.list_images(datapath))
	imagePaths.sort(key=lambda f: int(re.sub('\D', '', f)))

	# split in training data and test data
	i = int(len(imagePaths) * TEST_DATA_RATIO)
	trainPaths = imagePaths[:i]
	testPaths = imagePaths[i:]

	# define the datasets
	datasets = [
		("training", trainPaths, PROCESSED_DATA_FOLDER+"images_training"),
		("testing", testPaths, PROCESSED_DATA_FOLDER+"images_testing")
	]

	#load the audio presure data
	St = np.load(datapath+'/'+AUDIO_DATA_NAME)

	# loop over the data splits

	i = 0

	for (dType, imagePaths, outputPath) in datasets:

		j = 0

		#filedata

	 	# open the output file for writing
		print("[INFO] building '{}' split... for ".format(dType), end = '')
		print(path+'\0')
	
	 	# loop over all input images
		for imagePath in imagePaths:
		# load the input image
			image = cv2.imread(imagePath)

	  		# create a flattened list of pixel values
			idata = [np.array(x,dtype=np.uint8) for x in image.flatten()]

			# extract two labels, one for each channel
			ldata = np.array([St[i,0],St[i,1]],dtype=np.float32)

			# idata = np.append(label,image,axis=0)

			if (j==0):
				outputdata = idata
				outputlabel = ldata
			else:
				outputdata = np.vstack((outputdata,idata))
				outputlabel = np.vstack((outputlabel,ldata))

			j+=1
			i+=1

		try:
			#Before saving to disk, try loading the processed data already saved in disk
			#This way, at the end of the prosesing, we shall simply stack the new data on top of the
			#Already processed one
		
			loadedData = np.load(outputPath+"-img.npy")
			loadedLabel = np.load(outputPath+"-lbl.npy")

			outputdata = np.vstack((outputdata,loadedData))
			outputlabel = np.vstack((outputlabel,loadedLabel))

		except:
			pass

		np.save(outputPath+"-img",outputdata)
		np.save(outputPath+"-lbl",outputlabel)

# Save the information of all videos on file 
with open(PROCESSED_DATA_FOLDER+"config","wb") as fp:
    pickle.dump(processedData_config, fp)

print("training and test data where processed and are ready to be used")