# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os
import re
import pickle
from include.telegram_logger import *
from include.global_constants import *

telegramSendMessage('dataset_build started processing')

try:    #load the configuration file os the dataset
    with open(dataset_datapath+dataset_config_train_filename,"rb") as fp:
        dataset_config_train = pickle.load(fp)
except: #new dataset, config file does not exist
	print("The dataset path lacks of a config-train file. Please run dataset_rawVideoProcess before trying to build")
	exit()
try:    #load the configuration file os the dataset
    with open(dataset_datapath+dataset_config_test_filename,"rb") as fp:
        dataset_config_test = pickle.load(fp)
except: #new dataset, config file does not exist
	print("The dataset path lacks of a config-test file. Please run dataset_rawVideoProcess before trying to build")
	exit()

#Creates folder for the processed data
try:
    if not os.path.exists(PROCESSED_DATA_FOLDER):
        os.makedirs(PROCESSED_DATA_FOLDER)
except OSError:
	print ('Error: Creating directory for building the dataset')
	exit ()

try:    #Trys loading the config file from the processed data folder
    with open(PROCESSED_DATA_FOLDER+"config-train","rb") as fp:
        processedData_config_train = pickle.load(fp)

    dataset_config_train = [item for item in dataset_config_train if item not in processedData_config_train]
    processedData_config_train += dataset_config_train
except:     #first time building dataset, config file does not exist
    processedData_config_train = dataset_config_train

try:    #Trys loading the config file from the processed data folder
    with open(PROCESSED_DATA_FOLDER+"config-test","rb") as fp:
        processedData_config_test = pickle.load(fp)

    dataset_config_test = [item for item in dataset_config_test if item not in processedData_config_test]
    processedData_config_test += dataset_config_test
except:     #first time building dataset, config file does not exist
    processedData_config_test = dataset_config_test

# If dataset_config is empty, there are no new videos to be built
if not dataset_config_train:
    print("Train built is already up to date")
if not dataset_config_test:
    print("Test built is already up to date")

for path in dataset_config_train:
	datapath = dataset_train_datapath + path.replace(".","")

	# grab all image paths and create and orders it
	imagePaths = list(paths.list_images(datapath))
	imagePaths.sort(key=lambda f: int(re.sub('\D', '', f)))

	# split in training data and test data
	# i = int(len(imagePaths) * TEST_DATA_RATIO)
	trainPaths = imagePaths[:]
	# testPaths = imagePaths[i:]

	# define the datasets
	datasets = [
		("training", trainPaths, PROCESSED_DATA_FOLDER+"images_training")
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

for path in dataset_config_test:
	datapath = dataset_test_datapath + path.replace(".","")

	# grab all image paths and create and orders it
	imagePaths = list(paths.list_images(datapath))
	imagePaths.sort(key=lambda f: int(re.sub('\D', '', f)))

	# split in training data and test data
	# i = int(len(imagePaths) * TEST_DATA_RATIO)
	testPaths = imagePaths[:]
	# testPaths = imagePaths[i:]

	# define the datasets
	datasets = [
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
with open(PROCESSED_DATA_FOLDER+"config-train","wb") as fp:
    pickle.dump(processedData_config_train, fp)
with open(PROCESSED_DATA_FOLDER+"config-test","wb") as fp:
    pickle.dump(processedData_config_test, fp)

print("training and test data where processed and are ready to be used")

telegramSendMessage('dataset_build ended successfully')