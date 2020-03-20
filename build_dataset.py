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


# this constant indicates how much of the data will be used as train data
TEST_DATA_RATIO = 0.75
AUDIO_DATA_NAME = "audioPower.npy"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path dataset of input images")
args = vars(ap.parse_args())

# grab all image paths and create and orders it
imagePaths = list(paths.list_images(args["dataset"]))
imagePaths.sort(key=lambda f: int(re.sub('\D', '', f)))

# split in training data and test data
i = int(len(imagePaths) * TEST_DATA_RATIO)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# define the datasets
datasets = [
	("training", trainPaths, "images_training"),
	("testing", testPaths, "images_testing")
]

#load the audio presure data
St = np.load(args["dataset"]+'/'+AUDIO_DATA_NAME)

# loop over the data splits

i = 0

for (dType, imagePaths, outputPath) in datasets:
	j = 0

	#filedata

	# open the output CSV file for writing
	print("[INFO] building '{}' split...".format(dType))
	
	# loop over all input images
	for imagePath in imagePaths:
		# load the input image and resize it to 64x64 pixels
		image = cv2.imread(imagePath)

 		# create a flattened list of pixel values
		idata = [np.array(x,dtype=np.uint8) for x in image.flatten()]

		# extract the label from the St as the média between channel right and left
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
	np.save(outputPath+"-img",outputdata)
	np.save(outputPath+"-lbl",outputlabel)

print("training and test data where processed and are ready to be used")