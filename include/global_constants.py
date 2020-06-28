#Dataset creation constants
# ---------------------------------------------------- #
# this constant indicates how much of the data will be used as train data
TEST_DATA_RATIO = 0.75
AUDIO_DATA_NAME = "audioPower.npy"
PROCESSED_DATA_FOLDER = "processedData/"
dataset_datapath = "./dataset/" #Datapath for the dataset
dataset_train_datapath = "./dataset/train/"  #Path for train dataset
dataset_test_datapath = "./dataset/test/"    #Path for test dataset   
dataset_raw = "./dataset/raw/"  #Path in which all raw video files are
dataset_config_filename = "config"
dataset_config_train_filename = "config-train"
dataset_config_test_filename = "config-test"
output_resolution = "240x240"   #This string determines the output resolution of the resized video
frameJump = 10                  #This number determines what is the next frame in the amostration process
dataset_test_raw = "./dataset/raw/test/"    #Path in which all raw video files are
dataset_train_raw = "./dataset/raw/train/"  #Path in which all raw video files are

PROCESSED_DATA_FOLDER = "processedData/"
image_shape = (240,240,3)
timeSteps = 100


#---------------------------------#