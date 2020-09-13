try:
    import cv2
    import numpy as np
    import os
    from os import listdir
    from os.path import isfile, join
    import scipy.io.wavfile
    from math import floor
    import re
    import pickle
    from imutils import paths
    from include.telegram_logger import *
    from include.globals_and_functions import *

    #check if needed programs are installed
    print_info("Checking necessary tools")
    #if not is_tool("ffmpeg"):
    #    print_error("Please install ffmpeg for the video processing")
    #    exit()

    if not is_tool("ffprobe"):
        print_error("Please install ffprobe for the audio extraction")
        exit()

    print_info("Reading filepath off all videos")
    #Get the path of ALL videos in the raw folder. All videos are stored in the dataset/raw path
    dataset_raw_datapath = [f for f in listdir(CONST_STR_DATASET_RAW_DATAPATH) if isfile(join(CONST_STR_DATASET_RAW_DATAPATH, f))]

    # # First, check what videos were already processed. To do this, we load the "config" file in the dataset directory,
    # # and check for which videos are "new" in the raw folder. We only need to process those

    try:    #load the config file
        with open(CONST_STR_DATASET_DATAPATH+CONST_STR_DATASET_CONFIG_FILENAME, "rb") as fp:
            dataset_config_file = pickle.load(fp)

        #   here we see what videos were already processed and which were not
        unprocessed_videos = [item for item in dataset_raw_datapath if item not in dataset_config_file]

        #   ..and we add the unprocessed videos to the dataset_config_file (because they will be processed at the end of the script)
        dataset_config_file += unprocessed_videos

    except:     #   new dataset, config file does not exist
        print_warning("Could not find dataset config file. Creating one")
        #   ... there is no config file
        dataset_config_file = dataset_raw_datapath
        #   ... all data is unprocessed
        unprocessed_videos = dataset_raw_datapath

    if not unprocessed_videos:
        print_info("There are no new videos to be processed in the dataset")
        telegramSendMessage('Dataset is already up to date')
        exit()
    else:
        print_info("The following new videos were added to the dataset")
        for video_name in unprocessed_videos:
            print('\t'+video_name)

        telegramSendMessage(str(len(unprocessed_videos))+" new videos were added to the dataset")

    print_info("Starting dataset processing")
    telegramSendMessage("Starting dataset processing")

    #extract frames and sound for each video in train data!
    for video_name in unprocessed_videos:
        #   Datapath of raw video (where the raw video is)
        video_raw_datapath = CONST_STR_DATASET_RAW_DATAPATH + video_name
        #   Datapath of the processed information of video
        video_datapath = CONST_STR_DATASET_DATAPATH + video_name.replace(".", "")

        #   Make directory to hold all extracted information from the video
        try:
            if not os.path.exists(video_datapath):
                os.makedirs(video_datapath)
        except OSError:
            print_error("Could not make directory for video '"+str(video_name)+"'")
            telegramSendMessage('Error: Creating directory')
            exit()

        #   ------------------- Extraction of frames from video

        #   Resize the raw video to the desired resolution
        #os_command = "ffmpeg -i "+video_raw_datapath+" -s "+CONST_STR_DATASET_OUTPUT_RESOLUTION+" -c:a copy "+CONST_STR_DATASET_DATAPATH+"resized-"+video_name
        #os.system(os_command)

        # Extact frames from resized video
        #cap = cv2.VideoCapture(CONST_STR_DATASET_DATAPATH+"resized-"+video_name)
        #   I stoped using ffmpeg after noticing undesired compression artifacts 
        #   on the output frames. Using cv2.resize() is a much better alternative
        cap = cv2.VideoCapture(CONST_STR_DATASET_RAW_DATAPATH+video_name)

        currentFrame = 0    # This variable counts the frame in the extracted video
        videoFrame = 0      # This variable counts the actual frame in the raw video
                            # I use those 2 variables so that I can change the fps of extraction
                            # by decimation

        total_number_of_video_frames = 0    # Total number of extracted video frames

        print_info("Extracting frames from video "+video_name)
        telegramSendMessage("Extracting frames from video "+video_name)

        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if videoFrame%CONST_INT_DATASET_DECIMATION_FACTOR == 0:
                if ret:
                    # Saves image of the current frame in png file
                    frame_name = video_datapath+'/'+ str(currentFrame) + '.png'
                    #   Frame resize
                    frame = cv2.resize(frame, CONST_VEC_DATASET_OUTPUT_RESOLUTION)
                    cv2.imwrite(frame_name, frame)
                    # To stop duplicate images
                    currentFrame += 1
                else:
                    break

            videoFrame += 1
            total_number_of_video_frames = currentFrame

        #   The next section is maintained only for compatibility porposues. I previously
        #   had made the script force the number of images extracted to be a multiple of
        #   27 (for a easier handling on the training process). Note that this is NOT
        #   required anymore, and the script should only do that with programers consent.
        #   When building a dataset from scratch, it can be disable.
        if False:
            print_warning("Forcing the number of extracted frames to be multiple of 27. You should disable it if building a dataset from scratch")

            number_of_extra_extracted_frames = total_number_of_video_frames % 27
            
            #   We here delete those 'extra' video frames
            for i in range(total_number_of_video_frames-number_of_extra_extracted_frames, total_number_of_video_frames, 1):
                extra_frame = video_datapath+'/'+ str(i) + '.png'
                os.system('rm -f '+extra_frame)
            total_number_of_video_frames = total_number_of_video_frames - number_of_extra_extracted_frames
            print_warning(str(number_of_extra_extracted_frames)+" were deleted. Total number of frames: "+str(total_number_of_video_frames))
        # Save the number of frames in this video on the frames folder
        with open(video_datapath+'/'+CONST_STR_DATASET_NMB_OF_FRAMES_FILENAME, "wb") as fp:
            pickle.dump(total_number_of_video_frames, fp)

        print_info(str(total_number_of_video_frames)+" frames were extracted from video "+video_name)
        # When everything done, release the capture
        cap.release()

        #   ------------------- Extraction of audio from video
        #   I think I will not have to reprocess this, so I will simply not execute this on server
        #   (Because it is very time consuming)
        if True:
            print_info("Extracting audio information from video "+video_name)
            telegramSendMessage("Extracting audio information from video "+video_name)

            #   Execute command to extract only audio from video
            audio_filepath = video_datapath+CONS_STR_DATASET_AUDIOFILE_FILENAME
            os_command = "ffmpeg -i "+video_raw_datapath+" "+audio_filepath
            os.system(os_command)

            #   Get samples from audio file generated
            print_info("Reading audio file ...")
            FSample, samples = scipy.io.wavfile.read(audio_filepath)
            samples = np.array(samples)
            original_audio = samples

            #   ------------------- Calculate the total power for each frame

            M = floor(samples.shape[0]/total_number_of_video_frames)    #Number of Samples used for each frame calculatiom
            St = np.zeros((total_number_of_video_frames, 2))             #Array of audio power in each frame

            print_info("Calculating audio power for each frame ...")
            telegramSendMessage("Calculating audio power for each frame ...")

            #   Square and divide all samples by M
            samples = np.square(samples, dtype='int64')
            samples = np.divide(samples, M)

            #   Do the partial sum of everything
            for i in range(0, total_number_of_video_frames):
                St[i] = np.sum(samples[i*M:(i+1)*M], axis=0)

            #   Clip the zeros to a minor value, and log everything
            St = np.clip(St, 1e-12, None)
            St = np.log(St)

            """
            Previous algorithm. This was very time consuming to run

            for i in range(0, total_number_of_video_frames):
                partialSumRight = 0
                partialSumLeft = 0
                for j in range(0, M):
                    partialSumLeft += (1/M)*((samples[j+i*M, 0])**2)
                    partialSumRight += (1/M)*((samples[j+i*M, 1])**2)

                if partialSumLeft > 0:
                    St[i, 0] = log(partialSumLeft)
                if partialSumRight > 0:
                    St[i, 1] = log(partialSumRight)
            """
            # save numpy array as .npy file
            np.save(video_datapath+CONS_STR_DATASET_AUDIODATA_FILENAME, St)

    #   ------------------- dataset_build

    for video_name in unprocessed_videos:
        print_info("Stacking images for video "+video_name)
        telegramSendMessage("Stacking images for video "+video_name)

        first_frame = True

        #   Datapath of raw video (where the raw video is)
        video_raw_datapath = CONST_STR_DATASET_RAW_DATAPATH + video_name
        #   Datapath of the processed information of video
        video_datapath = CONST_STR_DATASET_DATAPATH + video_name.replace(".", "")

        # grab all image paths and order it correctly
        frame_datapaths = list(paths.list_images(video_datapath))
        frame_datapaths.sort(key=lambda f: int(re.sub('\D', '', f)))

        for frame_path in frame_datapaths:
            #   Read image frame
            frame = cv2.imread(frame_path)

            # create a flattened list of pixel values
            frame_data = [np.array(x, dtype=np.uint8) for x in frame.flatten()]

            # We then stack all frames on top of each other
            # Image stacking is now what consumes the most time in processing
            if first_frame:
                stacked_frames_array = frame_data
                first_frame = False
            else:
                stacked_frames_array = np.vstack((stacked_frames_array, frame_data))

        #   Save the stacked frames numpy to the corresponding video folder
        print_info("Saving stacked frames data to "+video_datapath+CONS_STR_DATASET_STACKED_FRAMES_FILENAME)
        np.save(video_datapath+CONS_STR_DATASET_STACKED_FRAMES_FILENAME, stacked_frames_array)

    #Last step: Calculate mean and std for each video in dataset. Save this information in disk
    for video_name in unprocessed_videos:
        print_info("Loading numpy dataset for mean and std calculation")

        #   Load numpy array
        video_datapath = CONST_STR_DATASET_DATAPATH + video_name.replace(".", "")

        video_data = np.load(video_datapath+CONS_STR_DATASET_STACKED_FRAMES_FILENAME)
        video_data = np.reshape(video_data, (video_data.shape[0],)+CONST_VEC_DATASET_OUTPUT_IMAGE_SHAPE)

        #   Calculate mean and std of video
        mean = np.mean(video_data, axis=(0,1,2)).astype(float)
        std = np.std(video_data, axis=(0,1,2)).astype(float)

        statistics = [mean, std]

        print("mean: "+'\t'+str(statistics[0]))
        print("std: " +'\t'+str(statistics[1]))

        #   Save it to a file
        with open(video_datapath+CONS_STR_DATASET_STATISTICS_FILENAME, "wb") as fp:
            pickle.dump(statistics, fp)

    # Save the information of all videos on file
    with open(CONST_STR_DATASET_DATAPATH+CONST_STR_DATASET_CONFIG_FILENAME, "wb") as fp:
        pickle.dump(dataset_config_file, fp)

    print_info("Script ended successfully")
    telegramSendMessage("Script ended successfully")

except Exception as e:
    print_error('An error has occurred')
    print_error(str(e))
    telegramSendMessage('[ERROR]: An error has occurred')
    telegramSendMessage(str(e))
