import cv2
import numpy as np
import os

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None
if not is_tool("ffmpeg"):
    print("Please install ffmpeg for the video processing")
    exit()
    pass
if not is_tool("ffprobe"):
    print("Please install ffmpeg for the video processing")
    exit()
    pass

# Resizes the video for the 240x240 screen resolution
output_resolution = "240x240"   #This string determines the output resolution of the resized video
frameJump = 1                   #This number determines what is the next frame in the amostration process

video_name = input("Please write the video name (w/ extension): ")

try:
    if not os.path.exists('dataset/images'):
        os.makedirs('dataset/images')
except OSError:
    print ('Error: Creating directory of data')

os_command = "ffmpeg -i "+video_name+" -s "+output_resolution+" -c:a copy dataset/resized-"+video_name
os.system(os_command)

# Extact frames from video
cap = cv2.VideoCapture("dataset/resized-"+video_name)
currentFrame = 0
print("Extracting the frames from the video...")
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Saves image of the current frame in jpg file
        name = './dataset/images/' + str(currentFrame) + '.jpg'
        #print ('Creating...' + name)
        cv2.imwrite(name,frame)
        # To stop duplicate images
        currentFrame += frameJump
    else: 
        print(str(currentFrame)+" frames were extracted")
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# Informations from the video
print("Informations from the resized video:")
os_command = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration,bit_rate -of default=noprint_wrappers=1 dataset/resized-"+video_name
os.system(os_command)
bit_rate = input("Please write the bitrate: ")
#Extract the audio file from video
os_command = "ffmpeg -i "+video_name+" -f wav -ar 48000 -ab "+bit_rate+" -vn dataset/audio-"+video_name[:len(video_name)-4]+".wav"
os.system(os_command)