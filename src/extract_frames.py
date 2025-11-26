import cv2
import os 
from pathlib import Path


VIDEO='/Users/Vivaan/Documents/Smart Productivity Tracker(student monitoring)/data/raw/lab_video.mov' # Path for the video from which frames are to be extracted
OUT_DIR=Path("data/frames") # Path where the frames are to be saved
OUT_DIR.mkdir(parents=True,exist_ok=True) # Creates the folder fi it doesnt exist


cap=cv2.VideoCapture(VIDEO) # cv2.videocapture opens the video
if not cap.isOpened: # If the path is wrong and error is rasied
    raise SystemError(f"Could not find the path to the video: {VIDEO}")


FRAME_SKIP=1 # Chooses how many frames to skip, 1 means it captures every frame


frame_index=0 # Counter for every frame read in the video
saved=0 # Counter for how many frames are actually saved


while True:
    ret,frame=cap.read() # .read() gives 2 values, a boolean to indicate whether the video has ended or no and the actual frame itself
    if not ret:
        break
    
    if frame_index%FRAME_SKIP==0: # This allwos you to skip frames if your FRAME_SKIP is 5 or 10
        out_path=OUT_DIR/f"frame_{saved:06d}.jpg" # Creates a unique file name for every frame
        cv2.imwrite(str(out_path),frame) # Saves the image to the given path as a JPEG
        saved+=1
        
        if saved%200==0: # Prints progress every 200 frames
            print(f"Saved {saved} frames")
    
    frame_index+=1 


cap.release() # Releases the video file
print(f"Total frames saved: {saved}")