import os, glob, shutil, math
from pathlib import Path
import numpy as np 
from facenet_pytorch import MTCNN, InceptionResnetV1 # Face detector and embedding model
from PIL import Image # Image IO
import cv2 # Video IO
from sklearn.cluster import DBSCAN # Unsupervised clustering of images
import torch 


VIDEO='/Users/Vivaan/Documents/Smart Productivity Tracker(student monitoring)/data/raw/lab_video.mov'
OUT_FACES=Path('data/enroll_auto') # Folder to save the face crops
OUT_FACES.mkdir(parents=True,exist_ok=True)
MODELDIR=Path('models') # Folder to save embeddings
MODELDIR.mkdir(parents=True,exist_ok=True)


ENROLL_SECONDS=40 # Seconds at the start if the video used to enrol faces
FRAME_SKIP=1 # Samples every 3rd frame
DBSCAN_EPS=0.45
DBSCAN_MIN=15


DEVICE='mps' if torch.backends.mps.is_built() else 'cpu' # use apples hardware acceleration if available
# MTCNN is the model used for face detection and it uses the provided device, where each face has the size 160x160
mtcnn=MTCNN(image_size=160,margin=2,device='cpu') 

# InceptionResnetV1 is the neural network model used for face recognition 
# using the pretrained model vggface2 
# .eval() puts the model into evaluation mode which uses fixed data to evaluate the given frames
# .to(DEVICE) moves the training to the given piece of hardware
resnet=InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE) 


# Takes a video frame in RGB format(numpy array) 
def face_extractor_from_tensors(frame_rgb):
    pil=Image.fromarray(frame_rgb) # Converts the numpy image to a pil image as mtcnn expects a pillow image
    boxes,_=mtcnn.detect(pil) # Creates a list of coordinates [x1,x2,y1,y1] for the face bounding boxes
    out=[] # Expty output list
    
    h, w = frame_rgb.shape[:2]
    
    if boxes is None: # If no face detected
        return out
    for box in boxes:
        x1,y1,x2,y2=map(int,box) # Converts the floats to ints for cropping, the coordinates are the face location in the frame
        
        # clip coordinates to image bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        
        # guard against degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue
        # crop as numpy then convert to PIL for mtcnn; also check non-empty
        crop_np = frame_rgb[y1:y2, x1:x2]
        if crop_np.size == 0:
            continue
        
        crop_pil = Image.fromarray(crop_np)
        
        # Try to get aligned face tensor; skip on failure
        try:
            face_tensor = mtcnn(crop_pil)
        except Exception:
            # if MTCNN fails internally (empty tensors), skip this face
            face_tensor = None
        
        if face_tensor is not None: 
            out.append(((x1,y1,x2,y2),face_tensor)) # Appends the bounding box and the aligned face tensor
    return out


def main():
    cap=cv2.VideoCapture(VIDEO)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0 # Gets the FPS of the video or falls back to 30
    max_frames=int(ENROLL_SECONDS*fps)
    frame_i=0
    saved=0
    embeddings=[]
    meta=[]
    
    while frame_i<max_frames:
        ret, frame=cap.read() # read the frame from the video
        if not ret:
            break
        if frame_i%FRAME_SKIP!=0:
            frame_i+=1 # increment the frame until it is a multiple of FRAME_SKIP 
            continue
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # Converts the openCV BGR frame to RGB as Pillow expects RGB
        
        # Calls the function to check if there are any faces present in the frame
        # Returns a list of coordinates and a face tensor if a face is detected, if not the list is empty
        faces=face_extractor_from_tensors(rgb) 
        
        # Iterate over every detected face in the frame
        for (x1,y1,x2,y2), tensor in faces:
            
            # tensor.unsqueeze[0] adds a batch dimension -> shape [1,2,160,160]
            # .to(DEVICE) moves it to mps
            # resnet(...) runs a torch tensor of shape (1,512)
            # .cpu() moves it back to the cpu
            # .numpy() converts to a numpy array of shape (1,512)
            # [0] chooses the 512-D vector for this single face
            with torch.no_grad():
                emb=resnet(tensor.unsqueeze(0).to(DEVICE)).cpu().detach().numpy()[0]
            
            file_name=OUT_FACES/f"face_{saved:06d}.jpg" # builds a path for the saved face
            
            # Saves the cropped PIL image to the path
            crop=rgb[y1:y2,x1:x2]
            
            if crop.size==0: # Ignores all the invalid crops
                continue
            
            Image.fromarray(crop).save(file_name)
            
            embeddings.append(emb)
            
            # Appends the filename and coordinates for each face which links it to the embeddings
            meta.append((str(file_name),x1,y1,x2,y2))
            
            saved+=1
        
        frame_i+=1
        
    cap.release()
    
    # Converts the embeddings list to a numpy array
    embeddings=np.array(embeddings)
    if len(embeddings) == 0:
        raise SystemExit("No faces found in first %d seconds. Increase ENROLL_SECONDS or ensure faces visible."%ENROLL_SECONDS)
    
    # Creates a clustering object
    # eps is how close objects must be to be considered neighbours
    # min_samples is minimum number of embedding points
    # metric is euclidean to consider distance in space
    # .fit runs DBSCAN on the whole (N, 512) embeddings 
    cl = DBSCAN(eps=DBSCAN_EPS,min_samples=DBSCAN_MIN,metric='euclidean').fit(embeddings)
    
    labels=cl.labels_ # gets the labels for each cluster
    unique=sorted(set(labels)-{-1}) # gets unique labels 
    
    print("clusters found:", {int(l): int((labels==l).sum()) for l in unique}, "noise:", int((labels==-1).sum()))
    
    # make a directory for each cluster
    for lab in unique:
        d=OUT_FACES/f"cluster_{lab}"
        d.mkdir(parents=True,exist_ok=True)
    
    # This creates a noise folder for crops belonging to no cluster
    (OUT_FACES/"cluster_-1").mkdir(exist_ok=True)
    
    # Iterate over each cropped face to get folders belonging to faces of each person
    for (file_name,x1,y1,x2,y2), lab in zip(meta,labels):
        dest=OUT_FACES/f"cluster_{lab}"
        if not dest.exists(): 
            dest = OUT_FACES/"cluster_-1"
        shutil.copy(file_name, dest / Path(file_name).name)
    
    known_emb=[]
    known_labels=[]
    
    for lab in unique:
        mask=labels==lab # Mask is a boolean which is true if labels is the same as lab
        
        # Skips empty clusters
        if mask.sum()==0:
            continue 
        
        # Computes the mean embeddings for the clusters
        mean_emb=embeddings[mask].mean(axis=0)
        
        # Appends the embeddings and gives them labels
        known_emb.append(mean_emb)
        known_labels.append(f"student_{lab}")
    
    # Convert to numpy array
    known_emb=np.array(known_emb)
    
    np.save(MODELDIR/"known_embeddings.npy", known_emb)   # save for inference
    np.save(MODELDIR/"known_labels.npy", np.array(known_labels))
    
    print("Saved known embeddings for", len(known_labels), "students to models/")

if __name__=='__main__':
    main()