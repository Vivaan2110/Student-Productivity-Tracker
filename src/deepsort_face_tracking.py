import cv2
import time
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from numpy.linalg import norm
from facenet_pytorch import MTCNN,InceptionResnetV1
from deep_sort_realtime.deepsort_tracker import DeepSort
import csv
from ultralytics import YOLO

torch.set_num_threads(8)
torch.set_num_interop_threads(8)

start=time.time()

VIDEO='/Users/Vivaan/Documents/Smart Productivity Tracker(student monitoring)/data/raw/lab_video.mov'
OUT_DIR=Path("output")
OUT_DIR.mkdir(parents=True,exist_ok=True)
OUT_VID=OUT_DIR/"deepsort_face_out.avi"
LOG_CSV=OUT_DIR/"tracking_log.csv" 


MODEL_DIR=Path('models') # Path to the model directory
KNOWN_EMB_F=MODEL_DIR/"known_embeddings.npy" # Path to the Embeddings file
KNOWN_LABELS_F=MODEL_DIR/"known_labels.npy" # Path to the label file
if not KNOWN_EMB_F.exists(): # Causes the program to exit if the embedding file isnt found
    raise SystemExit("Run enrollment first and ensure models exists")


DEVICE='mps' if torch.backends.mps.is_built() else "cpu"
print(f"Device: {DEVICE}")


resnet=InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

YOLO_DEVICE="mps" if torch.backends.mps.is_built() else "cpu"
yolo_model=YOLO("yolov8n.pt")
yolo_pose_model=YOLO("yolov8n-pose.pt")

def action_classift(kpts):
    
    # Keypoints: 
                
    # 0 nose
    # 1 left eye
    # 2 right eye
    # 5 left shoulder
    # 6 right shoulder
    # 7 left elbow
    # 8 right elbow
    # 9 left wrist
    # 10 right wrist
    
    if kpts is None:
        return "Not found"
    
    nose_y = kpts[0][1]
    shoulder_y = (kpts[5][1] + kpts[6][1]) / 2

    if nose_y > shoulder_y + 30:
        return "sleeping"
                
    if kpts[9][1] > shoulder_y and kpts[10][1] > shoulder_y:
        return "writing"  
                
    nose_x, nose_y = kpts[0]

    lw_x, lw_y = kpts[9]
    rw_x, rw_y = kpts[10]

    if abs(lw_y - nose_y) < 40 or abs(rw_y - nose_y) < 40:
        return "phone" 
    
    return "attentive"   

mtcnn=MTCNN(image_size=160,margin=20,device="cpu")

# Loading the embedding and label files  
known_emb=np.load(KNOWN_EMB_F) # Shape (K,512)
known_labels=np.load(KNOWN_LABELS_F,allow_pickle=True)


# Creates a DeepSort tracker
# max_age determines how many frames to keep the tracker active without new detections
tracker=DeepSort(max_age=60,n_init=3, max_iou_distance=0.7, max_cosine_distance=0.3)

DETECT_EVERY = 1
FRAME_SKIP = 2
POSE_FRAME = 15
EMB_MATCH_THRESHOLD = 0.40
BOX_PAD = 1.3 # Expand face box to include head and shoulders
LOCK_THRESHOLD = 0.65
DOWNSACLE = 0.6 # Downscaling the resolution for faster processing

# Cosine similarity=normalised dot product
def cos_sim(a,b):
    a=a/(norm(a)+1e-8)
    b=b/(norm(b)+1e-8)
    return float(np.dot(a,b))

# Input = crop_rgb: HxWx3 numpy array(RGB)
# Output = 512-d embedding vector
def get_embeddings_from_crop(crop_rgb):
    
    pil=Image.fromarray(crop_rgb)

    try:
        face_t=mtcnn(pil)
    except Exception:
        face_t=None
    
    if face_t is None:  
        return None    
    
    if isinstance(face_t, torch.Tensor) and face_t.ndim == 4:
        face_t = face_t[0] 
        
    with torch.no_grad():
        emb=resnet(face_t.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    return emb 

    '''
    # Create a fallback deterministic appearance vector from color
    small=cv2.resize(crop_rgb,(160,160))
    mean=small.mean(axis=(0,1)).astype(np.float32)
    vec=mean/(np.linalg.norm(mean)+1e-8)
    reps=int(np.ceil(512/vec.size))
    emb=np.tile(vec,reps)[:512]
    return emb
'''

track_label_map={} # Dictionary storing track id->assigned label
track_best_score={} # Dictionary storing track id->best similarity score
track_action_map={}  # Dictionary storing track id->current action


cap=cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cant open the video: {VIDEO}")

fps=cap.get(cv2.CAP_PROP_FPS) or 25.0

MAX_FRAMES = int(10 * 60 * fps)

orig_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get width and height for the bounding boxes
w=int(orig_w*DOWNSACLE)
h=int(orig_h*DOWNSACLE)

print("orig:", orig_w, orig_h)
print("scaled:", w, h)
print("fps raw:", fps, "fps used:", fps/FRAME_SKIP)

# Create a video writer to produce a preview file
# Takes in the filename, a fourcc for the format, fps(fps/FRAME_SKIP) if FRAME_SKIP wasnt 1, frameSize
# A fourcc is a fourCC(four character code) to choose a format, mp4v chosen here
vw=cv2.VideoWriter(str(OUT_VID),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    fps/FRAME_SKIP,
                    (w,h))

if not vw.isOpened():
    raise SystemExit("VideoWriter failed to open. Codec/size issue.")

frame_i=0
processed_i=0
print("Start tracking...")

# Prepare the csv for logging(timestamp, frame, track_id, label, bbox)
log_f=open(LOG_CSV,'w',newline="")
csvw=csv.writer(log_f)
csvw.writerow(["timestamp","frame","track_id","label","x1","y1","x2","y2"])

while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame_i+=1
    
    if frame_i%FRAME_SKIP!=0:
        continue
    
    if frame_i >= MAX_FRAMES:
        break
    
    processed_i+=1
    
    # Converts the frame from openCV's BGR to RGB
    frame_small=cv2.resize(frame,(w,h))
    rgb=cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    
    # Will hold tuples: ((x1,y1,x2,y2), score, embedding)
    detections=[]
    
    # Run face detection periodically to produce fresh detections
    if processed_i%DETECT_EVERY==0:
        
        results=yolo_model(
            frame_small,
            imgsz=480,
            conf=0.6,
            iou=0.5,
            max_det=20,
            device=YOLO_DEVICE,
            verbose=False
        )
        
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls=int(b.cls[0])
                if cls!=0:
                    continue
                conf=float(b.conf[0])
                x1,y1,x2,y2=map(int,b.xyxy[0]) # typecast float to int
                
                
                # Expand the face box to capture more of the head/shoulders
                fw=x2-x1
                fh=y2-y1
                
                # Center of the box
                cx=x1+fw/2
                cy=y1+fh/2
                
                # Padded height and width
                pw=int(fw*BOX_PAD)
                ph=int(fh*BOX_PAD)
                
                # Clamp to frame bounds to avoid invalid coordinates
                x1p=max(0,int(cx-pw/2))
                y1p=max(0,int(cy-ph/2))
                x2p=min(w-1,int(cx+pw/2))
                y2p=min(h-1,int(cy+ph/2))
                
                # Skip the degenerate boxes
                if x2p<=x1p or y2p<=y1p:
                    continue
                
                detection_crop=rgb[y1p:y2p,x1p:x2p]
                if detection_crop.size==0: # Skip any empty crops
                    continue

                # Get the embeddings for this crop
                detection_emb=get_embeddings_from_crop(detection_crop)
                
                if detection_emb is None:
                    detection_emb=np.zeros(512)
                
                w_box=x2p-x1p
                h_box=y2p-y1p
                
                detections.append(([x1p,y1p,w_box,h_box],conf,detection_emb))
                
                #print(f"[DEBUG] frame {frame_i}, processed {processed_i}, YOLO dets={len(detections)}")
    
    # Pass detections to the DeepSort tracker which will associate detections with existing tracks
    # Create new tracks and confirm track states(confirmed, lost, deleted)
    tracks=tracker.update_tracks(detections,frame=frame_small)
    
    confirmed_ids = [tr.track_id for tr in tracks if tr.is_confirmed()]
    print(f"[DEBUG] frame {frame_i}, confirmed tracks: {confirmed_ids}")
    
    # Iterate over tracks produced by DeepSort and assign labels
    for tr in tracks:
        '''
        if not tr.is_confirmed():
            continue # Ignore tentative tracks
        '''
        
        tid=tr.track_id
        
        current_label=track_label_map.get(tid,None)
        
        tlbr=tr.to_tlbr() # Convert coordinates to top left bottom right
        x1,y1,x2,y2=map(int,tlbr)
        
        # Clamp bbox
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))
        
        # Skip invalid bboxes
        if x2<=x1 or y2<=y1:
            continue 
        
        # Grab crop for this track
        crop=rgb[y1:y2,x1:x2]
        
        if processed_i%POSE_FRAME==0:
        
            pose_results=yolo_pose_model(crop, device="cpu", verbose=False) # Model ran on cropped boxes

            if pose_results and pose_results[0].keypoints is not None and len(pose_results[0].keypoints.xy)>0:
                kpts=pose_results[0].keypoints.xy[0].cpu().numpy() # Add keypoints if they dont exist, i.e. body joints
            else:
                kpts=None
            
            if kpts is not None:
                action=action_classift(kpts)
                if tid not in track_action_map:
                    track_action_map[tid] = []
                    
                track_action_map[tid].append(action)
                
                if len(track_action_map[tid]) > 20:
                    track_action_map[tid].pop(0)
        
        if processed_i % 10 == 0:
            emb = get_embeddings_from_crop(crop)
        else:
            emb = None
        
        # Compare this embedding against the known embeddings using cosine similarity
        if emb is not None and len(known_emb)>0:
            sim_score=[cos_sim(emb,k) for k in known_emb] # List of similarity score vs known identity
            best_idx=int(np.argmax(sim_score)) # Returns the index for the best match
            best_score=sim_score[best_idx]
            
            if best_score>=EMB_MATCH_THRESHOLD:
                
                prev_best=track_best_score.get(tid,-1.0)
                
                if best_score>prev_best:
                    track_label_map[tid]=str(known_labels[best_idx])
                    track_best_score[tid]=best_score # Assign the best score
                
                if best_score>=LOCK_THRESHOLD:
                    track_best_score[tid]=1.0

        # If face not seen keep the same id as before
        
        draw_label=track_label_map.get(tid)
        if draw_label is None:
            draw_label="Unknown"
        
        actions=track_action_map.get(tid,[])
        
        if len(actions) > 0:
            action = max(set(actions), key=actions.count)
        else:
            action = "attentive"
        
        # Draw bounding box on original BGR frame for visualization
        cv2.rectangle(frame_small,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame_small,f"{tid}:{draw_label}|{action}",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        
        # Log the current event: timestamp, frame number, trackid, label and bbox coordinate
        csvw.writerow([time.time(),frame_i,tid,draw_label,x1,y1,x2,y2])
    
    # Write the annotated frame to output video
    vw.write(frame_small)


end=time.time()
total_time=end-start
print("Processed frames: ",frame_i)
print("Total time(s): ",total_time)
print("FPS: ",frame_i/total_time)

cap.release()
vw.release()
log_f.close()

print(f"Done. Output: {OUT_VID} {LOG_CSV}")