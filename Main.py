import numpy as np  
 
import cv2 as cv  
 
  # Read the input video  
 
cap = cv.VideoCapture('sample.mp4')  
   
# take first frame of video 
 
ret, frame = cap.read()  # frame = first frame var

# Get the width and height of the first video frame
frame_width = frame.shape[1]  # width of the first frame
frame_height = frame.shape[0] # height of the first frame
   
# setup initial region of tracker  
# Get the width and height of the video frame
#frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) # auto tracking
#frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) # auto tracking

print(f"Frame Width: {frame_width}, Frame Height: {frame_height}")
 
#x, y, width, height = 400, 500, 628, 640 # static
#x, y, width, height = 425, 500, frame_width, frame_height # left leg
x, y, width, height = 250,150, frame_width, frame_height # whole body
#x, y, width, height = 50,150, frame_width, frame_height
#x, y, width, height = 250,400, frame_width, frame_height

track_window = (x, y, width, height)  
 
   
# set up the Region of interest for tracking  
 
roi = frame[y:y + height, x : x + width]  
   
# convert ROI from BGR to HSV format  
 
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)  
 
   
# perform masking operation  
 
mask = cv.inRange(hsv_roi, np.array((0., 50., 50.)), np.array((180., 255., 255.)))
 
 
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])  
 
   
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)  
 
  # Setup the termination criteria, either 15 iteration or move by atleast 2 pt  
 
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 2)  
 
   
while(1):      
 
    ret, frame = cap.read()      
 
    # Resize the video frames.  
 
    frame = cv.resize(frame, (720, 720), fx = 0, fy = 0, 
                      interpolation = cv.INTER_CUBIC)  
 
    cv.imshow('Original', frame)   
 
    # perform thresholding on the video frames  
 
    ret1, frame1 = cv.threshold(frame, 180, 155, cv.THRESH_TOZERO_INV)  
   
    # convert from BGR to HSV format.  
 
    hsv = cv.cvtColor(frame1,cv.COLOR_BGR2HSV)   
 
    dst = cv.calcBackProject([hsv],[0],roi_hist, [0, 180], 1)  
 
    # apply Camshift to get the new location  
 
    ret2, track_window = cv.CamShift(dst,track_window,term_crit)    
 
    # Draw it on image  
 
    pts = cv.boxPoints(ret2)       
 
    # convert float to int
 
    pts = np.int64(pts)
   
 
    # Draw Tracking window on video frame
 
    Result = cv.polylines(frame,[pts],True,(0, 255, 255),2)  
 
    cv.imshow('Camshift', Result)  
 
    # set ESC key as the exit button.  
 
    k = cv.waitKey(30) & 0xff 
 
    if k == 27:  
      break  
 
# Release the cap object  
cap.release()    
# close all opened windows  
cv.destroyAllWindows()