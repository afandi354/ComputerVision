# import the necessary packages
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import numpy as np

# Take args from command
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the JSON configuration file")
args = vars(ap.parse_args())

warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None


# initialize the camera
video_capture = cv2.VideoCapture(0)

# uploaded timestamp, and frame motion counter
print ("[INFO] Starting...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0
out = None
path = None

fourcc = cv2.VideoWriter_fourcc(*'XVID')

# capture frames from the camera
while True:
    # grab the raw NumPy array representing the image and initialize
    # the timestamp and occupied/unoccupied text
    ret, frame = video_capture.read()
    timestamp = datetime.datetime.now()
    text = "Tidak Terdeteksi"

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the average frame is None, initialize it
    if avg is None:
        print ("[INFO] starting model...")
        avg = gray.copy().astype("float")
        continue

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.1)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Terdeteksi"

    # draw the text and timestamp on the frame
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, "Status Ruangan: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # check to see if the room is occupied
    if text == "Occupied":

        newPath = timestamp.strftime("%d-%m-%y" + ".avi")

        if (path == None) | (newPath != path):
            path = newPath
            width = np.size(frame, 1)
            height = np.size(frame, 0)
            out = cv2.VideoWriter(
                filename=newPath, fourcc=fourcc, fps=15, frameSize=(width, height))

        out.write(frame)

        lastUploaded = timestamp
    # otherwise, the room is not occupied
    else:
        # Min record time  
        if (timestamp - lastUploaded).seconds <= 20:
            newPath = timestamp.strftime("%d-%m-%y" + ".avi")

            # If new day started then change change the storage location 
            if (path == None) | (newPath != path):
                path = newPath
                width = np.size(frame, 1)
                height = np.size(frame, 0)
                out = cv2.VideoWriter(
                    filename=newPath, fourcc=fourcc, fps=15, frameSize=(width, height))

            abc = out.write(frame)
            
    # check to see if the frames should be displayed to screen
    if conf["show_video"]:
        # display the security feed
        cv2.imshow("Security Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
