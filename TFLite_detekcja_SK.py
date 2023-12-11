######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/2/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util



# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--video', help='Name of the video file',
                    default='test.mp4')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2



# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Calculate the height of the region of interest (ROI)
roi_height = int(imH / 5)

# Calculate the width and height of the region of interest (ROI)
roi_width = int(imW)
roi_start_y = int(imH)
# roi_start_y = int(2 * imH / 3)
roi_end_y = roi_start_y - roi_height
# roi_end_y = roi_start_y + roi_height

# Create a resizable window
cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
cv2.namedWindow('Object detector (No Video)', cv2.WINDOW_NORMAL)


while(video.isOpened()):

    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Draw the region of interest (ROI) using a white dashed line
    cv2.rectangle(frame, (0, roi_start_y), (roi_width, roi_end_y), (255, 255, 255), 2, cv2.LINE_8, 0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    frame_no_video = np.zeros_like(frame)
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            
            center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
            radius = min(xmax - xmin, ymax - ymin) // 2

            if labels[int(classes[i])] == 'Carrot':
                color = (0, 255, 0)  # Green circle for "Carrot"
                label_color = (0, 255, 0)  # Green label
            elif labels[int(classes[i])] == 'Weed':
                color = (0, 0, 255)  # Red circle for "Weed"
                label_color = (0, 0, 255)  # Red label
            else:
                color = (255, 255, 0)  # Default to blue circle for other objects
                label_color = (255, 255, 0)  # Default to blue label

            # Check if the object is within the ROI
            if ymax <= roi_start_y and ymin >= roi_end_y:
                # Draw a red dot in the center of the object
                center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
                dot_radius = 15  # You can adjust the radius as needed
                color = (0, 0, 255)  # Red color for the dot
                cv2.circle(frame_no_video, center, dot_radius, color, -1)
                cv2.circle(frame, center, dot_radius, color, -1)  # Filled red circle
                cv2.circle(frame, center, radius, color, 4)  # Colored circle
                cv2.putText(frame, labels[int(classes[i])], (xmin, ymin - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)  # Colored label
                cv2.putText(frame, f"Detection: {int(scores[i]*100)}%", (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)  # Detection result    

            cv2.circle(frame, center, radius, color, 4)  # Colored circle
            cv2.putText(frame, labels[int(classes[i])], (xmin, ymin - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)  # Colored label
            cv2.putText(frame, f"Detection: {int(scores[i]*100)}%", (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)  # Detection result
            

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Display frame in the window without video
    cv2.imshow('Object detector (No Video)', frame_no_video)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
