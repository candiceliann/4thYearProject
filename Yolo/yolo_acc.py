# USAGE
# python yolo.py --input images/object-detection-crowdai --output output_images --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from collections import namedtuple
import json


def args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to input image")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output image")
    ap.add_argument("-y", "--yolo", required=True,
                    help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    return args


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def yolo(args):
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    dictPath = os.path.sep.join([args["yolo"], "data.json"])
    onlyfiles = [f for f in listdir(
        args["input"]) if isfile(join(args["input"], f))]
    #path = os.path.sep.join([args["input"], "labels_crowdai.csv"])
    data = pd.read_csv(path)
    data = data.dropna()
    print(data.head())
    xmin = list(data.xmin)
    ymin = list(data.ymin)
    xmax = list(data.xmax)
    ymax = list(data.ymax)
    frame = list(data.Frame)
    label = list(data.Label)
    accuracy = 0
    total = 0
    boxdictionary = []
    dictionary = []

    for picture in range(len(onlyfiles)):  # range(len(onlyfiles))
        if '.jpg' in onlyfiles[picture]:
            image = cv2.imread(args["input"]+'/'+onlyfiles[picture])
            (H, W) = image.shape[:2]

            # determine only the *output* layer names that we need from YOLO
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            # construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()

            # show timing information on YOLO
            print("[INFO] YOLO took {:.6f} seconds".format(
                end - start), onlyfiles[picture])

            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > args["confidence"]:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                                    args["threshold"])

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(
                        LABELS[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)
                    # print(x,y,x+w,y+h,text)
                    pred = [x, y, x+w, y+h]
                    all_iou = []
                    positions = []
                    for a in range(len(frame)):
                        if frame[a] == onlyfiles[picture]:
                            gt = [int(xmin[a]), int(ymin[a]),
                                  int(xmax[a]), int(ymax[a])]
                            iou = bb_intersection_over_union(gt, pred)
                            all_iou.append(iou)
                            positions.append(a)
                    if len(all_iou) > 0:
                        big = all_iou.index(max(all_iou))
                        # print(all_iou)
                        for i in range(len(frame)):
                            if i == positions[big]:
                                if all_iou[big] >= 0.5:
                                    ##                                                        predicted_label = text.split(':')
                                    # if ((label[i] == "Car" and predicted_label[0] == "car") or (label[i] == "Truck" and predicted_label[0] == "truck") or (label[i] == "Pedestrian" and predicted_label[0] == "person")):
                                    accuracy = accuracy + 1
        # print(x,'lies between',int(xmin[i])-50,int(xmin[i])+50,'\n')

                    predicted_label = text.split(':')
                    singledict = {"xmin": x,
                                  "ymin": y,
                                  "xmax": x+w,
                                  "ymax": y+h,
                                  "Label": predicted_label[0]}
                    boxdictionary.append(singledict)

            for i in range(len(frame)):
                if frame[i] == onlyfiles[picture]:
                    total = total + 1
            cv2.imwrite(args["output"]+'/'+onlyfiles[picture], image)
            print('Accuracy = ', accuracy, total, accuracy/total)
            filedictionary = {onlyfiles[picture]: boxdictionary}
            dictionary.append(filedictionary)

    with open('data.json', 'w') as fp:
        json.dump(dictionary, fp, indent=4)

    return None


if __name__ == '__main__':
    args = args()
    yolo(args)
