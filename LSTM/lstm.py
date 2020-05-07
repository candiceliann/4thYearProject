from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from sklearn.preprocessing import normalize
from pathlib import Path
import numpy as np
import tensorflow as tf
import os,glob
import pandas as pd
import json

def preprocessYolo(object_path):
# Preprocesses Yolo outputs to (m, 30, 12) to input to lstm where m is number of 30-frame testing sequences

    with open(object_path) as f:
        dictionary = json.load(f)

    data = []

    for image in dictionary:
        if len(dictionary[image]) > 3:
            dictionary[image] = dictionary[image][0:3]

        while len(dictionary[image]) < 3:
            dictionary[image].append([0,0,0,0,0,0])

        data.append(dictionary[image])

    outerlist = []
    innerlist = []
    megalist = []
    x = 1

    for i in data:
        for j in i:
            for k in range(4):
                innerlist.append(j[k])

            if x % 3 == 0:
                outerlist.append(innerlist)
                innerlist = []

                if len(outerlist) % 30 == 0:
                    megalist.append(outerlist)
                    outerlist = []

            x = x + 1

    while len(outerlist) % 30 != 0:
        outerlist.append([0,0,0,0,0,0,0,0,0,0,0,0])

    if len(outerlist) == 30:
        megalist.append(outerlist)

    return np.array(megalist)

def normalizeYolo(yolo):
# Normalizes the yolo output before inputting to lstm

    reshaped = []

    for i in range(yolo.shape[0]):
        reshaped.append([yolo[i, :, 0], yolo[i, :, 1], yolo[i, :, 2], yolo[i, :, 3], yolo[i, :, 4], yolo[i, :, 5],
                         yolo[i, :, 6], yolo[i, :, 7], yolo[i, :, 8], yolo[i, :, 9], yolo[i, :, 10], yolo[i, :, 11]])

    reshaped = np.array(reshaped).astype(np.float)

    norm = np.zeros(reshaped.shape)
    minmax = np.zeros((reshaped.shape[0], reshaped.shape[1], 2))

    for i in range(reshaped.shape[0]):
        for j in range(reshaped.shape[1]):
            min_val = np.amin(reshaped[i, j])
            max_val = np.amax(reshaped[i, j])
            minmax[i, j, 0] = min_val
            minmax[i, j, 1] = max_val
            if max_val - min_val != 0:
                for k in range(reshaped.shape[2]):
                    norm[i, j, k] = (float(reshaped[i, j, k]) - min_val)/(max_val - min_val)
            else:
                norm[i, j, :] = reshaped[i, j, :]

    yolo_norm = np.zeros(yolo.shape)

    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            for k in range(norm.shape[2]):
                yolo_norm[i, k, j] = norm[i, j, k]

    return yolo_norm, minmax

def loadModel(model_path):
# Loads a pre-trained model from json architecture file and h5 weights file

    with open(os.path.join(model_path, 'modelacc.json'),'r') as f:
        model = model_from_json(f.read())

    model.load_weights(os.path.join(model_path, 'modelacc.h5'))

    return model

def detectLaneChange(lane_path, predictions, minmax):
# Detects whether a lane change has occured given the lstm prediction, lane detection file, and normalization
# parameters

    lane_data = []
    lane_frames = []

    with open(lane_path) as f:
        for line in f:
            if line:
                lane_data.append(json.loads(line))

    # Lane Preprocessing
    lane_curves = {}

    for image in lane_data:
        for key in image:
            lane_frames.append(int(key.split('.')[0]))
            llist = []
            h_samples_list = []
            for i in range(0,len(image[key]['lanes'])):
                llist = []

                if image[key]['lanes'][i]:
                    llist = image[key]['lanes'][i]
                    hlist = image[key]['h_samples'][i]
                    curve_coeffs = np.polyfit(np.array(llist), np.array(hlist), 2)
                    y = np.linspace(np.amin(llist),np.amax(llist))

                    if key in lane_curves.keys():
                        lane_curves[key].append(((curve_coeffs[0] * y ** 2+ curve_coeffs[1] * y + curve_coeffs[2]), y))
                    else:
                        lane_curves[key] = [((curve_coeffs[0] * y ** 2+ curve_coeffs[1] * y + curve_coeffs[2]), y)]

    dummy_counter = 0

    for key in lane_curves.keys():
        for lane in range(0, len(lane_curves[key])):
            try:
                if lane_curves[key][lane][1][-1] < lane_curves[key][lane+1][1][-1]:
                    dummy = lane_curves[key][lane+1]
                    lane_curves[key][lane+1] = lane_curves[key][lane]
                    lane_curves[key][lane] = dummy
            except:
                dummy_counter +=1

    # Un-normalizing the predictions
    pred_norm = np.zeros(predictions.shape)

    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            for k in range(predictions.shape[2]):
                pred_norm[i, j, k] = (predictions[i, j, k] *
                                      (minmax[i, k, 1] - minmax[i, k, 0]))+ minmax[i, k, 0]

    # Finding the centre points of the predictions
    pred_centre = []

    for sequence in pred_norm:
        centre_list = []
        for tuples in sequence:
            obj_1_centre = ((tuples[2] + tuples[0])/2, (tuples[3] + tuples[1])/2)
            obj_2_centre = ((tuples[6] + tuples[4])/2, (tuples[7] + tuples[5])/2)
            obj_3_centre = ((tuples[10] + tuples[8])/2, (tuples[11] + tuples[9])/2)
            centre_list.append((obj_1_centre, obj_2_centre, obj_3_centre))
        pred_centre.append(centre_list)

    pred_centre = np.array(pred_centre)
    pred_centre = pred_centre.reshape(1, pred_centre.shape[0]*pred_centre.shape[1], 3, 2)
    pred_centre = pred_centre.tolist()

    # Determining whether a lane change has occurred or not
    obj_states = [0, 0, 0]
    obj_changes = []
    lane_changed_obj = []
    image_offest = int(lane_frames[0])


    for object_pairs in pred_centre[0]:
        for object_centre in object_pairs:
            current_obj = object_pairs.index(object_centre)
            object_region = obj_states[current_obj]
            current_obj_state = object_region
            for key in lane_curves.keys():
                current_lane = list(lane_curves.keys()).index(key)
                if ('00' + str(pred_centre[0].index(object_pairs)+ image_offest) + '.') in key:
                    current_state = 0
                    lane_quart_points = []
                    for i in range(0, len(lane_curves[key])):
                        xquart = lane_curves[key][i][1][0]
                        lane_quart_points.append(xquart)

                    for quart_point in lane_quart_points:
                        if (object_centre[0] < quart_point) and quart_point - 30 < object_centre[0]:
                            current_state += 1
                    if (current_state) != object_region:
                        obj_changes.append((key, current_obj))
                        obj_states[current_obj] = current_state

            if len(obj_changes) > 3:
                if obj_changes[-1][1] == obj_changes[-2][1] == obj_changes[-3][1]:
                    if (obj_changes[-2], obj_changes[-1]) not in lane_changed_obj:
                        if int((obj_changes[-3][0]).split('.')[0]) + 5 >= int((obj_changes[-2][0]).split('.')[0]):
                            lane_changed_obj.append((obj_changes[-2], obj_changes[-1]))
    obj_count = []
    for pairs in lane_changed_obj:
        obj_count.append(pairs[0][1])
    if len(lane_changed_obj) > 3 or len(lane_changed_obj) == 0:
        print('No lane change detected')
    else:
        print('Lane change occuring from: ' + lane_changed_obj[0][0][0] + ' -> '+ lane_changed_obj[-1][1][0])
        print('Object: ' + str(mode(obj_count)))

def laneChange():
    object_path = os.path.join(Path.cwd(), 'Images/JSON/data.json')
    model_path = os.path.join(Path.cwd(), 'LSTM')
    lane_path = os.path.join(Path.cwd(), 'Images/JSON/inf_data.json')

    yolo = preprocessYolo(object_path)
    yolo_norm, minmax = normalizeYolo(yolo)

    model = loadModel(model_path)
    model.summary()

    predictions = model.predict(yolo_norm, verbose=1)

    print("\n")
    detectLaneChange(lane_path, predictions, minmax)
