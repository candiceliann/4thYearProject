from Yolo import yolo
from LaneNet import lane_inference
from LSTM import lstm
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True,
    help="path to input image")
ap.add_argument("--output", required=False,
    help="path to output image")
ap.add_argument("--yolo_weights", required=True,
    help="path to yolo weights directory")
ap.add_argument("--lanenet_weights", required=True,
    help="path to lanenet weights directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# Run yolo
yolo.yolo(args)

# Run LaneNet
lane_inference.test_lanenet_batch(src_dir=args["input"], weights_path=args["lanenet_weights"], save_json=True,
                           save_dir=args["output"]+'/lanes')

# Detect whether a lane change has occured or not in the sequence
lstm.laneChange()
