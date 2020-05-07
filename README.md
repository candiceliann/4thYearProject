# Prolific Pandemic Project
## Integrated Model
### Weights
Download the YOLOv3 weights [here](https://pjreddie.com/darknet/yolo/). Use the YOLOv3-416 as that has a good mAP with over 30 fps. Put it in `Yolo/weights`. The corresponding cfg file is already included.

Download the LaneNet weights [here](https://www.dropbox.com/sh/tnsf0lw6psszvy4/AAA81r53jpUI3wLsRW6TiPCya?dl=0). Download all the files in the folder and put it in `LaneNet/weights`.

### Parameters
LaneNet parameters can be changed in the `LaneNet/config/global_config.py` file. The last three postprocessing parameters are the most relevant ones to find the optimal accuracy/speed tradeoff.

### Running
Put images you want to test in the `Images/input` folder. Then go to the command line, cd to this directory and run `python main.py --input images/input --output images/output --yolo_weights Yolo/weights --lanenet_weights LaneNet/weights/tusimple_lanenet_vgg.ckpt`.

The object detection output are be displayed in `Images/Output/objects`, the lanes are displayed in `Images/Output/lanes`, and the JSON outputs are in `Images/JSON`.
