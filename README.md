## Yolo:
Dataset used for testing from: https://github.com/udacity/self-driving-car/tree/master/annotations (Dataset 1).
This is to be implemented in the 'images' folder.
Add the weights file which can be found in the yolo***.md google drive folder to the 'yolo-coco' file.
(Move this in yolo folder later if needed)

## Integrated Model:
### Weights:
Download the YOLOv3 weights [here](https://pjreddie.com/darknet/yolo/). Use the YOLOv3-416 as that has a good mAP with over 30 fps. Put it in `Yolo/weights`. The corresponding cfg file is already included.

Download the LaneNet weights [here](https://www.dropbox.com/sh/tnsf0lw6psszvy4/AAA81r53jpUI3wLsRW6TiPCya?dl=0). Download all the files in the folder and put it in `LaneNet/weights`. (maybe change the names of the files later to make it look nicer).

### Running:
Put images you want to test in the `Images/input` folder. Then go to the command line, cd to this directory and run `python main.py --input images/input --output images/output --yolo_weights Yolo/weights --lanenet_weights LaneNet/weights/tusimple_lanenet_vgg.ckpt`.

## TODO:
1. [ ] Fix LaneNet output image and check if json file outputs correctly
2. [ ] Put jsons into their own folder
3. [ ] Clean up unneccessary folders and README files
