## YOLOv3
### Original Model
The original model can be found [here](https://drive.google.com/drive/folders/1mr7u4xbx2WG24jI1o4o01c3Aa-yUK54U?usp=sharing):

To run the original model use `python yolo_video.py --input videos/overpass.mp4 --output output/overpass.avi --yolo yolo-coco` or whatever the video input name is and desired output name.

### Accuracy Testing
The dataset used for testing can be found [here](https://github.com/udacity/self-driving-car/tree/master/annotations) (Dataset 1). This is to be added in the `images` folder. Then run `python yolo_acc.py --input images/object-detection-crowdai --output output_images --yolo yolo-coco`.

Note: I dont think this one has been updated with the weights folder name changes, will do it later.

CrowdAI Dataset - 59%


