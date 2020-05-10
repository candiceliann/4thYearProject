## YOLOv3
### Original Model
The original model can be found [here](https://drive.google.com/drive/folders/1mr7u4xbx2WG24jI1o4o01c3Aa-yUK54U?usp=sharing):

To run the original model use `python yolo_video.py --input videos/overpass.mp4 --output output/overpass.avi --yolo yolo-coco` or whatever the video input name is and desired output name.

### Accuracy Testing
The CrowdAI dataset used for testing can be found [here](https://github.com/udacity/self-driving-car/tree/master/annotations) (Dataset 1). This is to be added in the `images` folder. Then run `python yolo_acc.py --input images/object-detection-crowdai --output output_images --yolo yolo-coco`. 

The Autti dataset used for testing can be found [here](https://github.com/udacity/self-driving-car/tree/master/annotations) (Dataset 2). This is to be added in the `images` folder. Then run `python yolo.py --input images/object-autti --output output_images --yolo yolo-coco`. 

The BDD dataset used for testing can be found [here](https://bdd-data.berkeley.edu/index.html). This is to be added in the `images` folder. Then run `python yolo_bdd.py --input images/object-bdd --output output_images --yolo yolo-coco`. 

Accuracies when looking at only the 'car' labels:

CrowdAI Dataset - 59%

Autti Dataset - 51%

BDD dataset - 26% 

Accuracies when looking at only the 'car' labels and only the labels that matter (within a certain proximity of the vehicle):

CrowdAI Dataset - 78%

Autti Dataset - 66%

BDD dataset - 78% 
