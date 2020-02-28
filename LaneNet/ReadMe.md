Download the LaneNet Git repo from: https://github.com/MaybeShewill-CV/lanenet-lane-detection

Follow instructions there for the download

Download:
  - The lane_inference.py file (currently run from ./lanenet-lane-detection - this is the sub directory of lanenet)
  - The modified lanenet_postprocessing.py (to replace the file of the same name in ./lanenet-lane-detection/lanenet_model/ )
  
Run lane_inference.py - currently takes input of the following args:
 '--image_dir' The source lane test data dir, can include sub folders 
 '--weights_path' The model weights path, I saved mine to ./model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt
 '--json_save' True/False if you also want to save the data results to a json - default is false
