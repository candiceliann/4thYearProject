import argparse
import glob
import os
import os.path as ops
import time

import cv2
import glog as log
import numpy as np
import tensorflow as tf
import tqdm

from LaneNet.config import global_config
from LaneNet.lanenet_model import lanenet
from LaneNet.lanenet_model import lanenet_postprocess

from LaneNet.tools import evaluate_model_utils
import json
CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
        help='path to input image')
    parser.add_argument('--lanenet_weights', required=True,
        help='The model weights path')
    parser.add_argument('--json_save', type=bool,
        help='Save inference data to file inf_data.json (True/False')
    parser.add_argument('--output', type=str,
        help='The test output save root dir')
    return vars(parser.parse_args())


def test_lanenet_batch(src_dir, weights_path, save_dir, save_json=True):
    """

    :param src_dir:
    :param weights_path:
    :param save_dir:
    :return:
    """
    assert ops.exists(src_dir), '{:s} not exist'.format(src_dir)

    os.makedirs(save_dir, exist_ok=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        image_list = glob.glob('{:s}/**/*.jpg'.format(src_dir), recursive=True)
        avg_time_cost = []
        #json_gt = [json.loads(line) for line in open('/Users/mylesfoley/git/lanenet-lane-detection/ROOT_DIR/TUSIMPLE_DATASET/test_set/label_data.json')]
        lane_list = []
        for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
            avg_time_cost.append(time.time() - t_start)
            image_name = image_path.split('/')[-1]
            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis,
                raw_file=image_name
            )
            lane_list.append(postprocess_result['lane_data'])
            if save_json == True:
                if os.path.isfile('Images/JSON/inf_data.json'):
                    with open('Images/JSON/inf_data.json', 'a+') as json_file:
                        json.dump(postprocess_result['lane_data'], json_file)
                        json_file.write('\n')
                else:
                    with open('Images/JSON/inf_data.json', 'w+') as json_file:
                        json.dump(postprocess_result['lane_data'], json_file)
                        json_file.write('\n')
            image_name = image_path.split('/')[-1]
            if index % 10 == 0:
                log.info('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)))
                avg_time_cost.clear()


            input_image_dir = image_path.split('/')[-2]
            input_image_name = image_path.split('/')[-1]
            output_image_dir = save_dir
            os.makedirs(output_image_dir, exist_ok=True)
            output_image_path = ops.join(output_image_dir, input_image_name)
            if ops.exists(output_image_dir):
                cv2.imwrite(output_image_path, postprocess_result['source_image'])
                #cv2.imwrite(output_image_path, postprocess_result['mask_image'])
                #cv2.imwrite(output_image_path, binary_seg_image[0] * 255)

    return lane_list


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    inference_data = test_lanenet_batch(
        src_dir=args.input,
        weights_path=args.lanenet_weights,
        save_json=args.json_save,
        save_dir=args.output
    )
    print(inference_data)
