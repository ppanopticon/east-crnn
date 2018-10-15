#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午1:39
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : train_shadownet.py
# @IDE: PyCharm Community Edition
"""
Train shadow net script
"""
import os
import tensorflow as tf
import os.path as ops
import time
import numpy as np
import argparse
import cv2
import json
import functools
import collections

from models.crnn import crnn_model
from models.east import model
from shutil import copyfile

from eval import resize_image, sort_poly, detect

from utils import data_utils, log_utils
from utils.config_utils import load_config

logger = log_utils.init_logger()

# Read configuration for width/height
cfg = load_config().cfg

@functools.lru_cache(maxsize=100)
def get_crnn(checkpoint_path):
    logger.info('Loading CRNN model...')

    # Read width / height
    w, h = cfg.ARCH.INPUT_SIZE

    # Determine the number of classes.
    decoder = data_utils.TextFeatureIO().reader
    num_classes = len(decoder.char_dict) + 1

    g_2 = tf.Graph()
    with g_2.as_default():
        inputdata = tf.placeholder(dtype=tf.float32, shape=[1, h, w, 3], name='input')
        net = crnn_model.ShadowNet(phase='Test', hidden_nums=cfg.ARCH.HIDDEN_UNITS, layers_nums=cfg.ARCH.HIDDEN_LAYERS, num_classes=num_classes)
        with tf.variable_scope('shadow'):
            net_out = net.build_shadownet(inputdata=inputdata)

        decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=cfg.ARCH.SEQ_LENGTH * np.ones(1), merge_repeated=False)
        decoder = data_utils.TextFeatureIO()

        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=g_2)

        ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
        model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))

        saver.restore(sess=sess, save_path=model_path)

        def infer(img):
            img = cv2.resize(img, (w, h))
            img = np.expand_dims(img, axis=0).astype(np.float32)
            preds = sess.run(decodes, feed_dict={inputdata: img})
            preds = decoder.writer.sparse_tensor_to_str(preds[0])
            return preds

        return infer

@functools.lru_cache(maxsize=100)
def get_east(checkpoint_path):
    logger.info('Loading EAST model...')

    g_1 = tf.Graph()
    with g_1.as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(cfg, input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=g_1)

        ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
        model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        logger.info('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)

        def infer(img):
            """
            :return: {
                'text_lines': [
                    {
                        'score': ,
                        'x0': ,
                        'y0': ,
                        'x1': ,
                        ...
                        'y3': ,
                    }
                ],
                'rtparams': {  # runtime parameters
                    'image_size': ,
                    'working_size': ,
                },
                'timing': {
                    'net': ,
                    'restore': ,
                    'nms': ,
                    'cpuinfo': ,
                    'meminfo': ,
                    'uptime': ,
                }
            }
            """
            start_time = time.time()
            timer = collections.OrderedDict([
                ('net', 0),
                ('restore', 0),
                ('nms', 0)
            ])
            im_resized, (ratio_h, ratio_w) = resize_image(img)
            start = time.time()
            score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized[:,:,::-1]]})
            timer['net'] = time.time() - start

            boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
            logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

            if boxes is not None:
                scores = boxes[:,8].reshape(-1)
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h

            duration = time.time() - start_time
            timer['overall'] = duration
            logger.info('[timing] {}'.format(duration))

            text_lines = []
            if boxes is not None:
                text_lines = []
                for box, score in zip(boxes, scores):
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                        continue
                    tl = collections.OrderedDict(zip(
                        ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                        map(float, box.flatten())))
                    tl['score'] = float(score)
                    text_lines.append(tl)
            ret = {
                'text_lines': text_lines,
                'timing': timer,
            }
            return ret

        return infer


def extract(extraction_dir: str, output: str = None, copy: bool = False):
    # Load relevant models
    infer_boxes = get_east(cfg.PATH.EAST_MODEL_SAVE_DIR)
    infer_text = get_crnn(cfg.PATH.CRNN_MODEL_SAVE_DIR)

    # Some statistics
    east_count = 0
    crnn_count = 0
    crnn_duration = 0
    east_duration = 0

    # Iterate over items in extraction directory
    for root, subdirs, files in os.walk(extraction_dir):

        # Iterate over files
        for filename in files:
            if (filename.endswith(('.jpg','.jpeg','.png','.tiff','.tif','bmp'))):
                image_path = os.path.join(root,filename)
                logger.info('Processing {}'.format(image_path))
                img = cv2.imdecode(np.fromfile(image_path, dtype='uint8'), 1)

                # Infer bounding boxes
                start = time.time()
                boxes = infer_boxes(img)
                east_duration += time.time() - start
                east_count += 1

                if len(boxes["text_lines"]) > 0:
                    for line in boxes["text_lines"]:
                        xt = int(min(line["x0"], line["x2"])) - 1
                        yt = int(min(line["y0"], line["y2"])) - 1
                        xb = int(max(line["x1"], line["x3"])) + 1
                        yb = int(max(line["y1"], line["y3"])) + 1
                        cropped_img = img[yt:yb, xt:xb]
                        if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                            logger.warn('Error in image {}: Nullary bounding box...'.format(image_path))
                            continue

                        # Infer text
                        start = time.time()
                        line["text"] = infer_text(cropped_img)[0]
                        crnn_duration += time.time() - start
                        crnn_count += 1

                    # Generate output
                    if output is None:
                        write_output(boxes["text_lines"], root, image_path, False)
                    else:
                        write_output(boxes["text_lines"], output, image_path, copy)

    crnn_duration /= crnn_count
    east_duration /= east_count

    logger.info('Done! Took {:.2f}s on average (EAST: {:.2f}s, CRNN: {:.2f}s).'.format((crnn_duration + east_duration), east_duration, crnn_duration))




def write_output(data, output_dir, image_path, copy = False):
    with open(os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".json"), 'w') as outf:
        json.dump(data, outf)
        if copy:
            copyfile(image_path, os.path.join(output_dir, os.path.basename(image_path)))


if __name__ == '__main__':
    # Initialize arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--extraction_dir', type=str, help='Path to dir containing images for extraction.')
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Path to directory that should hold the final output.')
    parser.add_argument('-c', '--copy_images', type=bool, default=False, help='Only valid for -o; copies the image.')

    args = parser.parse_args()

    if not ops.exists(args.extraction_dir):
        raise ValueError('{:s} doesn\'t exist'.format(args.extraction_dir))

    # Start extraction
    extract(args.extraction_dir, args.output_dir, args.copy_images)