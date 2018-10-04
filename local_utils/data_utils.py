#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午6:46
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : data_utils.py
# @IDE: PyCharm Community Edition
"""
Implement some utils used to convert image and it's corresponding label into tfrecords
"""
import numpy as np
import tensorflow as tf
import os
import os.path as ops
import sys

from global_configuration import config
from local_utils import establish_char_dict


class FeatureIO(object):
    """
        Implement the base writer class
    """
    def __init__(self, char_dict_path=ops.join(os.getcwd(), 'data/char_dict/char_dict.json'),
                 ord_map_dict_path=ops.join(os.getcwd(), 'data/char_dict/ord_map.json')):
        self.__char_dict = establish_char_dict.CharDictBuilder.read_char_dict(char_dict_path)
        self.__ord_map = establish_char_dict.CharDictBuilder.read_ord_map_dict(ord_map_dict_path)
        return

    @property
    def char_dict(self):
        """

        :return:
        """
        return self.__char_dict

    @staticmethod
    def int64_feature(value):
        """
            Wrapper for inserting int64 features into Example proto.
        """
        if not isinstance(value, list):
            value = [value]
        value_tmp = []
        is_int = True
        for val in value:
            if not isinstance(val, int):
                is_int = False
                value_tmp.append(int(float(val)))
        if is_int is False:
            value = value_tmp
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_feature(value):
        """
            Wrapper for inserting float features into Example proto.
        """
        if not isinstance(value, list):
            value = [value]
        value_tmp = []
        is_float = True
        for val in value:
            if not isinstance(val, int):
                is_float = False
                value_tmp.append(float(val))
        if is_float is False:
            value = value_tmp
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def bytes_feature(value):
        """
            Wrapper for inserting bytes features into Example proto.
        """
        if not isinstance(value, bytes):
            if not isinstance(value, list):
                value = value.encode('utf-8')
            else:
                value = [val.encode('utf-8') for val in value]
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def char_to_int(self, char):
        """

        :param char:
        :return:
        """
        temp = ord(char)
        # convert upper character into lower character
        if 65 <= temp <= 90:
            temp = temp + 32

        for k, v in self.__ord_map.items():
            if v == str(temp):
                temp = int(k)
                return temp
        raise KeyError("Character {} missing in ord_map.json".format(char))

        # TODO
        # Here implement a double way dict or two dict to quickly map ord and it's corresponding index


    def int_to_char(self, number):
        """

        :param number:
        :return:
        """
        if number == '1':
            return '*'
        if number == 1:
            return '*'
        else:
            return self.__char_dict[str(number)]

    def encode_labels(self, labels):
        """
            encode the labels for ctc loss
        :param labels:
        :return:
        """
        encoded_labeles = []
        lengths = []
        for label in labels:
            encode_label = [self.char_to_int(char) for char in label]
            encoded_labeles.append(encode_label)
            lengths.append(len(label))
        return encoded_labeles, lengths

    def sparse_tensor_to_str(self, spares_tensor: tf.SparseTensor):
        """
        :param spares_tensor:
        :return: a str
        """
        indices = spares_tensor.indices
        values = spares_tensor.values
        values = np.array([self.__ord_map[str(tmp)] for tmp in values])
        dense_shape = spares_tensor.dense_shape

        number_lists = np.ones(dense_shape, dtype=values.dtype)
        str_lists = []
        res = []
        for i, index in enumerate(indices):
            number_lists[index[0], index[1]] = values[i]
        for number_list in number_lists:
            str_lists.append([self.int_to_char(val) for val in number_list])
        for str_list in str_lists:
            res.append(''.join(c for c in str_list if c != '*'))
        return res


class TextFeatureWriter(FeatureIO):
    """
        Implement the crnn feature writer
    """
    def __init__(self):
        super(TextFeatureWriter, self).__init__()
        return

    def write_features(self, tfrecords_path, labels, images, imagenames):
        """

        :param tfrecords_path:
        :param labels:
        :param images:
        :param imagenames:
        :return:
        """
        assert len(labels) == len(images) == len(imagenames)

        labels, length = self.encode_labels(labels)

        if not ops.exists(ops.split(tfrecords_path)[0]):
            os.makedirs(ops.split(tfrecords_path)[0])

        with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
            for index, image in enumerate(images):
                features = tf.train.Features(feature={
                    'labels': self.int64_feature(labels[index]),
                    'images': self.bytes_feature(image),
                    'imagenames': self.bytes_feature(imagenames[index])
                })
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                sys.stdout.write('\r>>Writing {:d}/{:d} {:s} tfrecords'.format(index+1, len(images), imagenames[index]))
                sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
        return


class TextFeatureReader(FeatureIO):
    """
        Implement the crnn feature reader
    """
    def __init__(self):
        super(TextFeatureReader, self).__init__()
        return

    @staticmethod
    def read_features(cfg: EasyDict, tfrecords_path, batch_size: int, num_threads: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        :param cfg:
        :param tfrecords_path:
        :param batch_size:
        :param num_threads:
        :return: input_images, input_labels, input_image_names
        """

        assert ops.exists(tfrecords_path), "tfrecords file not found: %s" % tfrecords_path

        def extract_batch(x):
            return TextFeatureReader.extract_features_batch(x, cfg.ARCH.INPUT_SIZE, cfg.ARCH.INPUT_CHANNELS)

        dataset = tf.data.TFRecordDataset(tfrecords_path)
        dataset = dataset.batch(cfg.TRAIN.BATCH_SIZE, drop_remainder=True)
        dataset = dataset.map(extract_batch, num_parallel_calls=num_threads)
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(batch_size * num_threads))
        dataset = dataset.prefetch(buffer_size=batch_size * num_threads)
        iterator = dataset.make_one_shot_iterator()
        input_images, input_labels, input_image_names = iterator.get_next()
        return input_images, input_labels, input_image_names

    @staticmethod
    def extract_features_batch(serialized_batch, input_size: Tuple[int, int], input_channels: int) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        features = tf.parse_example(serialized_batch,
                                    features={'images': tf.FixedLenFeature((), tf.string),
                                              'imagenames': tf.FixedLenFeature([1], tf.string),
                                              'labels': tf.VarLenFeature(tf.int64), })
        bs = features['images'].shape[0]
        images = tf.decode_raw(features['images'], tf.uint8)
        w, h = input_size
        images = tf.cast(x=images, dtype=tf.float32)
        images = tf.reshape(images, [bs, h, w, input_channels])
        labels = features['labels']
        labels = tf.cast(labels, tf.int32)
        imagenames = features['imagenames']
        return images, labels, imagenames


class TextFeatureIO(object):
    """
        Implement a crnn feture io manager
    """
    def __init__(self):
        """

        """
        self.__writer = TextFeatureWriter()
        self.__reader = TextFeatureReader()
        return

    @property
    def writer(self):
        """

        :return:
        """
        return self.__writer

    @property
    def reader(self):
        """

        :return:
        """
        return self.__reader
