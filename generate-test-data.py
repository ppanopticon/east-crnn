import argparse
import os.path as ops
import numpy as np
import cv2

from local_utils import data_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()
    feature_io = data_utils.TextFeatureIO()

    array = []
    with open(ops.join(args.root, 'lexicon.txt'), 'r') as lexicon:
        array = np.array([tmp.strip().split() for tmp in lexicon.readlines()])

    with open(ops.join(args.root, 'annotation_test.txt'), 'r') as training:
        info = np.array([tmp.strip().split() for tmp in training.readlines()])
        test_images_org = [cv2.imread(ops.join(args.root, tmp), cv2.IMREAD_COLOR) for tmp in info[:, 0]]
        test_images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in np.array([cv2.resize(tmp, (100, 32)) for tmp in test_images_org])]
        test_labels = np.array([array[tmp] for tmp in info[:, 1]])
        test_imagenames = np.array([ops.basename(tmp) for tmp in info[:, 0]])

        test_tfrecord_path = ops.join(args.out, 'test_features.tfrecords')
        feature_io.writer.write_features(tfrecords_path=test_tfrecord_path, labels=test_labels, images=test_images, imagenames=test_imagenames)

    with open(ops.join(args.root, 'annotation_val.txt'), 'r') as training:
        info = np.array([tmp.strip().split() for tmp in training.readlines()])
        val_images_org = [cv2.imread(ops.join(args.root, tmp), cv2.IMREAD_COLOR) for tmp in info[:, 0]]
        val_images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in np.array([cv2.resize(tmp, (100, 32)) for tmp in val_images_org])]
        val_labels = np.array([array[tmp] for tmp in info[:, 1]])
        val_imagenames = np.array([ops.basename(tmp) for tmp in info[:, 0]])

        test_tfrecord_path = ops.join(args.out, 'val_features.tfrecords')
        feature_io.writer.write_features(tfrecords_path=test_tfrecord_path, labels=val_labels, images=val_images, imagenames=val_imagenames)

    with open(ops.join(args.root, 'annotation_train.txt'), 'r') as training:
        info = np.array([tmp.strip().split() for tmp in training.readlines()])
        train_images_org = [cv2.imread(ops.join(args.root, tmp), cv2.IMREAD_COLOR) for tmp in info[:, 0]]
        train_images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in np.array([cv2.resize(tmp, (100, 32)) for tmp in train_images_org])]
        train_labels = np.array([array[tmp] for tmp in info[:, 1]])
        train_imagenames = np.array([ops.basename(tmp) for tmp in info[:, 0]])

        test_tfrecord_path = ops.join(args.out, 'train_features.tfrecords')
        feature_io.writer.write_features(tfrecords_path=test_tfrecord_path, labels=train_labels, images=train_images, imagenames=train_imagenames)




if __name__ == '__main__':
    main()