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

    sets = ['test', 'train', 'val']
    with open(ops.join(args.root, 'lexicon.txt'), 'r') as lex_file:
        lexicon = np.array([tmp.strip().split() for tmp in lex_file.readlines()])

    for set in sets:
        with open(ops.join(args.root, 'annotation_{}.txt'.format(set)), 'r') as data:
            info = np.array([tmp.strip().split() for tmp in data.readlines()])
            images = np.array([])
            labels = np.array([])
            imagenames = np.array([])
            for item in info:
                np.append(images, cv2.resize(cv2.imread(ops.join(args.root, item[0]), (100,32)), cv2.IMREAD_COLOR))
                np.append(labels, lexicon[int(item[1])])
                np.append(imagenames, ops.basename(item[0]))
            images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in images]

            tfrecord_path = ops.join(args.out, '{}_features.tfrecords'.format(set))
            feature_io.writer.write_features(tfrecords_path=tfrecord_path, labels=labels, images=images, imagenames=imagenames)

if __name__ == '__main__':
    main()