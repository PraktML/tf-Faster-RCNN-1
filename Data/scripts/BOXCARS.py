#!/usr/bin/env python

"""
|--tf-Faster-RCNN_ROOT
    |--Data/
        |--BoxCars/
            |--Annotations/
                |--*.txt (Annotation Files: (x1,y1,x2,y2,l))
            |--Images/
                |--*.png (Image files)
            |--Names/
                |--[train/valid/test].txt (List of data)
"""

from scipy.misc import imsave

import pickle
import shutil
import argparse
import numpy as np
import os
import tensorflow as tf

# Global Flag Dictionary
flags = {
    'data_directory': '../BoxCars/',
    'nums': {"train": 55000, "valid": 0, "test": 0},
    'all_names': ["train", "valid", "test"],
    'num_classes': 1
}


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='BoxCars Arguments')
    parser.add_argument('-p', '--path')
    parser.add_argument('-n', '--numpy_pickle')
    args = vars(parser.parse_args())
    data_dir = flags['data_directory']

    # Load BoxCars pickle file
    if not args['numpy_pickle']:
        read_pickle(args)

    return
    data = np.load(os.path.join(args['path'], 'BoxCars.npy'), encoding='latin1').item()

    make_Im_An_Na_directories(data_dir)
    # Just do training data for now
    split = flags['all_names'][0]

    i = 0
    for idx, sample_group in data['samples'].items():
        for sample in sample_group['vehicleSamples']:
            fname = split + '_img' + str(i)
            # Add class (0 = background, 1= car) to ground truth
            gt = np.array([np.append(sample['3DBB'].flatten(), [1])])
            shutil.move(os.path.join(args['path'], sample['path']), data_dir + 'Images/' + fname + '.png')

            np.savetxt(data_dir + 'Annotations/' + fname + '.txt', gt, fmt='%i')
            with open(data_dir + 'Names/' + split + '.txt', 'a') as f:
                f.write(fname + '\n')
            print(i)
            i += 1

    # Create data directory
    # make_directory(flags['data_directory'])

    # Create and save the cluttered MNIST digits
    # process_digits(all_data, all_labels, flags['data_directory'], args)

def read_pickle(args):
    data_dir = flags['data_directory']

    with open(os.path.join(args['path'], 'dataset.pkl'), 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    make_Im_An_Na_directories(data_dir)
    # Just do training data for now
    split = flags['all_names'][0]

    i = 0
    for sample_group in data['samples']:
        for sample in sample_group['instances']:
            fname = split + '_img' + str(i)
            # Add class (0 = background, 1= car) to ground truth
            box_delta = np.array(sample['3DBB_offset'])
            box = np.array(sample['3DBB'])
            offsetted_box = np.subtract(box, box_delta)

            gt = np.array([np.append(sample['2DBB'], np.append(offsetted_box.flatten(), [1]))])
            # shutil.move(os.path.join(args['path'], sample['path']), data_dir + 'Images/' + fname + '.png')

            np.savetxt(data_dir + 'Annotations/' + fname + '.txt', gt, fmt='%i')
            with open(data_dir + 'Names/' + split + '.txt', 'a') as f:
                f.write(fname + '\n')
            if i == 2:
                return
            print(i)
            i += 1


def process_digits(all_data, all_labels, data_directory, args):
    """ Generate data and saves in the appropriate format """

    for s in range(len(flags['all_names'])):
        split = flags['all_names'][s]
        print('Processing {0} Data'.format(split))
        key = 'train' if split == 'train' else 'eval'

        # Create writer (tf_records) or Image/Annotations/Names directories (PNGs)
        if args[key] == 'tfrecords':
            tf_writer = tf.python_io.TFRecordWriter(data_directory + 'boxcars_' + split + '.tfrecords')
        elif args[key] == 'PNG':
            make_Im_An_Na_directories(data_directory)
        else:
            raise ValueError('{0} is not a valid data format option'.format(args[key]))

        # Generate data
        for i in trange(flags['nums'][split]):
            # Generate cluttered MNIST image
            im_dims = [im_dims_generator(), im_dims_generator()]

            # Save data
            if args[key] == 'tfrecords':
                img = np.float32(img.flatten()).tostring()
                gt_boxes = np.int32(np.array(gt_boxes).flatten()).tostring()
                tf_write(img, gt_boxes, [flags['im_dims'], flags['im_dims']], tf_writer)
            elif args[key] == 'PNG':
                fname = split + '_img' + str(i)
                imsave(data_directory + 'Images/' + fname + '.png', np.float32(img))
                np.savetxt(data_directory + 'Annotations/' + fname + '.txt', np.array(gt_boxes), fmt='%i')
                with open(data_directory + 'Names/' + split + '.txt', 'a') as f:
                    f.write(fname + '\n')


###############################################################################
# Image generation functions
###############################################################################

def create_gt_bbox(image, minimum_dim, label):
    # Tighten box
    rows = np.sum(image, axis=0).round(1)
    cols = np.sum(image, axis=1).round(1)

    left = np.nonzero(rows)[0][0]
    right = np.nonzero(rows)[0][-1]
    upper = np.nonzero(cols)[0][0]
    lower = np.nonzero(cols)[0][-1]

    # If box is too narrow or too short, pad it out to >12
    width = right - left
    if width < minimum_dim:
        pad = np.ceil((minimum_dim - width) / 2)
        left = int(left - pad)
        right = int(right + pad)

    height = lower - upper
    if height < minimum_dim:
        pad = np.ceil((minimum_dim - height) / 2)
        upper = int(upper - pad)
        lower = int(lower + pad)

    # Save Ground Truth Bounding boxes with Label in 4th position
    if label == 0:  # Faster RCNN regards 0 as background, so change the label for all zeros to 10
        label = 10
    gt_bbox = [int(left), int(upper), int(right), int(lower), int(label)]

    return gt_bbox


###############################################################################
# .tfrecords writer and features helper functions
###############################################################################

def tf_write(pixels, gt_boxes, dims, writer):
    """Write image pixels and label from one example to .tfrecords file"""
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features contains a map of string to Feature proto objects
            feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'dims': _int64_list_features(dims),
                'gt_boxes': _bytes_features(gt_boxes),
                'image': _bytes_features(pixels)
            }))
    # Use the proto object to serialize the example to a string and write to disk
    serialized = example.SerializeToString()
    writer.write(serialized)


def _int64_features(value):
    """Value takes a the form of a single integer"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_features(list_ints):
    """Value takes a the form of a list of integers"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_ints))


def _bytes_features(value):
    """Value takes the form of a string of bytes data"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


###############################################################################
# Miscellaneous
###############################################################################

def im_dims_generator():
    """ Allow user to specify hardcoded image dimension or random rect dims """
    if flags['im_dims'] == 'random':
        return np.random.randint(100, 200)
    else:
        assert flags['im_dims'] > 0
        return flags['im_dims']


def make_directory(folder_path):
    """Creates directory if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def make_Im_An_Na_directories(data_directory):
    '''Creates the Images-Annotations-Names directories for a data split'''
    make_directory(data_directory + 'Images/')
    make_directory(data_directory + 'Annotations/')
    make_directory(data_directory + 'Names/')


if __name__ == "__main__":
    main()
