# This code is modified from 'https://github.com/fyu/lsun/blob/master/data.py'
# !/usr/bin/env python2.7

from __future__ import print_function
import argparse
import cv2
import lmdb
import numpy
import os
from os.path import exists, join

__author__ = 'Fisher Yu'
__email__ = 'fy@cs.princeton.edu'
__license__ = 'MIT'


def convert(db_path):
    print('Converting', db_path)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    idx = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            print('[', str(idx).zfill(7), '] ', 'Current key:', key)
            if idx > 607315:
                break
            img = cv2.imdecode(numpy.fromstring(val, dtype=numpy.uint8), 1)
            filedir = './data/Img_' + str(idx).zfill(7) + '.png'
            cv2.imwrite(filedir, img)
            idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', type=str,
                        choices=['view', 'export'],
                        help='view: view the images in the lmdb database '
                             'interactively.\n'
                             'export: Export the images in the lmdb databases '
                             'to a folder. The images are grouped in subfolders'
                             ' determinted by the prefiex of image key.')
    parser.add_argument('lmdb_path', nargs='+', type=str,
                        help='The path to the lmdb database folder. '
                             'Support multiple database paths.')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--flat', action='store_true',
                        help='If enabled, the images are imported into output '
                             'directory directly instead of hierarchical '
                             'directories.')
    args = parser.parse_args()

    command = args.command
    lmdb_paths = args.lmdb_path

    for lmdb_path in lmdb_paths:
        if command == 'convert':
            convert(lmdb_path)


if __name__ == '__main__':
    main()
