from __future__ import print_function
import sys, os
import argparse
import subprocess
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))
from dataset.ylbdb import ylbdb
from dataset.concat_db import ConcatDB

def load_ylb(root_path):
    """
    wrapper function for loading pascal voc dataset

    Parameters:
    ----------
    image_set : str
        train, trainval...
    year : str
        2007, 2012 or combinations splitted by comma
    devkit_path : str
        root directory of dataset
    shuffle : bool
        whether to shuffle initial list

    Returns:
    ----------
    Imdb
    """

    imdbs = []
    #for s, y in zip(image_set, year):
    imdbs.append(ylbdb("ylb", root_path, is_train=True))
    if len(imdbs) > 1:
        return ConcatDB(imdbs, False)
    else:
        return imdbs[0]

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lists for dataset')
    parser.add_argument('--dataset', dest='dataset', help='dataset to use',
                        default='pascal', type=str)
    parser.add_argument('--year', dest='year', help='which year to use',
                        default='2007,2012', type=str)
    parser.add_argument('--set', dest='set', help='train, val, trainval, test',
                        default='trainval', type=str)
    parser.add_argument('--target', dest='target', help='output list file',
                        default=os.path.join(curr_path, '..', 'train.lst'),
                        type=str)
    parser.add_argument('--root', dest='root_path', help='dataset root path',
                        default=os.path.join(curr_path, '..', 'data', 'VOCdevkit'),
                        type=str)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle list',
                        type=bool, default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    db = load_ylb(args.root_path)
    print("saving list to disk...")
    db.save_imglist(args.target) #, root=args.root_path)

    print("List file {} generated...".format(args.target))

    subprocess.check_call(["python",
        os.path.join(curr_path, "/home/dingkou/dev/incubator-mxnet/tools/im2rec.py"),
        os.path.abspath(args.target), os.path.abspath(args.root_path),
        "--shuffle", str(int(args.shuffle)), "--pack-label", "1"])

    print("Record file {} generated...".format(args.target.split('.')[0] + '.rec'))
