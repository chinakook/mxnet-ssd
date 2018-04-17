from __future__ import print_function
import sys, os
import argparse
import subprocess
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))
from dataset.dgdb import dgdb
from dataset.concat_db import ConcatDB
import mxnet

def load_dg(root_path):
    imdbs = []
    #for s, y in zip(image_set, year):
    imdbs.append(dgdb("dg", ["obj"],root_path))
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
    parser.add_argument('--no-shuffle', dest='shuffle', help='shuffle list',
                        action='store_false')
    parser.add_argument('--num-thread', dest='num_thread', type=int, default=1,
                        help='number of thread to use while runing im2rec.py')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    db = load_dg(args.root_path)
    print("saving list to disk...")
    db.save_imglist(args.target) #, root=args.root_path)

    print("List file {} generated...".format(args.target))

    cmd_arguments = ["python",
                    os.path.join(mxnet.__path__[0], 'tools/im2rec.py'),
                    os.path.abspath(args.target), os.path.abspath(args.root_path),
                    "--pack-label", "--num-thread", str(args.num_thread)]

    if not args.shuffle:
        cmd_arguments.append("--no-shuffle")

    subprocess.check_call(cmd_arguments)

print("Record file {} generated...".format(args.target.split('.')[0] + '.rec'))