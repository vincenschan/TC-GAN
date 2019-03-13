import numpy as np
from argparse import ArgumentParser
import os
from train import Train

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode",dest="mode",
                        help="chose mode to run the code, train or test",
                        metavar="MODE", default="train")
    parser.add_argument("--dir",dest="dir",
                        help="chose dir to save files while training",
                        default="./train/")
    parser.add_argument("--data",dest="data",
                        help="chose dir to read training data",
                        default="./data/")
    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists(options.data):
        print("Error, data dir is not exists.")
    if not os.path.exists(options.dir):
        os.makedirs(options.dir)
    if options.mode == "train":
        Train().train()



if __name__ == "__main__":
    main()