from models.PageMaker import main
import argparse

parser = argparse.ArgumentParser(description="Display results of monet2photo in a webpage")
parser.add_argument('--refresh', action='store_true',
                        help='run neural network to overwrite input and output images')
parser.add_argument('--dataroot', default='../data/monet2photo',
                        help='path to dataset (must have testA testB as subfolders)')
parser.add_argument('--ckpt', default='./checkpoints/original_cyclegan.pth',
                        help='generator checkpoint path')

args, _ = parser.parse_known_args()
main(args)