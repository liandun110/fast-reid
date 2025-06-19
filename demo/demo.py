# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
import os
import sys

import torch.nn.functional as F
import cv2
import numpy as np
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager

from predictor import FeatureExtractionDemo

# import some modules added in project like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        default='configs/Market1501/mgn_R50-ibn.yml',
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--seq-path",
        default='datasets/yisuo/人脸追踪01/',
        help="Base sequence path containing person_crops folder"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'datasets/market_mgn_R50-ibn.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    # Set up paths
    seq_path = args.seq_path
    print('为视频{}提取特征……'.format(seq_path))
    input_dir = os.path.join(seq_path, 'person_crops')
    output_dir = os.path.join(seq_path, 'reid_features')
    
    PathManager.mkdirs(output_dir)
    
    # Get all image files from input directory
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg')) + \
                 glob.glob(os.path.join(input_dir, '*.png')) + \
                 glob.glob(os.path.join(input_dir, '*.jpeg'))
    
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")
    
    # Process each image
    for path in tqdm.tqdm(image_paths, desc="Extracting features"):
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read image {path}, skipping")
            continue
            
        feat = demo.run_on_image(img)
        feat = postprocess(feat)
        
        # Save feature with same name as input image but .npy extension
        base_name = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(output_dir, f"{base_name}.npy"), feat)