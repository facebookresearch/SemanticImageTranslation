# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from pathlib import Path
from functools import partial
import os
from types import SimpleNamespace as nspace
import submitit
import yaml
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
import json

from src.flexit_editer import FlexitEditer

from utils.io import load, dump
from utils.load_config import load_config
from utils.submit import submit

IN_ROOT = Path(load('global.yaml')['IMAGENET_ROOT'])

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='exp_configs/final.yaml', help='config file')
    parser.add_argument("--test", action='store_true', default='dev', help="whether to use the test domain")
    parser.add_argument("--output", type=str, default='generated/default/', help='where to store images')
    parser.add_argument("--nsteps", type=int, default=256, help='# optimization steps')
    parser.add_argument("--ngpus", type=int, default=32, help='# GPUs for editing')
    parser.add_argument("--debug", action='store_true', help='Set to true to run on limited eval')
    
    args, unknown = parser.parse_known_args()
    
    if args.config:
        cfg_args, str_args = load_config(args.config)
        cfg_args.__dict__.update(args.__dict__) # override with command line args
        args = cfg_args
        
    args.output = Path(args.output)
        
    return args

def main(sample_df, args):
    transformer = args.editer()
    args.__dict__.update({k:v for k, v in transformer.args.__dict__.items() if not '__' in k})
    print(args)
    dump(args.__dict__, args.output / 'config.yaml')


    for i, l in sample_df.iterrows():
        img = load(IN_ROOT / l.path)
        out_imgs, *history = transformer(img, l.source, l.target) # dict or PIL Image
        out_imgs = out_imgs if isinstance(out_imgs, dict) else dict(images=out_images) # convert PIL Image
        for j, pil_im in out_imgs.items():
            (args.output / str(j)).mkdir(exist_ok=True)
            pil_im.save(args.output / f'{j}/{i}.png')

        if history:
            history[0]['source_id'] = l.source_id
            history[0]['target_id'] = l.target_id
            (args.output / 'logs').mkdir(exist_ok=True)
            torch.save(history, args.output / f'logs/{i}.pt')



if __name__ == '__main__':
    args = get_parser()
    args.output.mkdir(exist_ok=True)
    
    eval_df = load('dataset/queries.csv')
    eval_df = eval_df[eval_df.is_test == args.test]
    
    if args.debug:
        eval_df = eval_df.iloc[:args.ngpus]
        
    submit(main, eval_df, args, 
           folder=args.output.name, 
           ngpus=args.ngpus)
    