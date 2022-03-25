# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import argparse
from pathlib import Path
import torchvision.transforms as T
import json
from PIL import Image
import lpips
import pandas as pd
import numpy as np
import clip 
import os
import yaml
from utils.io import load, dump
from utils.torch_utils import *

resize = lambda n:T.Compose([
    T.Resize(n),
    T.CenterCrop(n),
    T.ToTensor(),
])

torch.set_grad_enabled(False)

GLOBAL = load('global.yaml')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, default="", help="folder for images")
    parser.add_argument("-d", "--device", type=str, default='cuda:0', help="cuda id")
    parser.add_argument("-m", "--metric", type=str, default="", help="which metric to run (by default: all)")
    parser.add_argument("-b", "--block", type=int, default=3, help="Inception block")
    
    args, unknown = parser.parse_known_args()
    args.folder = Path(args.folder)
      
    args.metrics = {'l1':True,
                    'deit': True,
                    'lpips': True,
                    'fid': True}
    if args.metric:
        args.metrics = {k: (k == args.metric) for k in args.metrics}
        
    return args
        

def evaluate(args):    
    out_path = args.folder.parent / (args.folder.name + '_analysis.pt')
    # if there is an existing analysis file, load it, else start from scratch
    output = load(out_path) if out_path.exists() else {}
        
    #evaluate only on images present in the folder
    img_paths = [x for x in args.folder.iterdir() if x.name[-3:] in ('png', 'jpg')]
    present_ids = [int(x.name[:-4]) for x in img_paths]
    imgs = [load(x) for x in img_paths]
    imgt = torch.stack([resize(256)(im) for im in imgs])
    
    
    #fetch evaluation data
    df = load('dataset/queries.csv')
    class_ids = list(sorted(set(df.target_id))) # all classes in transformation requests
    source_ids = torch.tensor(list(df.loc[present_ids].source_id))
    target_ids = torch.tensor(list(df.loc[present_ids].target_id))
    set_target_ids = set(target_ids.tolist())
    

    if args.metrics['deit']:
        imgt_classif = torch.stack([resize(384)(im) for im in imgs])
        
        in_norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        model = torch.hub.load('facebookresearch/deit:main', 
                               'deit_base_distilled_patch16_384', 
                               pretrained=True,
                              verbose=False).to(args.device)
        model.training = False
        
        preds = model.batch_forward(imgt_classif, batch_size=32)
        pred_idx = torch.Tensor([class_ids[i] for i in preds[:, class_ids].argmax(-1)])

        # compute accuracy
        acc = (pred_idx == target_ids).float().mean().item()
        same_acc = (pred_idx == source_ids).float().mean().item()
        
        output['%Correctly edited (DeiT)'] = 100*acc
        output['% Unedited (DeiT)'] = 100*same_acc
        output['%Wrongly edited (DeiT)'] = 100*(1 - acc - same_acc)
    

    # compute LPIPS w.r.t. input images
    if args.metrics['lpips'] or args.metrics['l1']:       
        input_paths = [Path(GLOBAL['IMAGENET_ROOT']) / df.loc[i].path for i in present_ids]
        input_imgt = torch.stack([resize(256)(load(x)) for x in input_paths])
        
    if args.metrics['lpips']:
        lpnet = lpips.LPIPS(net='alex').to(args.device)
        lpips_dist = lambda x, y: lpnet.forward(x.to(args.device), 
                                                y.to(args.device), 
                                                normalize=True).detach().cpu()
        sim_scores = torch.cat([lpips_dist(imgti, input_imgti) for imgti, input_imgti 
                                in zip(imgt.split(32), input_imgt.split(32))])
        output['Mean LPIPS distance'] = 100*sim_scores.mean().item()


    if args.metrics['l1']:
        l1_dists = (input_imgt - imgt).abs().reshape(imgt.shape[0], -1).mean(1)
        output['Mean L1 distance'] = l1_dists.mean().item()
            
    
    # compute inception scores
    if args.metrics['fid']:
        inception_scores = load('dataset/inception_stats_256.pt')
        dataset_m, dataset_s, dataset_cw = inception_scores['mean'], inception_scores['std'], inception_scores['classwise']
        
        from utils.inception import InceptionV3
        inception_net = InceptionV3([3]).to(args.device).eval()
        preds = inception_net.batch_forward(imgt, 32).squeeze()
        
        id2mean = {i: preds[target_ids == i].mean(0) for i in set_target_ids}
        id2std = {i: preds[target_ids == i].std(0).nan_to_num(0) for i in set_target_ids}
        
        inception_mean_dist = [(id2mean[i] - dataset_cw[i]['mean']).pow(2).sum() for i in set_target_ids]
        inception_std_dist = [(id2std[i] - dataset_cw[i]['mean']).pow(2).sum() for i in set_target_ids]
        
        output['SFID(mean) score'] = (preds.mean(0) - dataset_m).pow(2).sum().item()
        output['CSFID(mean) score'] = torch.stack(inception_mean_dist).mean().item()
        
        # we don't use those in the paper
        
        #output['SFID(std) score'] = (preds.std(0) - dataset_s).pow(2).sum().item()
        #output['CSFID(std) score'] = torch.stack(inception_std_dist).mean().item()
        
        # classwise score for dissection
        if False:
            output['CSFID(mean) Classwise'] = dict(zip(set_target_ids, inception_mean_dist.tolist()))
    
    torch.save(output, out_path)
    return output
    
                            
if __name__ == "__main__":
    args = get_args()
    print(f'Starting evaluation of folder {str(args.folder)}')
    output = evaluate(args)
    for k, v in output.items():
        if isinstance(v, float) or isinstance(v, int):
            print(f'{k}: {v:.3f}')
        else:
            print(f'{k}: {v}')    