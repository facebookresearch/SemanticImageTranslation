# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
from pathlib import Path
from .vqc_core import *
from types import SimpleNamespace as nspace
import torchvision.transforms as T

mean = lambda x:sum(x)/len(x)
resize = lambda n:T.Compose([T.Resize(n),
                             T.CenterCrop(n),
                             T.ToTensor()])

class FlexitEditer:
    names = ['RN50x4', 'ViT-B/32', 'RN50']
    img_size = 288
    lr = 0.05
    lr_increment = 0
    max_iter = 256
    n_augment = 8
    device = 'cuda:0'
    gradient_filtering = 0.0
    comment = 'no lpips'
    lpips_opt_steps = 0
    lpips_weight = 0.2
    znorm_loss = 'l1l2'
    znorm_weight = 0.05
    lbd = 1
    normalize_gradient = True
    normalize_dv = False
    optimizer = 'sgd'
    latent_space = 'vqgan'
    erasing = False
    im0_coef = 0.0
    source_coef = 0.4 # coef for removing source !
    target_coef = 1.0
    
    args = nspace(**locals())
    
    def __init__(self, **kwargs):
        args = self.args
        self.args.__dict__.update(**kwargs)
                
        # init networks
        self.clip_net = CLIPS(names=args.names, device=args.device, erasing=args.erasing)
        self.lpips_net = LPIPS(device = args.device)
        
        if args.latent_space == 'vqgan':
            vq_net = VQGAN().to(args.device)
            self.image_encoder = vq_net.encode
            self.image_decoder = vq_net.decode
            self.args.optimizer = 'sgd' 
            
        elif args.latent_space == 'icgan':
            self.args.optimizer = 'adam' 
            self.args.lr = 5e-4
            self.args.img_size = 256
            self.args.znorm_loss = 'l2'
            
            from src.icgan_main import ICGAN_Encoder, ICGAN_Decoder
            self.image_decoder = ICGAN_Decoder()
            self.image_encoder = ICGAN_Encoder(decoder=self.image_decoder)
            
        elif args.latent_space == 'pixels':
            self.image_encoder = lambda x: x
            self.image_decoder = lambda x: x

    def log(self, **kwargs):
        for k, v in kwargs.items():
            self.history.setdefault(k, [])
            self.history[k].append(v)
        
        
    def __call__(self, img, txt1, txt2):
        # call this to initialize img and txt1, txt2
        
        args = self.args
        
        with torch.no_grad():
            self.img0 = resize(args.img_size)(img).to(args.device)
            self.z0 = self.image_encoder(self.img0).cpu().detach()

            #get target
            self.E_I0 = E_I0 = self.clip_net.encode_image(self.img0, ncuts=0)
            self.E_S, self.E_T = E_S, E_T =  self.clip_net.encode_text([txt1, txt2])
            

            self.tgt = args.target_coef * E_T + args.im0_coef * E_I0 - args.source_coef * E_S
                
            self.imgt = None
            
        self.lr = args.lr
        
        self.z = self.z0.clone().detach()
        self.z.requires_grad = True
        self.z.retain_grad()
        
        if args.optimizer == 'adam':
            self.opt = torch.optim.Adam([self.z], lr=args.lr, betas=(0.5, 0.8))
            
        elif args.optimizer == 'lbfgs':
            self.opt = torch.optim.LBFGS([self.z])
            
        self.history = {}
        
        img_hist = {}
        
        for i in range(args.max_iter+1):
            pil_im, _ = self.step()
            if i in [16, 32, 64, 160]:
                img_hist[i] = pil_im
        
        return img_hist, self.history
        
    def step(self):
        args = self.args
        history = self.history

        # loss z
        z = self.z

        noise = 1e-6 * torch.randn_like(z) # for differentiability
        if args.znorm_loss == 'l1l2':
            loss_z = (z - self.z0 + noise).pow(2).sum(1).sqrt().sum() / self.z0.pow(2).sum(1).sqrt().sum().detach()
        elif args.znorm_loss == 'l1':
            loss_z = (z - self.z0 + noise).abs().sum() / self.z0.abs().sum().detach()
        elif args.znorm_loss == 'l2':
            loss_z = (z - self.z0 + noise).pow(2).sum().sqrt() / self.z0.pow(2).sum().sqrt().detach()


        self.imgt = dec = self.image_decoder(z)

        loss_lpips = self.lpips_net(dec, self.img0)
        pred = self.clip_net.encode_image(dec, ncuts=args.n_augment)
        
        loss_clip  = - (pred @ self.tgt.T).flatten().reduce(mean)


        signal = loss_clip  + args.lpips_weight * loss_lpips + args.znorm_weight * loss_z

        signal.backward(retain_graph=True)
        
        self.log(loss=signal.item(),
                 loss_z=loss_z.item(),
                loss_clip=loss_clip.item(),
                loss_lpips=loss_lpips.item(),
                 sim_source=(pred @ self.E_S.T).reduce(mean).item(),
                 sim_target =(pred @ self.E_T.T).reduce(mean).item())
        
        g = z.grad[0]
        if args.normalize_gradient:
            g /= g.norm(p=2)
            
        if args.gradient_filtering:
            g_n = g.pow(2).sum(1, keepdim=True).sqrt()[0, 0] # 16 x 16
            g_n2 = (g_n - g_n.min())/(g_n.max() - g_n.min())
            g[:, :, g_n2<args.gradient_filtering] = 0

        if args.optimizer == 'sgd':
            self.z = z.detach() - self.lr * g * self.z0.norm().item()
            self.z.requires_grad = True
            self.z.retain_grad()

        else:
            self.opt.step()
        
        if args.latent_space == 'pixels':
            with torch.no_grad():
                self.z.clamp_(0,1)
        
        self.lr += args.lr_increment
        
        return T.ToPILImage()(dec.detach().squeeze(0)), g