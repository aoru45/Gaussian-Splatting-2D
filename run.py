#!/usr/bin/env python
# coding=utf-8
import torch
import torchvision
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
class GS2D:
    def __init__(self,
                 img_size = (100, 100, 3),
                 num_epoch = 20,
                 num_iter_per_epoch = 100,
                 num_samples = 300,
                 num_max_samples = 3000,
                 sigma_thre = 10,
                 grad_thre = 0.06,
                 device = "cuda"
                 ):
        self.img_size = img_size
        self.num_epoch = num_epoch
        self.num_iter_per_epoch = num_iter_per_epoch
        self.num_samples = num_samples
        self.device = device
        self.sigma_thre = sigma_thre
        self.grad_thre = grad_thre
        self.num_max_samples = num_max_samples
        
        h, w = self.img_size[:2]
        xx,yy = torch.arange(w), torch.arange(h)
        x,y = torch.meshgrid(xx,yy,indexing="xy")#(h,w)
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        
    def draw_gaussian(self, sigma, rho, mean, color, alpha):
        # sigma(n,2)
        # rho(n,1)
        # mean(n,2)
        # color(n,3)
        # alpha(n,1)
        r = rho.view(-1,1,1)
        sx = sigma[:, :1, None] 
        sy = sigma[:, -1:, None]
        dx = self.x.unsqueeze(0) - mean[:, 0].view(-1,1,1)
        dy = self.y.unsqueeze(0) - mean[:, 1].view(-1,1,1)
        v = -0.5 * (((sy*dx)**2 + (sx*dy)**2)  - 2*dx*dy*r*sx*sy) / (sx**2 * sy**2 * (1-r**2) + 1e-8)

        # normaliza to 1 without scale
        v = torch.exp(v) #/ (2 * torch.pi * sx * sy + 1e-8)  # (n,h,w)
        #scale = torch.max(v.view(v.size(0), -1),dim=1)[0]
        #v = v / scale.view(-1,1,1)
        v = v * alpha.view(-1,1,1)
        img = torch.sum(v.unsqueeze(1) * color.view(-1,3,1,1), dim = 0)
        return torch.clamp(img, 0, 1)
    def random_init_param(self):
        sigma = torch.rand(size = (self.num_samples, 2)) - 3
        rho = torch.rand(size = (self.num_samples, 1)) * 2 
        mean = torch.atanh(torch.rand(size = (self.num_samples, 2))*2 - 1)
        color = torch.atanh(torch.rand(size = (self.num_samples, 3)))
        alpha = torch.zeros(size = (self.num_samples, 1))-0.01
        w = torch.cat([sigma, rho, mean, color, alpha], dim =1).to(self.device)
        return nn.Parameter(w)

    def parse_param(self, w):
        # w (b, 13)
        sigma  = (torch.sigmoid(w[:, 0:2])) * torch.tensor(self.img_size[:2][::-1]).to(self.device) * 0.25
        rho   = torch.tanh(w[:, 2:3])
        mean  = (0.5*torch.tanh(w[:, 3:5]) + 0.5) * torch.tensor(self.img_size[:2][::-1]).to(self.device)
        color = 0.5*torch.tanh(w[:, 5:8])+0.5
        alpha = 0.5*torch.tanh(w[:, 8:9]) +0.5
        return sigma, rho, mean, color, alpha
    def update_w(self, w_old, _grad):
        size_tensor = torch.tensor(self.img_size[:2][::-1]).to(self.device)
        #grad_norm = torch.norm(_grad[:, 3:5] ,dim=1,p=2) # (n, )
        # calculate the real gradient with tanh
        grad_norm = torch.norm(2. * _grad[:, 3:5] / (1-torch.tanh(w_old.data[:, 3:5])**2),dim=1,p=2) # (n, )
        sigma = (torch.sigmoid(w_old.data[:, 0:2])) * size_tensor * 0.25 
        #alpha = 0.5*torch.tanh(w_old.data[:, 8]) +0.5
        sigma_norm = torch.norm(sigma, dim=1,p=2) # (n,)

        grad_mask = grad_norm > self.grad_thre
        sigma_mask = sigma_norm > self.sigma_thre
        #alpha_mask = alpha > 0.01

        w_save = w_old[~grad_mask].data
        # grad large and sigma large
        w_scale = w_old[grad_mask & sigma_mask].data
        w_scale[:, :2] = torch.log(torch.sigmoid(w_scale[:, :2])) - torch.log(1.6-torch.sigmoid(w_scale[:, :2]))
        w_scale_copy = w_scale.clone()
        w_scale_copy[:,3:5] = w_scale_copy[:,3:5] - _grad[grad_mask & sigma_mask, 3:5]

        # grad large but sigma small
        w_split = w_old[grad_mask & (~sigma_mask)].data 
        w_split_copy = w_split.clone()
        # resample position TODO
        # w_split_copy[:, 3:5] = sigma[grad_mask & (~sigma_mask)] * torch.randn(w_split_copy.size(0),2).to(self.device) + (0.5*torch.tanh(w_split_copy[:, 3:5]) + 0.5) * size_tensor
        # w_split_copy[:, 3:5] = w_split_copy[:, 3:5] / size_tensor * 2 - 1
        # w_split_copy[:,3:5] = torch.atanh(w_split_copy[:,3:5])
        

        print(f"Before Update: {_grad.size(0)} Save: {w_save.size(0)} Split: {w_split.size(0)} Scale: {w_scale.size(0)}")
        del w_old
        w1 = torch.cat([w_save, w_scale, w_split])
        w2 = torch.cat([w_scale_copy, w_split_copy])
        w2 = w2[torch.randperm(w2.size(0))]
        # only support 3000 samples with 16GB GPU
        if w1.size(0) + w2.size(0) > self.num_max_samples:
            return torch.cat([w1, w2[:self.num_max_samples - w1.size(0)]])
        return torch.cat([w1, w2])
        

    def train(self, target):
        w = self.random_init_param()
        for epoch in range(self.num_epoch):
            torch.cuda.empty_cache()
            optimizer = torch.optim.AdamW([w], lr = 0.005)
            bar = tqdm(range(self.num_iter_per_epoch))
            for _iter in bar:
                optimizer.zero_grad()
                predicted = self.draw_gaussian(*self.parse_param(w))
                loss = nn.functional.l1_loss(predicted, target)
                loss.backward()
                optimizer.step()
                bar.set_description(f"[Ep@{epoch}] [Loss@{loss.item():.6f}]")

            _grad = w.grad.data
            optimizer.zero_grad()
            with torch.no_grad():
                w = self.update_w(w.detach(), _grad.detach())
                w = torch.nn.Parameter(w)
                #predicted_new = self.draw_gaussian(*self.parse_param(w))
                #pred_out = torchvision.utils.make_grid([predicted, predicted_new, target],nrow=3)
                pred_out = torchvision.utils.make_grid([predicted, target],nrow=2)
                torchvision.utils.save_image(pred_out, f"images/{epoch}.jpg")
                
        
def test():
    gs = GS2D(num_samples=1000,img_size=(256,256,3), device="cpu")
    w = gs.random_init_param()
    img = gs.draw_gaussian(*gs.parse_param(w))
    torchvision.utils.save_image(img,"out.png")

if __name__ == "__main__":
    #test()
    #exit()
    device = "cuda"
    img = Image.open("a.png").convert("RGB")
    tsfm = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(256,256)),
        torchvision.transforms.ToTensor()
    ])
    img = tsfm(img)
    h,w = img.size()[1:]
    gs = GS2D(num_epoch = 50,
                img_size = (h,w,3) ,
                device = device,
                sigma_thre=3,grad_thre=0.003,
                 num_iter_per_epoch = 300,
                 num_samples=1000)
    img = img.to(device)
    gs.train(img)
    
