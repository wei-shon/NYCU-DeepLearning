import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image , make_grid

import os


def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
# gt = ground truth
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def pred(validate_seq, validate_cond, modules, args, device):
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()

    # ht_1 means ht-1
    seq = []
    seq.append(validate_seq[0])
    h_seq = [modules['encoder'](validate_seq[i]) for i in range(args.n_past+args.n_future)]

    for i in range(1, args.n_past + args.n_future):
        # encoder & check teacher forcing
        if i < args.n_past or args.last_frame_skip:
            ht_1 , skip = h_seq[i-1]
        else:
            ht_1 , _ = h_seq[i-1]
        
        ht_1 = ht_1.detach()

        if i < args.n_past:
            z_t, _, _ = modules['posterior'](h_seq[i][0])
            modules['frame_predictor'](torch.cat([validate_cond[i-1], ht_1, z_t], 1)) 
            seq.append(validate_seq[i])
        else:
            z_t = torch.randn(args.batch_size, args.z_dim).to(device)
            g_t = modules['frame_predictor'](torch.cat((validate_cond[i-1] , ht_1 , z_t) , 1)).detach()
            pred = modules['decoder']( [g_t , skip]).detach()
            h_seq[i] = modules['encoder'](pred)
            seq.append(pred)

    return torch.stack(seq,dim=0)


def plot_pred(validate_seq, validate_cond, modules, epoch, args):
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'

    validate_seq = validate_seq.to(device)
    validate_cond = validate_cond.to(device)
    validate_seq = validate_seq.permute(1,0,2,3,4)
    validate_cond = validate_cond.permute(1,0,2)
    pred_sequence = pred(validate_seq, validate_cond, modules, args, device)
    images_to_gif = []
    pred_image_tensor_list = []
    gt_image_tensor_list = []

    sample = pred_sequence[ : , 0 , : , : , :]
    gt = validate_seq[: , 0 , : , : , :]

    for i in range(sample.shape[0]):
        image_name = "epoch-{}-pred-{}.png".format(epoch , i)
        save_image(sample[i] , image_name)
        images_to_gif.append(imageio.imread(image_name))
        pred_image_tensor_list.append(sample[i])
        os.remove(image_name)
        gt_image_tensor_list.append(gt[i])
    
    pred_grid = make_grid(pred_image_tensor_list , nrow = sample.shape[0])
    gt_grid = make_grid(gt_image_tensor_list , nrow = sample.shape[0])
    save_image(pred_grid , "{:s}/gen/epoch-{}-pred_grid.png".format(args.log_dir,epoch))
    save_image(gt_grid , "{:s}/gen/epoch-{}-gt_grid.png".format(args.log_dir,epoch))
    imageio.mimsave("{:s}/gen/epoch-{}-GIF.gif".format(args.log_dir,epoch) , images_to_gif , format= 'GIF')

