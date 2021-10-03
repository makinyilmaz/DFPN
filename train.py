import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import glob
import os
import itertools
import warnings
warnings.filterwarnings('ignore')
import imageio
from natsort import natsorted
from skimage.measure import compare_ssim,compare_psnr
import logging
import time
import random
from model import m



logging.basicConfig(filename="pred.log",level=logging.INFO)

device = torch.device("cuda")

def video_list(path="/home/ml/Akin/UCF/"):
    videos = natsorted(glob.glob(path+"*"))    
    return videos


def train_data_loader(video_list):

    batch_im_list = []
    for video in video_list:
        im_list = natsorted(glob.glob(video+"/*"))
        batch_im_list.append(im_list)
    return batch_im_list



def test_data_loader(path="/home/ml/Akin/mpeg/"):
    folders = natsorted(glob.glob(path+"*"))
    for f in folders:
        im_list = natsorted(glob.glob(f+"/*"))
        yield(im_list)


def float_to_uint8(image):
    clip = np.clip(image,-1,1)
    original_range = np.round((clip*127.5)+127.5)
    im_uint8 = np.uint8(original_range)
    return im_uint8


def prepare_train_data(gop_video_batch):

    X_train = []
    Y_train = []
    
    patch_size = 128
    size = 5

    for gop_ims in gop_video_batch:
    
        length = len(gop_ims)        
        s = random.randint(0, length - size)
        gop_split = gop_ims[s:s+size]
        
        sample_im = imageio.imread(gop_split[0], as_gray=True)
        
        x = random.randint(0, sample_im.shape[1] - patch_size)
        y = random.randint(0, sample_im.shape[0] - patch_size)
        
        for k in range(size):

            img = imageio.imread(gop_split[k], as_gray=True)
            img_cropped = img[y:y+patch_size,x:x+patch_size]
            (w,h) = img_cropped.shape
            img_cropped = img_cropped.reshape(1,w,h)
            
            if k == 0:
                img_concat = img_cropped
            else:
                img_concat = np.concatenate((img_concat,img_cropped),axis=0)
                                        
        X_train.append(img_concat)
    
    X = np.array(X_train)
    X_train = X.copy()[:,:-1]
    Y_train = X.copy()[:,-1:]
    
    
    return X_train,Y_train


def prepare_test_data(test_im_list):
    X_test = []
    Y_test = []
    
    size = 4
    
    for k in range(len(test_im_list)-size):
        x_test_array = []
        for i in range(size):
            
            x_image = imageio.imread(test_im_list[k+i])
            (nw,nh) = x_image.shape
            x_test_array.append(x_image)
        
        X_test.append(x_test_array)
        y_image = imageio.imread(test_im_list[k+i+1])
        Y_test.append(y_image.reshape(1,nw,nh))
    
    return np.array(X_test),np.array(Y_test)


def calculate_loss(out,real):
    loss = torch.mean(torch.abs(out-real))
    return loss

def normalize(tensor):
    norm = (tensor-127.5)/127.5
    return norm

def train_one_step(train_ims,model,optimizer):
    
    a = 20000
    b = 1
    alpha = a/(a+b)
    beta = b/(a+b)
    
    model = model.train()

    X_train,Y_train = prepare_train_data(train_ims)
    
    inp,real = normalize(torch.from_numpy(X_train).to(device).float()),normalize(torch.from_numpy(Y_train).to(device).float())
            
    out, norm = model.forward(inp)
    loss = alpha*calculate_loss(out,real) + beta*norm
                
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        

    return loss.item()

def test_and_save(model):
    with torch.no_grad():
        
        model.eval()

        total_test_loss = 0
        total_test_psnr = 0

        test_im_batch = test_data_loader()

        num_of_test_videos = 0
        folder_names = ["coastguard","container","football","foreman","garden","hall_monitor","mobile","tennis"]
        folder = 0

        total_frames = 0

        time_start = time.time()
        for test_ims in test_im_batch:
            X_test,Y_test = prepare_test_data(test_ims)
            
            (m,nc,nw,nh) = Y_test.shape

            video_psnr = 0
            video_ssim = 0

            d = 5

            for frame in range(m):
                inp = normalize(torch.from_numpy(X_test[frame:frame+1]).to(device).float())
                real = normalize(torch.from_numpy(Y_test[frame:frame+1]).to(device).float())


                out, norm = model.forward(inp)

                uint8_real = float_to_uint8(real.cpu().numpy()[0,0])
                uint8_out = float_to_uint8(out.cpu().detach().numpy()[0,0])
                psnr = compare_psnr(uint8_real,uint8_out,data_range=255)
                video_psnr += psnr

                d += 1
                total_frames += 1

            num_of_test_videos += 1

            average_video_psnr = video_psnr/X_test.shape[0]
            total_test_psnr += average_video_psnr


            folder += 1
        time_end = time.time()

        average_test_psnr = total_test_psnr / num_of_test_videos

        test_time = time_end - time_start

        fps = total_frames / test_time
    
    return average_test_psnr,fps



def save_model(model,optimizer):
    state = {
        "state_dict" : model.state_dict(),
        "optimizer" : optimizer.state_dict()
    }
    torch.save(state,"res_sc_mae.pth")


def load_pre_model(m, path, requires_grad):
    pre_dict = torch.load(path, map_location=lambda storage, loc: storage)["state_dict"]
    m_dict = m.state_dict()
    m_dict.update(pre_dict)
    m.load_state_dict(m_dict)
    for p in m.parameters():
        p.requires_grad = requires_grad
    return m


def main():
    
    np.random.seed(1)
    torch.manual_seed(2)
    random.seed(3)
    
    
    total_train_step = 500000
    train_step = 2000
    lr_step = 100000
    
    learning_rate = 1.e-4
           
    model = m.Model()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info("number of parameters: "+str(params))

    model = model.to(device).float()
    

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    average_train_loss = 0    

    best_test_psnr = 0

    all_videos = video_list()
    
    logging.info("train video samples: "+str(len(all_videos)))
    logging.info("**********")

    batch_size = 8
    
    time_start = time.time()
    for minibatch_processed in range(1,total_train_step+1):
        
        batch_video = random.sample(all_videos,batch_size)
        train_im_list = train_data_loader(batch_video)
        train_step_loss = train_one_step(train_im_list,model,optimizer)
        average_train_loss += train_step_loss
        
        print(minibatch_processed, train_step_loss)
                                        
        if  minibatch_processed % train_step == 0:
            average_test_psnr,test_fps = test_and_save(model)
            if average_test_psnr > best_test_psnr:
                best_test_psnr = average_test_psnr
                logging.info("NEW BEST !!!")
                
                save_model(model,optimizer)
            
            logging.info("number of iterations: "+str(minibatch_processed))
            logging.info("learning rate: "+str(learning_rate))
            logging.info("train_loss: "+str(average_train_loss/train_step))
            logging.info("-----")
            logging.info("fps: "+str(test_fps))
            logging.info("test psnr: "+str(average_test_psnr))
            logging.info("best psnr: "+str(best_test_psnr))
            
            logging.info("****************")

            average_train_loss = 0

        if minibatch_processed % lr_step == 0:
            learning_rate /= 2
            for g in optimizer.param_groups:
                g["lr"] = learning_rate
            
            
        
    time_end = time.time()
    training_time = time_end-time_start
    day = training_time // (24 * 3600)
    training_time = training_time % (24 * 3600)
    hour = training_time // 3600
    training_time %= 3600
    minutes = training_time // 60
    training_time %= 60
    seconds = training_time
    logging.info("day:hour:minute:second-> %d:%d:%d:%d" % (day, hour, minutes, seconds))




main()

