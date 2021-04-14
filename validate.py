
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import glob
import os
import itertools
import warnings
warnings.filterwarnings('ignore')
import imageio
from natsort import natsorted
import logging
import time
from model import m



logging.basicConfig(filename="result.log",level=logging.INFO)

device = torch.device("cuda")

def test_data_loader(path="mpeg/"):
    folders = natsorted(glob.glob(path+"*"))
    for f in folders:
        im_list = natsorted(glob.glob(f+"/*"))
        yield(im_list)


def float_to_uint8(image):
    clip = np.clip(image,-1,1)
    original_range = np.round((clip*127.5)+127.5)
    im_uint8 = np.uint8(original_range)
    return im_uint8


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

def normalize(tensor):
    norm = (tensor-127.5)/127.5
    return norm


def test_and_save(model):
    with torch.no_grad():
        
        model.eval()


        test_im_batch = test_data_loader()

        num_of_test_videos = 0
        folder_names = ["coastguard","container","football","foreman","garden","hall_monitor","mobile","tennis"]
        
        folder = 0
        total_frames = 0

        time_start = time.time()
        for test_ims in test_im_batch:
            X_test,Y_test = prepare_test_data(test_ims)
            
            (m,nc,nw,nh) = Y_test.shape


            d = 5
            os.makedirs("Results/"+folder_names[folder],exist_ok=True)

            for frame in range(m):
                inp = normalize(torch.from_numpy(X_test[frame:frame+1]).to(device).float())
                real = normalize(torch.from_numpy(Y_test[frame:frame+1]).to(device).float())


                out, norm = model.forward(inp)

                uint8_real = float_to_uint8(real.cpu().numpy()[0,0])
                uint8_out = float_to_uint8(out.cpu().detach().numpy()[0,0])

                save_path = "Results/"+folder_names[folder]+"/frame"+str(d)+".png"
                imageio.imsave(save_path,uint8_out)
                d += 1
                total_frames += 1
                
            folder += 1
            
        time_end = time.time()

        test_time = time_end - time_start

        fps = total_frames / test_time
    
    return fps


def main():
    path = "dfpn.pth"
    
    model = m.Model()
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)["state_dict"])
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device).float()
        
    test_fps = test_and_save(model)
            
    logging.info("fps: "+str(test_fps))

        

main()

