import os
import time
import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms

import pandas as pd


if __name__ == "__main__":
    cap = cv2.VideoCapture("../ff-datasets/ff_jackson_test_30fps.mp4")
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    subtractor = cv2.bgsegm.createBackgroundSubtractorMOG() # use the first frame as a template
    sparsity_mask = []
    sparsity_conv = []
    t_orig = []
    t_sparse = []
    frame_num = 0
    
    batch_size = 16
    #Using GPU
    device = torch.device('cpu')
    model= nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False).to(device)
    totensor = transforms.ToTensor()
    
    # Read until video is completed
    while(cap.isOpened()):
        print("frame number:",frame_num)
        
        
        #Capture frame-by-frame
        ret, frame = cap.read()
    
        if ret == True:
            # Sparsity of the original sparse image (mask) 
            mask = subtractor.apply(frame) # mask out all new pixels from the template
            th = cv2.threshold(mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]  # refine      
            th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)  # refine       
            dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2) # refine   
            
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #Convert Frame from BGR2RGB
            res = cv2.bitwise_and(frame,frame,mask = dilated) #Mask frame out
            orig_img = totensor(frame).unsqueeze(0).to(device)
            sparse_img = totensor(res).unsqueeze(0).to(device)
            
            if frame_num % batch_size == 0:
                orig_batch = orig_img
                sparse_batch = sparse_img
                
            else:
                orig_batch = torch.cat((orig_batch, orig_img), 0)
                sparse_batch = torch.cat((sparse_batch, sparse_img), 0)
                if frame_num % (batch_size - 1) == 0:    
                
                    # Conv2d on orig image
                    with torch.no_grad():
                        start = time.time()
                        model(orig_batch)
                        end = time.time()
                        t_orig.append(end - start)
            
                    # Conv2d on sparse image
                    with torch.no_grad():
                        start = time.time()
                        out = model(sparse_batch)
                        end = time.time()
                        t_sparse.append(end - start)
                
                    # Sparsity of the sparse image after conv2d
                    out_sparse = out.cpu().squeeze(0).numpy()
                    before_sparse = sparse_batch.cpu().squeeze(0).numpy()
                    sparsity_conv.append((1-np.count_nonzero(out_sparse)/out_sparse.size)*100)
                    sparsity_mask.append((1-np.count_nonzero(before_sparse)/before_sparse.size)*100)
            
            if frame_num % 1000 == 0:
                df = pd.DataFrame({'sparsity_before':sparsity_mask ,'sparsity_after':sparsity_conv,'t_sparse': t_sparse,'t_orig':t_orig},columns=['sparsity_before','sparsity_after','t_sparse','t_orig'])
                df.to_csv('Sparsity_ff_jackson_test_30fps.csv')
                       
        # Break the loop
        else:
            break
        frame_num = frame_num + 1
    # When everything done, release the video capture object
    cap.release()
     
    df = pd.DataFrame({'sparsity_before':sparsity_mask ,'sparsity_after':sparsity_conv,'t_sparse': t_sparse,'t_orig':t_orig},columns=['sparsity_before','sparsity_after','t_sparse','t_orig'])
    df.to_csv('Sparsity_ff_jackson_test_30fps.csv')
    
    
        
