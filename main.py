import os
import numpy as np
from matplotlib import pyplot as plt
from Horn_Schunk.Horn_schunk import *
from Interpolation.interpolation import *
from Lukas_kanade.lukas_kanade import *
from Multiscale_Lukas_kanade.Multiscale_lukas_kanade import *
from Multiscale_Horn_schunk.multiscale_horn_schunk import *
import cv2


if __name__=="__main__":
    ## Hyper parameter
    lambada=2
    max_iter=400
    epsilon=0.001
    print("This is running")
    img0=cv2.imread("./Dataset/corridor/bt.000.pgm")
    img1=cv2.imread("./Dataset/corridor/bt.001.pgm")
    img2=cv2.imread("./Dataset/corridor/bt.002.pgm")
    img0=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ## Calculating flow
    forward_flow,forward_grad=horn_schunk_flow(img0,img2,lambada,max_iter,epsilon)
    #backward_flow,backward_grad=horn_schunk_flow(img2,img0,lambada,max_iter,epsilon)
    #out=warp_flow(img0,img2,forward_flow,forward_grad,backward_flow,backward_grad)
    #forward_pred = img0 / 255 + forward_flow[0] * forward_grad[0] + forward_flow[1] * forward_grad[1] + forward_grad[2]

    plt.subplot(2,2,1)
    plt.imshow(img0,cmap="gray")
    plt.subplot(2,2,3)
    plt.imshow(img1,cmap="gray")
    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap="gray")
    #forward_flow,forward_grad=lukas_kanade_flow(img0, img2, 9)
    #forward_flow, forward_grad = multiscale_lukas_kanade_flow(img0, img2, 11, 2)
    forward_flow, forward_grad = multiscale_horn_schunk_flow(img0, img2,lambada,max_iter,epsilon,3)
    forward_pred = (img0/255)+ (forward_flow[0] * forward_grad[0] + forward_flow[1] * forward_grad[1] + forward_grad[2])
    plt.subplot(2,2,4)
    plt.imshow(forward_pred,cmap="gray")
    plt.show()



