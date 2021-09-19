import os
import numpy as np
from matplotlib import pyplot as plt
from Horn_Schunk.Horn_schunk import *
from Interpolation.interpolation import *
from Lukas_kanade.lukas_kanade import *
from Multiscale_Lukas_kanade.Multiscale_lukas_kanade import *
from Multiscale_Horn_schunk.multiscale_horn_schunk import *
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



def warp(img,flow,grad):
    img = (img / 255) + (flow[0] * grad[0] + flow[1] * grad[1] + grad[2])
    return img

if __name__=="__main__":
    ## Hyper parameter
    lambada=2
    max_iter=400
    epsilon=0.001
    print("This is running")

    # Creating dataframe for scores
    forward_pred_scores = pd.DataFrame(
        columns=["horn_schunk_ssim", "horn_schunk_psnr", "multiscale_horn_schunk_ssim",
                 "multiscale_horn_schunk_psnr", "lukas_kanade_ssim", "lukas_kanade_psnr",
                 "multiscale_lukas_kanade_ssim", "multiscale_lukas_kanade_psnr"], index=list(range(2, 11, 2)))

    backward_pred_scores = pd.DataFrame(
        columns=["horn_schunk_ssim", "horn_schunk_psnr", "multiscale_horn_schunk_ssim",
                 "multiscale_horn_schunk_psnr", "lukas_kanade_ssim", "lukas_kanade_psnr",
                 "multiscale_lukas_kanade_ssim", "multiscale_lukas_kanade_psnr"], index=list(range(0, 10, 2)))

    interpolated_frame_scores = pd.DataFrame(
        columns=["horn_schunk_ssim", "horn_schunk_psnr", "multiscale_horn_schunk_ssim",
                 "multiscale_horn_schunk_psnr", "lukas_kanade_ssim", "lukas_kanade_psnr",
                 "multiscale_lukas_kanade_ssim", "multiscale_lukas_kanade_psnr"], index=list(range(1, 10, 2)))

    # Corridor Dataset
    for i in range(0,10,2):

        cor="./Dataset/corridor/"
        name1=str("bt.00")+str(i)+str(".pgm")
        name3=str("bt.00")+str(i+1)+str(".pgm")
        if i+2!=10:
            name2 = str("bt.00") + str(i+2) + str(".pgm")
        else:
            name2 = str("bt.0") + str(i + 2) + str(".pgm")

        path1=str(cor)+str(name1)
        path2=str(cor)+str(name2)
        path3=str(cor)+str(name3)
        img0=cv2.imread(path1)
        img2=cv2.imread(path2)
        img1=cv2.imread(path3)

        img0=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
        img2= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


        # Calculating flow using hornshunk
        forward_flow,forward_grad=horn_schunk_flow(img0,img2,lambada,max_iter,epsilon)
        backward_flow,backward_grad=horn_schunk_flow(img2,img0,lambada,max_iter,epsilon)


        # forward prediction
        img=warp(img0,forward_flow,forward_grad)
        temp="./Results/Horn_schunk_results/corridor/forward_prediction/forward_pred_"+str(i+2)+".png"
        cv2.imwrite(temp,img*255)
        forward_pred_scores.horn_schunk_ssim[i+2]=ssim(img2,img*255)
        forward_pred_scores.horn_schunk_psnr[i + 2] = psnr(img2, img * 255)


        # backward prediction
        img = warp(img2, backward_flow, backward_grad)
        temp = "./Results/Horn_schunk_results/corridor/backward_prediction/backward_pred_" + str(i) + ".png"
        cv2.imwrite(temp, img*255)
        backward_pred_scores.horn_schunk_ssim[i] = ssim(img0, img * 255)
        backward_pred_scores.horn_schunk_psnr[i] = psnr(img0, img * 255)

        #Interpolated
        img=interpolate1(img0,img2,forward_flow,backward_flow)
        temp = "./Results/Horn_schunk_results/corridor/interpolated_frame/interpolated_" + str(i+1) + ".png"
        cv2.imwrite(temp, img)
        interpolated_frame_scores.horn_schunk_ssim[i + 1] = ssim(img1, img)
        interpolated_frame_scores.horn_schunk_psnr[i + 1] = psnr(img1, img)




        # Calculating flow using multiscale horn schunk
        forward_flow, forward_grad = multiscale_horn_schunk_flow(img0, img2, lambada, max_iter, epsilon,4)
        backward_flow, backward_grad = multiscale_horn_schunk_flow(img2, img0, lambada, max_iter, epsilon,4)

        # forward prediction
        img = warp(img0, forward_flow, forward_grad)
        temp="./Results/Multiscale_Horn_schunk_results/corridor/forward_prediction/forward_pred_"+str(i+2)+".png"
        cv2.imwrite(temp, img*255)
        forward_pred_scores.multiscale_horn_schunk_ssim[i + 2] = ssim(img2, img * 255)
        forward_pred_scores.multiscale_horn_schunk_psnr[i + 2] = psnr(img2, img * 255)

        # backward prediction
        img = warp(img2, backward_flow, backward_grad)
        temp="./Results/Multiscale_Horn_schunk_results/corridor/backward_prediction/backward_pred_"+str(i)+".png"
        cv2.imwrite(temp, img*255)
        backward_pred_scores.multiscale_horn_schunk_ssim[i] = ssim(img0, img * 255)
        backward_pred_scores.multiscale_horn_schunk_psnr[i] = psnr(img0, img * 255)

        # Interpolated
        img=interpolate1(img0,img2,forward_flow,backward_flow)
        temp = "./Results/Multiscale_Horn_schunk_results/corridor/interpolated_frame/interpolated_" + str(i+1) + ".png"
        cv2.imwrite(temp, img)
        interpolated_frame_scores.multiscale_horn_schunk_ssim[i + 1] = ssim(img1, img)
        interpolated_frame_scores.multiscale_horn_schunk_psnr[i + 1] = psnr(img1, img)



        # Calculating flow using lukas kanade
        forward_flow, forward_grad = lukas_kanade_flow(img0, img2, 9)
        backward_flow, backward_grad = lukas_kanade_flow(img2, img0, 9)

        # forward prediction
        img = warp(img0, forward_flow, forward_grad)
        temp="./Results/Lukas_kanade_results/corridor/forward_prediction/forward_pred_"+str(i+2)+".png"
        cv2.imwrite(temp, img*255)
        forward_pred_scores.lukas_kanade_ssim[i + 2] = ssim(img2, img * 255)
        forward_pred_scores.lukas_kanade_psnr[i + 2] = psnr(img2, img * 255)

        # backward prediction
        img = warp(img2, backward_flow, backward_grad)
        cv2.imwrite("./Results/Lukas_kanade_results/corridor/backward_prediction/backward_pred_"+str(i)+".png", img*255)
        backward_pred_scores.lukas_kanade_ssim[i] = ssim(img0, img * 255)
        backward_pred_scores.lukas_kanade_psnr[i] = psnr(img0, img * 255)

        # Interpolated
        img = interpolate1(img0, img2, forward_flow, backward_flow)
        temp = "./Results/Lukas_kanade_results/corridor/interpolated_frame/interpolated_" + str(i + 1) + ".png"
        cv2.imwrite(temp, img)
        interpolated_frame_scores.lukas_kanade_ssim[i + 1] = ssim(img1, img)
        interpolated_frame_scores.lukas_kanade_psnr[i + 1] = psnr(img1, img)



        # Calculating flow using multiscale Lukas kanade
        forward_flow, forward_grad = multiscale_lukas_kanade_flow(img0, img2,9, 4)
        backward_flow, backward_grad = multiscale_lukas_kanade_flow(img2, img0,9, 4)

        # forward prediction
        img = warp(img0, forward_flow, forward_grad)
        cv2.imwrite("./Results/Multiscale_Lukas_kanade_results/corridor/forward_prediction/forward_pred_"+str(i+2)+".png", img*255)
        forward_pred_scores.multiscale_lukas_kanade_ssim[i + 2] = ssim(img2, img * 255)
        forward_pred_scores.multiscale_lukas_kanade_psnr[i + 2] = psnr(img2, img * 255)


        # backward prediction
        img = warp(img2, backward_flow, backward_grad)
        cv2.imwrite("./Results/Multiscale_Lukas_kanade_results/corridor/backward_prediction/backward_pred_"+str(i)+".png", img*255)
        backward_pred_scores.multiscale_lukas_kanade_ssim[i] = ssim(img0, img * 255)
        backward_pred_scores.multiscale_lukas_kanade_psnr[i] = psnr(img0, img * 255)

        # Interpolated
        img = interpolate1(img0, img2, forward_flow, backward_flow)
        temp = "./Results/Multiscale_Lukas_kanade_results/corridor/interpolated_frame/interpolated_" + str(i + 1) + ".png"
        cv2.imwrite(temp, img)
        interpolated_frame_scores.multiscale_lukas_kanade_ssim[i + 1] = ssim(img1, img)
        interpolated_frame_scores.multiscale_lukas_kanade_psnr[i + 1] = psnr(img1, img)

    ## save scores in results/SSIM_PSNR_Scores/corridor
    forward_pred_scores.to_csv("./Results/SSIM_PSNR_Scores/corridor/forward_pred_scores4.csv")
    backward_pred_scores.to_csv("./Results/SSIM_PSNR_Scores/corridor/backward_pred_scores4.csv")
    interpolated_frame_scores.to_csv("./Results/SSIM_PSNR_Scores/corridor/interpolated_frame_scores4.csv")


    # Creating dataframe for scores (sphere dataset)
    forward_pred_scores = pd.DataFrame(
        columns=["horn_schunk_ssim", "horn_schunk_psnr", "multiscale_horn_schunk_ssim",
                 "multiscale_horn_schunk_psnr", "lukas_kanade_ssim", "lukas_kanade_psnr",
                 "multiscale_lukas_kanade_ssim", "multiscale_lukas_kanade_psnr"], index=list(range(2, 19, 2)))

    backward_pred_scores = pd.DataFrame(
        columns=["horn_schunk_ssim", "horn_schunk_psnr", "multiscale_horn_schunk_ssim",
                 "multiscale_horn_schunk_psnr", "lukas_kanade_ssim", "lukas_kanade_psnr",
                 "multiscale_lukas_kanade_ssim", "multiscale_lukas_kanade_psnr"], index=list(range(0, 19, 2)))

    interpolated_frame_scores = pd.DataFrame(
        columns=["horn_schunk_ssim", "horn_schunk_psnr", "multiscale_horn_schunk_ssim",
                 "multiscale_horn_schunk_psnr", "lukas_kanade_ssim", "lukas_kanade_psnr",
                 "multiscale_lukas_kanade_ssim", "multiscale_lukas_kanade_psnr"], index=list(range(1, 19, 2)))

    # Sphere Dataset
    for i in range(0,18,2):

        sph="./Dataset/sphere/"
        name1=str("sphere.")+str(i)+str(".ppm")
        name2 = str("sphere.") + str(i+2) + str(".ppm")
        name3=str("sphere.") + str(i+1) + str(".ppm")

        path1=str(sph)+str(name1)
        path2=str(sph)+str(name2)
        path3=str(sph)+str(name3)

        img0=cv2.imread(path1)
        img1=cv2.imread(path3)
        img2=cv2.imread(path2)

        img0=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
        img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Calculating flow using hornshunk
        forward_flow,forward_grad=horn_schunk_flow(img0,img2,lambada,max_iter,epsilon)
        backward_flow,backward_grad=horn_schunk_flow(img2,img0,lambada,max_iter,epsilon)


        # forward prediction
        img=warp(img0,forward_flow,forward_grad)
        cv2.imwrite("./Results/Horn_schunk_results/sphere/forward_prediction/forward_pred_"+str(i+2)+".png",img*255)
        forward_pred_scores.horn_schunk_ssim[i+2]=ssim(img2,img*255)
        forward_pred_scores.horn_schunk_psnr[i + 2] = psnr(img2, img * 255)

        # backward prediction
        img = warp(img2, backward_flow, backward_grad)
        cv2.imwrite("./Results/Horn_schunk_results/sphere/backward_prediction/backward_pred_"+str(i)+".png", img*255)
        backward_pred_scores.horn_schunk_ssim[i]=ssim(img0,img*255)
        backward_pred_scores.horn_schunk_psnr[i] = psnr(img0, img * 255)

        # Interpolated
        img = interpolate1(img0, img2, forward_flow, backward_flow)
        temp = "./Results/Horn_schunk_results/sphere/interpolated_frame/interpolated_" + str(i + 1) + ".png"
        cv2.imwrite(temp, img)
        interpolated_frame_scores.horn_schunk_ssim[i+1]=ssim(img1,img)
        interpolated_frame_scores.horn_schunk_psnr[i + 1] = psnr(img1, img)



        # Calculating flow using multiscale horn schunk
        forward_flow, forward_grad = multiscale_horn_schunk_flow(img0, img2, lambada, max_iter, epsilon,4)
        backward_flow, backward_grad = multiscale_horn_schunk_flow(img2, img0, lambada, max_iter, epsilon,4)

        # forward prediction
        img = warp(img0, forward_flow, forward_grad)
        cv2.imwrite("./Results/Multiscale_Horn_schunk_results/sphere/forward_prediction/forward_pred_"+str(i+2)+".png", img*255)
        forward_pred_scores.multiscale_horn_schunk_ssim[i + 2] = ssim(img2, img * 255)
        forward_pred_scores.multiscale_horn_schunk_psnr[i + 2] = psnr(img2, img * 255)

        # backward prediction
        img = warp(img2, backward_flow, backward_grad)
        cv2.imwrite("./Results/Multiscale_Horn_schunk_results/sphere/backward_prediction/backward_pred_"+str(i)+".png", img*255)
        backward_pred_scores.multiscale_horn_schunk_ssim[i] = ssim(img0, img * 255)
        backward_pred_scores.multiscale_horn_schunk_psnr[i] = psnr(img0, img * 255)

        # Interpolated
        img = interpolate1(img0, img2, forward_flow, backward_flow)
        temp = "./Results/Multiscale_Horn_schunk_results/sphere/interpolated_frame/interpolated_" + str(i + 1) + ".png"
        cv2.imwrite(temp, img)
        interpolated_frame_scores.multiscale_horn_schunk_ssim[i + 1] = ssim(img1, img)
        interpolated_frame_scores.multiscale_horn_schunk_psnr[i + 1] = psnr(img1, img)



        # Calculating flow using lukas kanade
        forward_flow, forward_grad = lukas_kanade_flow(img0, img2, 9)
        backward_flow, backward_grad = lukas_kanade_flow(img2, img0, 9)

        # forward prediction
        img = warp(img0, forward_flow, forward_grad)
        cv2.imwrite("./Results/Lukas_kanade_results/sphere/forward_prediction/forward_pred_"+str(i+2)+".png", img*255)
        forward_pred_scores.lukas_kanade_ssim[i + 2] = ssim(img2, img * 255)
        forward_pred_scores.lukas_kanade_psnr[i + 2] = psnr(img2, img * 255)

        # backward prediction
        img = warp(img2, backward_flow, backward_grad)
        cv2.imwrite("./Results/Lukas_kanade_results/sphere/backward_prediction/backward_pred_"+str(i)+".png", img*255)
        backward_pred_scores.lukas_kanade_ssim[i] = ssim(img0, img * 255)
        backward_pred_scores.lukas_kanade_psnr[i] = psnr(img0, img * 255)

        # Interpolated
        img = interpolate1(img0, img2, forward_flow, backward_flow)
        temp = "./Results/Lukas_kanade_results/sphere/interpolated_frame/interpolated_" + str(i + 1) + ".png"
        cv2.imwrite(temp, img)
        interpolated_frame_scores.lukas_kanade_ssim[i + 1] = ssim(img1, img )
        interpolated_frame_scores.lukas_kanade_psnr[i + 1] = psnr(img1, img )



        # Calculating flow using multiscale Lukas kanade
        forward_flow, forward_grad = multiscale_lukas_kanade_flow(img0, img2,9, 4)
        backward_flow, backward_grad = multiscale_lukas_kanade_flow(img2, img0,9, 4)

        # forward prediction
        img = warp(img0, forward_flow, forward_grad)
        cv2.imwrite("./Results/Multiscale_Lukas_kanade_results/sphere/forward_prediction/forward_pred_"+str(i+2)+".png", img*255)
        forward_pred_scores.multiscale_lukas_kanade_ssim[i+2] = ssim(img2, img * 255)
        forward_pred_scores.multiscale_lukas_kanade_psnr[i+2] = psnr(img2, img * 255)


        # backward prediction
        img = warp(img2, backward_flow, backward_grad)
        cv2.imwrite("./Results/Multiscale_Lukas_kanade_results/sphere/backward_prediction/backward_pred_"+str(i)+".png", img*255)
        backward_pred_scores.multiscale_lukas_kanade_ssim[i] = ssim(img0, img * 255)
        backward_pred_scores.multiscale_lukas_kanade_psnr[i] = psnr(img0, img * 255)


        # Interpolated
        img = interpolate1(img0, img2, forward_flow, backward_flow)
        temp = "./Results/Multiscale_Lukas_kanade_results/sphere/interpolated_frame/interpolated_" + str(i + 1) + ".png"
        cv2.imwrite(temp, img)
        interpolated_frame_scores.multiscale_lukas_kanade_ssim[i + 1] = ssim(img1, img )
        interpolated_frame_scores.multiscale_lukas_kanade_psnr[i + 1] = psnr(img1, img )


    ## Save scores for sphere dataset
    forward_pred_scores.to_csv("./Results/SSIM_PSNR_Scores/sphere/forward_pred_scores4.csv")
    backward_pred_scores.to_csv("./Results/SSIM_PSNR_Scores/sphere/backward_pred_scores4.csv")
    interpolated_frame_scores.to_csv("./Results/SSIM_PSNR_Scores/sphere/interpolated_frame_scores4.csv")

