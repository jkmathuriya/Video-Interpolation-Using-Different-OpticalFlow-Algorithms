import numpy as np
import scipy.ndimage
from Functions.grdient import *


def horn_schunk_flow(img0,img2,lambada,max_iter,epsilon):
    """

    :param img0: first frame
    :param img2: second frame
    :param lambada: hyper parameter
    :param max_iter: threshold for iterations
    :param epsilon: decay rate
    :return: flow and gradient
    """
    decay=10000
    i=0
    ## averaging kernel
    avg_kernel=np.array([[0,1,0],[1,0,1],[0,1,0]])/4

    ## Calculating gradient
    fx,fy,ft=grad_cal(img0,img2)

    a=np.zeros((img0.shape))
    b=np.zeros((img0.shape))

    while(decay>epsilon and i<=max_iter):
        i+=1
        ## Calculating
        a_avg = scipy.ndimage.convolve(input=a, weights=avg_kernel)
        b_avg = scipy.ndimage.convolve(input=b, weights=avg_kernel)

        temp = (fx * a_avg + fy * b_avg + ft) / (1+lambada*( fx ** 2 + fy ** 2))

        ## Updating flow
        a=a_avg-lambada*fx*temp
        b=b_avg-lambada*fy*temp

        ## calculating decay
        decay=np.max(np.max((abs(a-a_avg)+abs(b-b_avg))))
        #print(i,decay)

    return [a,b],[fx,fy,ft]
