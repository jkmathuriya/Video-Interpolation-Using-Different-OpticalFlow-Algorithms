from Lukas_kanade.lukas_kanade import *
from Functions.Pyramid import *
from matplotlib import pyplot as plt
from Functions.grdient import *


def iterative_lukas_kanade_flow(img0,img2,N,old_flow):

    a_old = old_flow[0]
    b_old = old_flow[1]
    fx, fy, ft = grad_cal(img0,img2)

    # warping image with old flow
    pred = np.round((img0/255 + fx*a_old + fy*b_old + ft)*255)
    pred[pred>255]=255

    # Calculating a1~ and b1~
    flow,grad=lukas_kanade_flow(pred,img2,N)

    # New flow
    new_a=a_old+flow[0]
    new_b=b_old+flow[1]

    return [new_a,new_b]


def multiscale_lukas_kanade_flow(img0,img2,N,levels):

    pyr0,shapes0=pyramid_down(img0,levels)
    pyr2,shapes2=pyramid_down(img2,levels)

    # for i in range(levels-1,-1,-1):
    #     plt.figure()
    #     plt.imshow(pyr0[0:shapes0[i][0],0:shapes0[i][1],i],cmap="gray")
    #     plt.figure()
    #     plt.imshow(pyr2[0:shapes0[i][0],0:shapes0[i][1],i], cmap="gray")
    # print(shapes0)
    # plt.show()
    
    # Calculate initial flow at lowest scale
    [a,b],grad=lukas_kanade_flow(pyr0[0:shapes0[levels-1][0],0:shapes0[levels-1][1],levels-1],pyr2[0:shapes0[levels-1][0],0:shapes0[levels-1][1],levels-1],N)
    
    # upsample flow for next level
    a2=cv2.pyrUp(a)
    b2=cv2.pyrUp(b)


    for i in range(levels-2,-1,-1):

        [a,b]=iterative_lukas_kanade_flow(pyr0[0:shapes0[i][0],0:shapes0[i][1],i],pyr2[0:shapes0[i][0],0:shapes0[i][1],i],N,[a2,b2])

        # upsample flow for next level
        a2 = cv2.pyrUp(a)
        b2 = cv2.pyrUp(b)


    grad=grad_cal(img0,img2)
    return [a,b],grad
