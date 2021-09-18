from Functions.grdient import *
import numpy as np

def lukas_kanade_flow(img0,img2,N):
    """
    :param img0: first image
    :param img2: second image (next frame)
    :param N: size of the window (no. of equations= N**2)
    :return: flow a,b and gradients fx, fy, ft
    """

    #Initializing flow with zero matrix
    a=np.zeros((img0.shape))
    b=np.zeros((img0.shape))

    # Calculating gradients
    fx,fy,ft=grad_cal(img0,img2)

    for x in range(N//2,img0.shape[0]-N//2):
        for y in range(N//2,img0.shape[1]-N//2):

            ## Selecting block(Window) around the pixel
            block_fx = fx[x - N // 2:x + N //2 + 1,  y - N // 2:y + N // 2 + 1]
            block_fy = fy[x - N // 2:x + N // 2 + 1, y - N // 2:y + N // 2 + 1]
            block_ft = ft[x - N // 2:x + N // 2 + 1, y - N // 2:y + N // 2 + 1]

            ## Flattening to genrate equations
            block_ft = block_ft.flatten()
            block_fy = block_fy.flatten()
            block_fx = block_fx.flatten()

            ## Reshaping to generate the format of Ax=B
            B=-1*np.asarray(block_ft)
            A=np.asarray([block_fx,block_fy]).reshape(-1,2)

            ## Solving equations using pseudo inverse
            flow=np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(A),A)),np.transpose(A)),B)

            ## Updating flow matrix a,b
            a[x,y]=flow[0]
            b[x,y]=flow[1]

    return [a,b],[fx,fy,ft]




