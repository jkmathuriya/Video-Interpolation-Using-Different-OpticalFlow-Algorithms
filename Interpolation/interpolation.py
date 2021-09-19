
import cv2
import numpy as np
import scipy.interpolate


def interpolate1(img0, img2, forward_flow,backward_flow):

    height, width = img0.shape

    af, bf = forward_flow

    ab, bb = backward_flow

    t = 0.5
    ut = np.full([af.shape[0], af.shape[1], 2], np.nan)

    # Occlusion detection
    occ_detect = True
    if occ_detect:
        similarity = np.full([height, width], np.inf)

    # Predict intermediate optical flow
    xx = np.broadcast_to(np.arange(width), (height, width))
    yy = np.broadcast_to(np.arange(height)[:, None], (height, width))

    xt = np.round(xx + t * af)
    yt = np.round(yy + t * bf)

    for i in range(height):
        for j in range(width):
            i_ind_image = int(yt[i, j])
            j_ind_image = int(xt[i, j])

            if i_ind_image >= 0 and i_ind_image < height and j_ind_image >= 0 and j_ind_image < width:
                if occ_detect:
                    e = np.square(int(img2[i_ind_image, j_ind_image]) - int(img0[i, j]))
                    s = np.sum(e)
                    #print(e)
                    if s < similarity[i_ind_image, j_ind_image]:
                        ut[i_ind_image, j_ind_image, 0] = af[i, j]
                        ut[i_ind_image, j_ind_image, 1] = bf[i, j]
                        similarity[i_ind_image, j_ind_image] = s
                else:
                    ut[i_ind_image, j_ind_image, 0] = af[i, j]
                    ut[i_ind_image, j_ind_image, 1] = bf[i, j]

    uti = outside_in_fill(ut)

    # Occlusion masks
    occlusion_first = np.zeros_like(img0)
    occlusion_second = np.zeros_like(img2)

    occlusion_x1 = np.round(xx + af).astype(np.int)
    occlusion_y1 = np.round(yy + bf).astype(np.int)

    occlusion_x1 = np.clip(occlusion_x1, 0, height - 1)
    occlusion_y1 = np.clip(occlusion_y1, 0, width - 1)

    for i in range(occlusion_first.shape[0]):
        for j in range(occlusion_first.shape[1]):
            if np.abs(af[i, j] + ab[occlusion_x1[i, j], occlusion_y1[i, j]]) > 0.5:
                occlusion_first[i, j] = 1

            if np.abs(bf[i, j] + bb[occlusion_x1[i, j], occlusion_y1[i, j]]) > 0.5:
                occlusion_first[i, j] = 1

    occlusion_x1 = np.round(xx + ab).astype(np.int)
    occlusion_y1 = np.round(yy + bb).astype(np.int)

    occlusion_x1 = np.clip(occlusion_x1, 0, height - 1)
    occlusion_y1 = np.clip(occlusion_y1, 0, width - 1)

    for i in range(occlusion_second.shape[0]):
        for j in range(occlusion_second.shape[1]):
            if np.abs(ab[i, j] + af[occlusion_x1[i, j], occlusion_y1[i, j]]) > 0.5:
                occlusion_second[i, j] = 1

            if np.abs(bb[i, j] + bf[occlusion_x1[i, j], occlusion_y1[i, j]]) > 0.5:
                occlusion_second[i, j] = 1

    # Intermediate image indices
    img0_for_x = xx - t * uti[:, :, 0]
    img0_for_y = yy - t * uti[:, :, 1]

    xt0 = np.clip(img0_for_x, 0, height - 1)
    yt0 = np.clip(img0_for_y, 0, width - 1)

    img1_for_x = xx + (1 - t) * uti[:, :, 0]
    img1_for_y = yy + (1 - t) * uti[:, :, 1]

    xt1 = np.clip(img1_for_x, 0, height - 1)
    yt1 = np.clip(img1_for_y, 0, width - 1)

    # Interpolate the images according to occlusion masks
    It = np.zeros(img0.shape)
    image1_interp = scipy.interpolate.RectBivariateSpline(np.arange(width), np.arange(height), img0.T)
    image2_interp = scipy.interpolate.RectBivariateSpline(np.arange(width), np.arange(height), img2.T)

    for i in range(It.shape[0]):
        for j in range(It.shape[1]):
            if not (occlusion_first[i, j] or occlusion_second[i, j]) or (
                    occlusion_first[i, j] and occlusion_second[i, j]
            ):
                It[i, j] = t * image1_interp(xt0[i, j], yt0[i, j]) + (1 - t) * image2_interp(xt1[i, j], yt1[i, j])
            elif occlusion_first[i, j]:
                It[i, j] = image2_interp(xt1[i, j], yt1[i, j])
            elif occlusion_second[i, j]:
                It[i, j] = image1_interp(xt0[i, j], yt0[i, j])

    It = It.astype(np.int)

    return It

def outside_in_fill(image):
    """
    Outside in fill mentioned in paper
    :param image: Image matrix to be filled
    :return: output
    """

    rows, cols = image.shape[:2]

    col_start = 0
    col_end = cols
    row_start = 0
    row_end = rows
    lastValid = np.full([2], np.nan)
    while col_start < col_end or row_start < row_end:
        for c in range(col_start, col_end):
            if not np.isnan(image[row_start, c, 0]):
                lastValid = image[row_start, c, :]
            else:
                image[row_start, c, :] = lastValid

        for r in range(row_start, row_end):
            if not np.isnan(image[r, col_end - 1, 0]):
                lastValid = image[r, col_end - 1, :]
            else:
                image[r, col_end - 1, :] = lastValid

        for c in reversed(range(col_start, col_end)):
            if not np.isnan(image[row_end - 1, c, 0]):
                lastValid = image[row_end - 1, c, :]
            else:
                image[row_end - 1, c, :] = lastValid

        for r in reversed(range(row_start, row_end)):
            if not np.isnan(image[r, col_start, 0]):
                lastValid = image[r, col_start, :]
            else:
                image[r, col_start, :] = lastValid

        if col_start < col_end:
            col_start = col_start + 1
            col_end = col_end - 1

        if row_start < row_end:
            row_start = row_start + 1
            row_end = row_end - 1

    output = image

    return output
