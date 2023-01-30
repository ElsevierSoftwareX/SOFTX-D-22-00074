#DPIV_ALGORITHM
import numpy as np
from scipy import ndimage
from scipy import interpolate
from scipy.io import savemat
import copy
import cv2
import time
from dpivsoft.Classes  import Parameters
from dpivsoft.Classes  import GPU
from dpivsoft.Classes  import grid

import matplotlib.pyplot as plt

#PROCESS PIV
def processing(Img1, Img2):

    #Generate x-y mesh for the PIV 
    grid.generate_mesh(Img1.shape[1], Img1.shape[0])

    # Apply mask to images
    if Parameters.mask:
        [Img1, Img2] = masking(Img1, Img2)

    # Firs cross-correlation
    if Parameters.direct_calc:
        # direct cross correlation
        [x1, y1, u1, v1] = corrDirect1(Img1, Img2)
    else:
        # FFT cros correlation
        [x1, y1, u1, v1] = corrFFT1(Img1, Img2)

    # Iterate on the first grid if specified
    if Parameters.no_iter_1 > 1:
        [x1, y1, u1, v1] = corrFFT1bis(Img1, Img2, x1, y1, u1, v1)

    # Second cross-correlation
    [x2, y2, u2, v2] = corrFFT2(Img1, Img2, x1, y1, u1, v1)


    return x2, y2, u2, v2


# FIRST STEP PIV CORRELATION FUNCTION
def corrFFT1(Img1, Img2):

    # Definition of Parameters to reduce length
    box_size_x = Parameters.box_size_1_x
    box_size_y = Parameters.box_size_1_y
    no_boxes_1_x = Parameters.no_boxes_1_x
    no_boxes_1_y = Parameters.no_boxes_1_y
    window_x = Parameters.window_1_x
    window_y = Parameters.window_1_y

    [Height,Width] = Img1.shape
    Parameters.Data.height = Height
    Parameters.Data.width = Width

    # Initialize all matrix
    box_origin_x_1 = grid.box_origin_x_1
    box_origin_y_1 = grid.box_origin_y_1
    x_1 = grid.x_1
    y_1 = grid.y_1
    u_1 = np.zeros([no_boxes_1_y, no_boxes_1_x])
    v_1 = np.zeros([no_boxes_1_y, no_boxes_1_x])

    # Prevents the size of the window bigger than the box_size
    window_x = np.amin([2*np.round(window_x/2), box_size_x-4])
    window_y = np.amin([2*np.round(window_y/2), box_size_y-4])

    if Parameters.weighting:
        [i_matrix, j_matrix] = np.meshgrid(np.arange(box_size_x),
                np.arange(box_size_y))
        Weighting_Function = weight_function(i_matrix, j_matrix,
                box_size_x, box_size_y)

    # Apply Gaussian filter to images only for first iteration
    if Parameters.gaussian_size:
        [Img1, Img2] = gaussian_filter(Img1, Img2, Parameters.gaussian_size)

    for i in range(0, no_boxes_1_x):
        for j in range(0, no_boxes_1_y):

            # Define sub images to work
            box_o_y = int(box_origin_y_1[j,i])
            box_o_x = int(box_origin_x_1[j,i])

            SubImg1 = (Img1[box_o_y : box_o_y + box_size_y,
                box_o_x : box_o_x + box_size_x])
            SubImg2 = (Img2[box_o_y : box_o_y + box_size_y,
                box_o_x : box_o_x + box_size_x])

            # Mask intensity changed to unmasked mean intensity
            if Parameters.mask:
                SubImg1, SubImg2 = change_mask(SubImg1, SubImg2)

            if Parameters.weighting:
                SubImg1 = np.multiply(SubImg1, Weighting_Function)
                SubImg2 = np.multiply(SubImg2, Weighting_Function)

            # Get the intensity of the images centered in the mean
            SubImg1 = SubImg1-np.sum(SubImg1)/(box_size_y*box_size_x)
            SubImg2 = SubImg2-np.sum(SubImg2)/(box_size_y*box_size_x)

            # Sum of intensity Image to Scale at end
            Sigma1 = max(0.1, np.sqrt(np.sum(SubImg1**2)))
            Sigma2 = max(0.1, np.sqrt(np.sum(SubImg2**2)))

            # Cross Correlation Function of the pair images scaled with
            # sigma1 and sigma2
            correlation = (np.fft.fftshift(np.real(np.fft.ifft2(
                np.multiply(np.conj(np.fft.fft2(SubImg1)),
                np.fft.fft2(SubImg2)))))/(Sigma1*Sigma2))

            # Find Peaks of the correlation Fucntion
            [epsilon_x,epsilon_y,col_idx,row_idx] = find_peaks(correlation,
                    window_x, window_y)

            u_1[j,i] = epsilon_x + col_idx-box_size_x/2
            v_1[j,i] = epsilon_y + row_idx-box_size_y/2

    return x_1, y_1, u_1, v_1


# ITERATIVE OPTION FOR FIRST STEP ON PIV CROSSCORRELATION
def corrFFT1bis(Img1, Img2, x_1, y_1, u_1, v_1):

    # Definition of Parameters to reduce length
    box_size_x = Parameters.box_size_1_x
    box_size_y = Parameters.box_size_1_y
    no_boxes_1_x = Parameters.no_boxes_1_x
    no_boxes_1_y = Parameters.no_boxes_1_y
    window_x = Parameters.window_1_x
    window_y = Parameters.window_1_y
    median_limit = Parameters.median_limit
    weighting = Parameters.weighting
    gaussian = Parameters.gaussian_size

    [Height,Width] = Img1.shape

    # Define origin of boxes
    box_origin_x = x_1 - box_size_x/2
    box_origin_y = y_1 - box_size_y/2

    [i_matrix, j_matrix] = np.meshgrid(np.arange(box_size_x),
            np.arange(box_size_y))

    if Parameters.weighting:
        Weighting_Function = weight_function(i_matrix, j_matrix,
                box_size_x, box_size_y)

    # Prevents the size of the window bigger than the box_size
    window_x = np.amin([2*np.round(window_x/2),box_size_x-4])
    window_y = np.amin([2*np.round(window_y/2),box_size_y-4])

    for calc in range(1,Parameters.no_iter_1):
        # Median filter
        u_1, v_1, err_vect = median_filter(u_1, v_1, median_limit)

        # If mask check points to prevent bleeding
        if Parameters.mask:
            # Check points inside the mask
            u_1, v_1 = check_mask(u_1, v_1, grid.mask_1)

        # Calculates the jacobian matrix onto de grid
        du_dx, du_dy, dv_dx, dv_dy = jacobian_matrix(u_1, v_1, x_1, y_1,
                no_boxes_1_x, no_boxes_1_y)

        for j in range(0,no_boxes_1_y):
            for i in range(0,no_boxes_1_y):

                # Obtain deformed image.
                SubImg1, SubImg2, u_index, v_index = deform_image(Img1, Img2,
                        Width, Height, box_origin_x, box_origin_y, i_matrix,
                        j_matrix, box_size_x, box_size_y, u_1, v_1, du_dx,
                        du_dy, dv_dx, dv_dy, i, j)

                # Mask intensity changed to unmasked mean intensity
                if Parameters.mask:
                    SubImg1, SubImg2 = change_mask(SubImg1, SubImg2)

                # Weighting if required
                if Parameters.weighting:
                    SubImg1 = np.multiply(SubImg1, Weighting_Function)
                    SubImg2 = np.multiply(SubImg2, Weighting_Function)

                # Sum of intensity Image to Scale at end
                Sigma1 = max(0.1, np.sqrt(np.sum(SubImg1**2)))
                Sigma2 = max(0.1, np.sqrt(np.sum(SubImg2**2)))

                # Cross Correlation Function of the pair images scaled with
                # sigma1 and sigma2
                correlation = (np.fft.fftshift(np.abs(np.fft.ifft2(
                    np.multiply(np.conj(np.fft.fft2(SubImg1)),
                    np.fft.fft2(SubImg2)))))/(Sigma1*Sigma2))

                # Find Peaks of the correlation Function
                epsilon_x, epsilon_y, col_idx, row_idx = find_peaks(
                        correlation, window_x, window_y)

                u_1[j,i] = (u_index[row_idx, col_idx] + epsilon_x +
                        col_idx-box_size_x/2)
                v_1[j,i] = (v_index[row_idx, col_idx] + epsilon_y +
                        row_idx-box_size_y/2)

    return x_1, y_1, u_1, v_1

# SECOND STEP ON PIV CROSS-CORRELATION
def corrFFT2(Img1, Img2, x_1, y_1, u_1, v_1):

    # Definition of Parameters to reduce length
    box_size_1_x = Parameters.box_size_1_x
    box_size_1_y = Parameters.box_size_1_y
    no_boxes_1_x = Parameters.no_boxes_1_x
    no_boxes_1_y = Parameters.no_boxes_1_y
    box_size_2_x = Parameters.box_size_2_x
    box_size_2_y = Parameters.box_size_2_y
    no_boxes_2_x = Parameters.no_boxes_2_x
    no_boxes_2_y = Parameters.no_boxes_2_y
    window_x = Parameters.window_2_x
    window_y = Parameters.window_2_y
    median_limit = Parameters.median_limit


    [Height,Width] = Img1.shape

    # Define index functions
    i_matrix, j_matrix = np.meshgrid(np.arange(0, box_size_2_x),
            np.arange(0, box_size_2_y))

    if Parameters.weighting:
        Weighting_Function = weight_function(i_matrix, j_matrix,
                box_size_2_x, box_size_2_y)

    weight_i = i_matrix[1,1:4]-2
    weight_j = j_matrix[1:4,1]-2

    # Prevents the size of the window bigger than the box_size
    window_x = np.amin([2*np.round(window_x/2),box_size_2_x-4])
    window_y = np.amin([2*np.round(window_y/2),box_size_2_y-4])

    x_margin = 3/2*np.amax([box_size_1_x, box_size_2_x])
    y_margin = 3/2*np.amax([box_size_1_y, box_size_2_y])

    # Second grid is placed completely inside first one
    x_2 = grid.x_2[0, :]
    y_2 = grid.y_2[:, 0]

    # Calculate Jacobian Matrix
    Interpolation_Start = time.time()
    du_dx, du_dy, dv_dx, dv_dy = jacobian_matrix(u_1, v_1, x_1, y_1,
            no_boxes_1_x, no_boxes_1_y)

    # Interpolate First Run Results on second grid
    du_dx, du_dy, dv_dx, dv_dy, u_2, v_2, x_2, y_2 = interpolations(
            du_dx, du_dy, dv_dx, dv_dy, u_1, v_1, x_1, y_1, x_2, y_2,
            no_boxes_1_x*no_boxes_1_y)
    Interpolation_End = time.time()
    Interpolation_Time = Interpolation_End-Interpolation_Start

    GPU.test_fft2 = u_2

    # Define origin of boxes without translation
    box_origin_x_2 = x_2 - box_size_2_x/2;
    box_origin_y_2 = y_2 - box_size_2_y/2;

    for j in range(0,no_boxes_2_y):
        for i in range(0,no_boxes_2_x):
            k = 0
            epsilon_x = 1
            epsilon_y = 1

            while ((np.abs(epsilon_x>0.5) or np.abs(epsilon_y>0.5)) and
                    k<Parameters.no_iter_2):

                k = k+1

                # Obtain deformed image.
                SubImg1, SubImg2, u_index, v_index = deform_image(Img1, Img2,
                        Width, Height, box_origin_x_2, box_origin_y_2,
                        i_matrix, j_matrix, box_size_2_x, box_size_2_y,
                        u_2, v_2, du_dx, du_dy, dv_dx, dv_dy, i, j)

                # Mask intensity changed to unmasked mean intensity
                if Parameters.mask:
                    SubImg1, SubImg2 = change_mask(SubImg1, SubImg2)

                # Weighting if required
                if Parameters.weighting:
                    SubImg1 = np.multiply(SubImg1,Weighting_Function)
                    SubImg2 = np.multiply(SubImg2,Weighting_Function)

                # Sum of intensity Image to Scale at end
                Sigma1 = max(0.1, np.sqrt(np.sum(SubImg1**2)))
                Sigma2 = max(0.1, np.sqrt(np.sum(SubImg2**2)))

                # Cross Correlation Function of the pair images scaled with
                # sigma1 and sigma2
                correlation = (np.fft.fftshift(np.abs(np.fft.ifft2(
                    np.multiply(np.conj(np.fft.fft2(SubImg1)),
                    np.fft.fft2(SubImg2)))))/(Sigma1*Sigma2))

                # Find Peaks of the correlation Fucntion
                epsilon_x, epsilon_y, col_idx, row_idx = find_peaks(
                        correlation, window_x, window_y)

                u_2[j,i] = (u_index[row_idx, col_idx] + epsilon_x +
                        col_idx - box_size_2_x/2)
                v_2[j,i] = (v_index[row_idx, col_idx] + epsilon_y +
                        row_idx - box_size_2_y/2)

    u_2, v_2, err_vect = median_filter(u_2, v_2, median_limit)

    # If mask check points to prevent bleeding
    if Parameters.mask:
        # Check points inside the mask
        u_2, v_2 = check_mask(u_2, v_2, grid.mask_2)


    return x_2, y_2, u_2, v_2

def corrDirect1(Img1, Img2):

    # Definition of Parameters to reduce length
    box_size_x = Parameters.box_size_1_x
    box_size_y = Parameters.box_size_1_y
    no_boxes_1_x = Parameters.no_boxes_1_x
    no_boxes_1_y = Parameters.no_boxes_1_y
    window_x = Parameters.window_x_1
    window_y = Parameters.window_y_1

    [Height,Width] = Img1.shape

    # Initialize all matrix
    box_origin_x_1 = np.zeros([no_boxes_1_y, no_boxes_1_x])
    box_origin_y_1 = np.zeros([no_boxes_1_y, no_boxes_1_x])
    x_1 = np.zeros([no_boxes_1_y, no_boxes_1_x])
    y_1 = np.zeros([no_boxes_1_y, no_boxes_1_x])
    u_1 = np.zeros([no_boxes_1_y, no_boxes_1_x])
    v_1 = np.zeros([no_boxes_1_y, no_boxes_1_x])
    correlation = np.zeros([window_y+1, window_x+1])

    if Parameters.weighting:
        [i_matrix, j_matrix] = np.meshgrid(np.arange(box_size_x),
                np.arange(box_size_y))
        Weighting_Function = weight_function(i_matrix, j_matrix,
                box_size_x, box_size_y)

    # Apply Gaussian filter to images only for first iteration
    if Parameters.gaussian_size:
        [Img1, Img2] = gaussian_filter(Img1, Img2, Parameters.gaussian_size)

    for j in range(0, no_boxes_1_y):
        for i in range(0, no_boxes_1_x):

            x_1[j,i] = (1 + round((window_x/2+box_size_x)/2) +
                round((i)*(Width-box_size_x-window_x/2-4)/(no_boxes_1_x-1)))
            y_1[j,i] = (1 + round((window_y/2+box_size_y)/2) +
                round((j)*(Height-box_size_y-window_y/2-4)/(no_boxes_1_y-1)))

            box_origin_x_1[j,i] = x_1[j,i]+1-round(box_size_x/2)
            box_origin_y_1[j,i] = y_1[j,i]+1-round(box_size_y/2)


            for jj in range(-round(window_y/2), round(window_y/2)+1):
                for ii in range(-round(window_x/2), round(window_x/2)+1):
                    box_o_y = int(box_origin_y_1[j,i]+round(jj/2))
                    box_o_x = int(box_origin_x_1[j,i]+round(ii/2))

                    SubImg1 = (Img1[box_o_y-jj:box_o_y-jj+box_size_y,
                            box_o_x-ii:box_o_x-ii+box_size_x])
                    SubImg2 = (Img2[box_o_y:box_o_y+box_size_y,
                            box_o_x:box_o_x+box_size_x])

                    if Parameters.weighting==1:
                        SubImg1 = np.multiply(SubImg1, Weighting_Function)
                        SubImg2 = np.multiply(SubImg2, Weighting_Function)

                    # Get the intensity of the images centered in the mean
                    SubImg1 = SubImg1-np.sum(SubImg1)/(box_size_y*box_size_x)
                    SubImg2 = SubImg2-np.sum(SubImg2)/(box_size_y*box_size_x)

                    SubImg1 = np.divide(SubImg1, max(0.1,
                        np.sqrt(np.sum(SubImg1**2))))
                    SubImg2 = np.divide(SubImg2, max(0.1,
                        np.sqrt(np.sum(SubImg2**2))))

                    temp1 = jj+round(window_y/2)
                    temp2 = ii+round(window_x/2)

                    correlation[temp1, temp2] = (
                        np.sum(np.multiply(SubImg1, SubImg2))
                    )

            # Find Peaks of the correlation Fucntion
            [epsilon_x,epsilon_y,col_idx,row_idx] = find_peaks(correlation,
                    window_x, window_y, 0)

            u_1[j,i] = epsilon_x + col_idx - (window_x/2)
            v_1[j,i] = epsilon_y + row_idx - (window_y/2)

    return x_1, y_1, u_1, v_1, box_origin_x_1



# FIND PEAKS POSITION ON CORRELATION FUNCTION
def find_peaks(correlation, window_x, window_y, westerweel = 1):

    box_size_y, box_size_x = correlation.shape

    # Indices of searching windows
    ini_cor_y = int(np.round(box_size_y/2-window_y/2))
    end_cor_y = int(np.round(box_size_y/2+window_y/2+1))
    ini_cor_x = int(np.round(box_size_x/2-window_x/2))
    end_cor_x = int(np.round(box_size_x/2+window_x/2+1))

    # Find first peak
    maxcor1 = (np.amax(correlation[ini_cor_y : end_cor_y,
        ini_cor_x:end_cor_x]))
    max_row1, max_col1 = np.where(correlation == maxcor1)
    max_row1 = max_row1[0].astype(int)
    max_col1 = max_col1[0].astype(int)

    if max_row1==0:
        max_row1 = int(box_size_y/2)
        max_col1 = int(box_size_x/2)

    # Obtain the average value of the first peak and boundaries
    lm = np.round(box_size_x/16).astype(int)
    matmax1 = correlation[np.round(max_row1-1):np.round(max_row1+2),
            np.round(max_col1-1):np.round(max_col1+2)]

    correlation_z = copy.copy(correlation)
    correlation_z[max_col1-lm:max_col1+lm+1,max_row1-lm:max_row1+lm+1] = 0

    # Find second peak
    maxcor2 = (np.amax(correlation_z[ini_cor_y : end_cor_y,
        ini_cor_x:end_cor_x]))
    max_row2, max_col2 = np.where(correlation_z == maxcor2)
    max_row2 = max_row2[0].astype(int)
    max_col2 = max_col2[0].astype(int)

    # Obtain the average value of the second peak and boundaries
    matmax2 = (correlation[np.round(max_row2-1):np.round(max_row2+2),
            np.round(max_col2-1):np.round(max_col2+2)])
    correlation_z[max_col2-lm:max_col2+lm+1,max_row2-lm:max_row2+lm+1] = 0

    sum_cor_1 = np.sum(np.sum(matmax1))-maxcor1
    sum_cor_2 = np.sum(np.sum(matmax2))-maxcor2

    if sum_cor_1 > sum_cor_2:
        fit_peak = matmax1
        max_row = max_row1
        max_col = max_col1

    else:
        fit_peak = matmax2
        max_row = max_row2
        max_col = max_col2


    # If sum_cor_1/(sum_cor_2+1e-10) > Parameters.peak_ratio:
    if (maxcor2/(maxcor1+1e-10) > Parameters.peak_ratio
            or fit_peak.shape != (3,3)):

        fit_peak = np.zeros([3,3])

    # Define weighting functions
    [weight_i, weight_j] = np.meshgrid(np.arange(-1,2),np.arange(-1,2))

    # Bias correction based on westerweel
    if westerweel:
        fit_peak = (np.divide(np.divide(
            fit_peak,1-np.abs(max_row-box_size_y/2 + weight_j)/box_size_y),
            1-np.abs(max_col-box_size_x/2+weight_i)/box_size_x))
        fit_peak[np.where(fit_peak<0.001)] = 0.001

    # Get sub-pixel accuracy with gaussian stimator
    if (fit_peak[1,1] == np.amin(fit_peak[1,:])):
        epsilon_x = 0
        max_col = box_size_x//2
    else:
        if ( fit_peak[1, 0] > 0 and
             fit_peak[1, 1] > 0 and
             fit_peak[1, 2] > 0 and
             (np.log(fit_peak[1,0]) +
              np.log(fit_peak[1,2]) -
              2*np.log(fit_peak[1,1])
             ) != 0):

            epsilon_x = (0.5*(np.log(fit_peak[1, 0]) -
                              np.log(fit_peak[1,2])) /
                         (np.log(fit_peak[1, 0]) +
                          np.log(fit_peak[1, 2]) -
                          2*np.log(fit_peak[1,1]))
                        )
        else:
            epsilon_x = 0

    if (fit_peak[1,1] == np.amin(fit_peak[:,1])):
        epsilon_y = 0
        max_row = box_size_y//2
    else:
        if ( fit_peak[0, 1] > 0 and
             fit_peak[1, 1] > 0 and
             fit_peak[2, 1] > 0 and
             ( np.log(fit_peak[0, 1]) +
               np.log(fit_peak[2, 1]) -
               2*np.log(fit_peak[1, 1])
             ) != 0):

            epsilon_y = (0.5*(np.log(fit_peak[0,1]) -
                              np.log(fit_peak[2,1])) /
                         (np.log(fit_peak[0, 1]) +
                          np.log(fit_peak[2, 1]) -
                          2*np.log(fit_peak[1,1]))
                        )
        else:
            epsilon_y = 0

    return  epsilon_x, epsilon_y, max_col, max_row


# MEDIAN FILTER ALGORITHM TO REMOVE SPURIOUS VECTORS
def median_filter(u,v,limit):
    err = 0
    err_vect = 0
    [num_row, num_col] = u.shape

    u_neigh = np.zeros([9,num_col-2])
    v_neigh = np.zeros([9,num_col-2])

    uf = np.zeros([num_row,num_col])
    vf = np.zeros([num_row,num_col])

    # Loop to calculate all 8-neighbors medians
    for j in range(1,num_row-1):
        u_neigh[0,:] = u[j-1,0:num_col-2]
        u_neigh[1,:] = u[j-1,1:num_col-1]
        u_neigh[2,:] = u[j-1,2:num_col]
        u_neigh[3,:] = u[j,0:num_col-2]
        u_neigh[4,:] = u[j,1:num_col-1]
        u_neigh[5,:] = u[j,2:num_col]
        u_neigh[6,:] = u[j+1,0:num_col-2]
        u_neigh[7,:] = u[j+1,1:num_col-1]
        u_neigh[8,:] = u[j+1,2:num_col]

        v_neigh[0,:] = v[j-1,0:num_col-2]
        v_neigh[1,:] = v[j-1,1:num_col-1]
        v_neigh[2,:] = v[j-1,2:num_col]
        v_neigh[3,:] = v[j,0:num_col-2]
        v_neigh[4,:] = v[j,1:num_col-1]
        v_neigh[5,:] = v[j,2:num_col]
        v_neigh[6,:] = v[j+1,0:num_col-2]
        v_neigh[7,:] = v[j+1,1:num_col-1]
        v_neigh[8,:] = v[j+1,2:num_col]

        u_median = np.median(u_neigh,0);
        v_median = np.median(v_neigh,0);

        median_magnitude = (np.sqrt(np.power(u_median, 2) +
            np.power(v_median,2)))
        delta_u = (np.sqrt(np.power(u[j, 1:num_col-1]-u_median, 2) +
            np.power(v[j,1:num_col-1]-v_median,2)))

        # Counter of vectors changed
        change_u = 1/2+np.sign(delta_u-median_magnitude*limit)/2

        # Save final value
        uf[j, 1:num_col-1] = (u[j, 1:num_col-1] + np.multiply(
            change_u, (u_median - u[j,1:num_col-1])))
        vf[j, 1:num_col-1] = (v[j, 1:num_col-1] + np.multiply(
            change_u,(v_median-v[j,1:num_col-1])))
        err_vect = err_vect+np.sum(change_u)

    # Change of outliers vectors on the lower using with 6-neighbors medians
    j=0
    u_neigh = np.zeros([6,num_col-2])
    v_neigh = np.zeros([6,num_col-2])

    u_neigh[0,:] = u[j,0:num_col-2]
    u_neigh[1,:] = u[j,1:num_col-1]
    u_neigh[2,:] = u[j,2:num_col]
    u_neigh[3,:] = u[j+1,0:num_col-2]
    u_neigh[4,:] = u[j+1,1:num_col-1]
    u_neigh[5,:] = u[j+1,2:num_col]

    v_neigh[0,:] = v[j,0:num_col-2]
    v_neigh[1,:] = v[j,1:num_col-1]
    v_neigh[2,:] = v[j,2:num_col]
    v_neigh[3,:] = v[j+1,0:num_col-2]
    v_neigh[4,:] = v[j+1,1:num_col-1]
    v_neigh[5,:] = v[j+1,2:num_col]

    u_median = np.median(u_neigh,0)
    v_median = np.median(v_neigh,0)

    median_magnitude = np.sqrt(np.power(u_median,2) + np.power(v_median, 2))
    delta_u = np.sqrt(np.power(u[j,1:num_col-1]-u_median,2) +
            np.power(v[j,1:num_col-1]-v_median,2))

    # Counter of vectors changed
    change_u = 1/2+np.sign(delta_u-median_magnitude*limit)/2

    # Save final value
    uf[j, 1:num_col-1] = u[j, 1:num_col-1] + np.multiply(
            change_u,(u_median-u[j,1:num_col-1]))
    vf[j, 1:num_col-1] = v[j, 1:num_col-1] + np.multiply(
            change_u,(v_median-v[j,1:num_col-1]))
    err_vect = err_vect+np.sum(change_u)

    # Change the outlier vectors on the upper row using with
    # 6-neighbors medians
    j=num_row-1
    u_neigh[0,:] = u[j-1,0:num_col-2]
    u_neigh[1,:] = u[j-1,1:num_col-1]
    u_neigh[2,:] = u[j-1,2:num_col]
    u_neigh[3,:] = u[j,0:num_col-2]
    u_neigh[4,:] = u[j,1:num_col-1]
    u_neigh[5,:] = u[j,2:num_col]

    v_neigh[0,:] = v[j-1,0:num_col-2]
    v_neigh[1,:] = v[j-1,1:num_col-1]
    v_neigh[2,:] = v[j-1,2:num_col]
    v_neigh[3,:] = v[j,0:num_col-2]
    v_neigh[4,:] = v[j,1:num_col-1]
    v_neigh[5,:] = v[j,2:num_col]

    u_median = np.median(u_neigh,0);
    v_median = np.median(v_neigh,0);

    median_magnitude = np.sqrt(np.power(u_median,2) + np.power(v_median,2))
    delta_u = (np.sqrt(np.power(u[j, 1:num_col-1]-u_median, 2) +
            np.power((v[j,1:num_col-1]-v_median),2)))

    # Counter of vectors changed
    change_u = 1/2 + np.sign(delta_u - median_magnitude * limit)/2;

    # Save final value
    uf[j, 1:num_col-1] = u[j, 1:num_col-1] + np.multiply(
            change_u, (u_median-u[j, 1:num_col-1]))
    vf[j, 1:num_col-1] = v[j, 1:num_col-1] + np.multiply(
            change_u, (v_median-v[j, 1:num_col-1]))
    err_vect=err_vect + np.sum(change_u);

    # Initialize again for sides
    u_neigh = np.zeros([9, num_row-2])
    v_neigh = np.zeros([9, num_row-2])

    # Change of outliers vectors on the left colum with 6-neighbors medians
    j = 0

    u_neigh[0,:] = u[0:num_row-2,j]
    u_neigh[1,:] = u[1:num_row-1,j]
    u_neigh[2,:] = u[2:num_row,j]
    u_neigh[3,:] = u[0:num_row-2,j+1]
    u_neigh[4,:] = u[1:num_row-1,j+1]
    u_neigh[5,:] = u[2:num_row,j+1]

    v_neigh[0,:] = v[0:num_row-2,j]
    v_neigh[1,:] = v[1:num_row-1,j]
    v_neigh[2,:] = v[2:num_row,j]
    v_neigh[3,:] = v[0:num_row-2,j+1]
    v_neigh[4,:] = v[1:num_row-1,j+1]
    v_neigh[5,:] = v[2:num_row,j+1]

    u_median = np.median(u_neigh,0)
    v_median = np.median(v_neigh,0)

    median_magnitude = np.sqrt(np.power(u_median,2) + np.power(v_median,2))
    delta_u = (np.sqrt(np.power(u[1:num_row-1,j]-u_median,2) +
        np.power(v[1:num_row-1,j]-v_median,2)))

    # Counter of vectors changed
    change_u = 1/2+np.sign(delta_u-median_magnitude*limit)/2

    # Save final value
    uf[1:num_row-1, j] = u[1:num_row-1, j]+ np.multiply(
            change_u, (u_median - u[1:num_row-1, j]))
    vf[1:num_row-1, j] = v[1:num_row-1,j]+ np.multiply(
            change_u, (v_median - v[1:num_row-1, j]))
    err_vect = err_vect + np.sum(change_u)

    # Change the outlier vectors on the right colum row using with
    # 6-neighbors medians
    j = num_col-1
    u_neigh[0,:] = u[0:num_row-2,j-1]
    u_neigh[1,:] = u[1:num_row-1,j-1]
    u_neigh[2,:] = u[2:num_row,j-1]
    u_neigh[3,:] = u[0:num_row-2,j]
    u_neigh[4,:] = u[1:num_row-1,j]
    u_neigh[5,:] = u[2:num_row,j]

    v_neigh[0,:] = v[0:num_row-2,j-1]
    v_neigh[1,:] = v[1:num_row-1,j-1]
    v_neigh[2,:] = v[2:num_row,j-1]
    v_neigh[3,:] = v[0:num_row-2,j]
    v_neigh[4,:] = v[1:num_row-1,j]
    v_neigh[5,:] = v[2:num_row,j]

    u_median = np.median(u_neigh,0)
    v_median = np.median(v_neigh,0)

    median_magnitude = np.sqrt(np.power(u_median, 2) + np.power(v_median, 2))
    delta_u = (np.sqrt(np.power(u[1:num_row-1, j]-u_median, 2) +
        np.power(v[1:num_row-1, j] - v_median,2)))

    # Counter of vectors changed
    change_u = 1/2 + np.sign(delta_u - median_magnitude * limit)/2

    # Save final value
    uf[1:num_row-1, j] = (u[1:num_row-1,j] + np.multiply(
        change_u, (u_median - u[1:num_row-1, j])))
    vf[1:num_row-1, j] = (v[1:num_row-1,j] + np.multiply(
        change_u, (v_median - v[1:num_row-1, j])))
    err_vect = err_vect + np.sum(change_u)

    # Changes the spurious vector on the lower left corner
    u_median = np.median([u[0,0], u[0,1], u[1,0], u[1,1]])
    v_median = np.median([v[0,0], v[0,1], v[1,0], v[1,1]])
    median_magnitude = np.sqrt(u_median*u_median + v_median*v_median)
    delta_u = np.sqrt((u[0,0]-u_median)**2 +(v[0,0]-v_median)**2)

    if delta_u>median_magnitude*limit:
        uf[0,0] = u_median
        vf[0,0] = v_median
        err_vect = err_vect+1
    else:
        uf[0,0] = u[0,0]
        vf[0,0] = v[0,0]

    # Changes the spurious vector on the upper left corner
    u_median = np.median([u[num_row-1,0], u[num_row-1,1],
        u[num_row-2,0], u[num_row-2,1]])
    v_median = np.median([v[num_row-1,0], v[num_row-1,1],
        v[num_row-2,0], v[num_row-2,1]])
    median_magnitude = np.sqrt(u_median*u_median + v_median*v_median)
    delta_u = np.sqrt((u[num_row-1,0]-u_median)**2 +
            (v[num_row-1,0] - v_median)**2)
    if delta_u>median_magnitude*limit:
        uf[num_row-1,0] = u_median
        vf[num_row-1,0] = v_median
        err_vect = err_vect+1
    else:
        uf[num_row-1,0] = u[num_row-1,0]
        vf[num_row-1,0] = v[num_row-1,0]

    # Changes the spurious vector on the lower right corner
    u_median = np.median([u[0,num_col-1], u[1,num_col-1],
        u[0,num_col-2], u[1,num_col-2]])
    v_median = np.median([v[0,num_col-1], v[1,num_col-1],
        v[0,num_col-2], v[1,num_col-2]])
    median_magnitude = np.sqrt(u_median*u_median + v_median*v_median)
    delta_u = np.sqrt((u[0,num_col-1]-u_median)**2 +
            (v[0,num_col-1]-v_median)**2)

    if delta_u > median_magnitude*limit:
        uf[0,num_col-1] = u_median
        vf[0,num_col-1] = v_median
        err_vect = err_vect + 1
    else:
        uf[0,num_col-1] = u[0,num_col-1]
        vf[0,num_col-1] = v[0,num_col-1]

    # Changes the spurious vector on the upper right corner
    u_median = np.median([u[num_row-1,num_col-1], u[num_row-2,num_col-1],
        u[num_row-2,num_col-2], u[num_row-1,num_col-2]])
    v_median = np.median([v[num_row-1,num_col-1], v[num_row-2,num_col-1],
        v[num_row-2,num_col-2], v[num_row-1,num_col-2]])
    median_magnitude = np.sqrt(u_median*u_median + v_median*v_median)
    delta_u = np.sqrt((u[num_row-1,num_col-1]-u_median)**2 +
            (v[num_row-1,num_col-1]-v_median)**2)

    if delta_u > median_magnitude*limit:
        uf[num_row-1,num_col-1] = u_median
        vf[num_row-1,num_col-1] = v_median
        err_vect=err_vect+1;
    else:
        uf[num_row-1,num_col-1] = u[num_row-1,num_col-1]
        vf[num_row-1,num_col-1] = v[num_row-1,num_col-1]

    return uf, vf, err_vect

def gaussian_filter(Image_1, Image_2, gaussian_size):

    B = gaussian_kernel(gaussian_size)

    # Apply Gaussian filter to particles in the image
    Image_1 = cv2.filter2D(Image_1,-1,B,borderType=cv2.BORDER_CONSTANT)
    Image_2 = cv2.filter2D(Image_2,-1,B,borderType=cv2.BORDER_CONSTANT)

    return Image_1, Image_2

def gaussian_kernel(gaussian_size):
    x = np.arange(1,2*np.ceil(2*gaussian_size).astype(int)+2)
    x = np.exp(-(x-np.ceil(2*gaussian_size)-1)**2/gaussian_size**2)
    B = np.outer(np.transpose(x),x)
    B = B/np.sum(np.sum(B))

    return B


def jacobian_matrix(u, v, x, y, no_box_x, no_box_y):

    # Initialize matrix
    du_dy = np.zeros([no_box_y, no_box_x])
    dv_dy = np.zeros([no_box_y, no_box_x])
    du_dx = np.zeros([no_box_y, no_box_x])
    dv_dx = np.zeros([no_box_y, no_box_x])

    # Calculates the jacobian matrix onto de grid
    delta_x = x[0,1]-x[0,0]
    delta_y = y[1,0]-y[0,0]

    du_dy[1:no_box_y-1,:] = (u[2:no_box_y,:]-u[0:no_box_y-2,:])/2/delta_y
    dv_dy[1:no_box_y-1,:] = (v[2:no_box_y,:]-v[0:no_box_y-2,:])/2/delta_y
    du_dy[0,:] = (u[1,:]-u[0,:])/delta_y
    dv_dy[0,:] = (v[1,:]-v[0,:])/delta_y
    du_dy[no_box_y-1,:] = (u[no_box_y-1,:]-u[no_box_y-2,:])/delta_y
    dv_dy[no_box_y-1,:] = (v[no_box_y-1,:]-v[no_box_y-2,:])/delta_y

    du_dx[:,1:no_box_x-1] = (u[:,2:no_box_x]-u[:,0:no_box_x-2])/(2*delta_x)
    dv_dx[:,1:no_box_x-1] = (v[:,2:no_box_x]-v[:,0:no_box_x-2])/(2*delta_x)
    du_dx[:,0] = (u[:,1]-u[:,0])/delta_x
    dv_dx[:,0] = (v[:,1]-v[:,0])/delta_x
    du_dx[:,no_box_x-1] = (u[:,no_box_x-1]-u[:,no_box_x-2])/delta_x
    dv_dx[:,no_box_x-1] = (v[:,no_box_x-1]-v[:,no_box_x-2])/delta_x

    # Convolution of the image
    k = 1/9*np.ones([3,3])  # Convolution Kernel
    du_dx = cv2.filter2D(du_dx,-1,k,borderType=cv2.BORDER_CONSTANT)
    du_dy = cv2.filter2D(du_dy,-1,k,borderType=cv2.BORDER_CONSTANT)
    dv_dx = cv2.filter2D(dv_dx,-1,k,borderType=cv2.BORDER_CONSTANT)
    dv_dy = cv2.filter2D(dv_dy,-1,k,borderType=cv2.BORDER_CONSTANT)

    return du_dx, du_dy, dv_dx, dv_dy

def interpolations(du_dx,du_dy,dv_dx,dv_dy,u_1,v_1,x_1,y_1,x_2,y_2,no_box):

    # This function interpolate the first grid into the second
    x_1 = x_1.reshape(no_box, order = 'F')
    y_1 = y_1.reshape(no_box, order = 'F')
    u_1 = u_1.reshape(no_box, order = 'F')
    v_1 = v_1.reshape(no_box, order = 'F')
    du_dx = du_dx.reshape(no_box, order = 'F')
    du_dy = du_dy.reshape(no_box, order = 'F')
    dv_dx = dv_dx.reshape(no_box, order = 'F')
    dv_dy = dv_dy.reshape(no_box, order = 'F')

    x_2, y_2 = np.meshgrid(x_2, y_2)

    u_2 = interpolate.griddata((x_1, y_1), u_1, (x_2, y_2), method='linear')
    v_2 = interpolate.griddata((x_1, y_1), v_1, (x_2, y_2), method='linear')
    du_dx = interpolate.griddata((x_1,y_1), du_dx, (x_2,y_2), method='linear')
    du_dy = interpolate.griddata((x_1,y_1), du_dy, (x_2,y_2), method='linear')
    dv_dx = interpolate.griddata((x_1,y_1), dv_dx, (x_2,y_2), method='linear')
    dv_dy = interpolate.griddata((x_1,y_1), dv_dy, (x_2,y_2), method='linear')

    return du_dx, du_dy, dv_dx, dv_dy, u_2, v_2, x_2, y_2

def translated_pixels(i_index, j_index, u_index, v_index, Width, Height,
        box_size_x, box_size_y):

    # Define position of translated pixel
    if ((np.max(np.max(i_index+np.abs(u_index/2)))<Width-2) and
            (np.min(np.min(i_index-np.abs(u_index/2)))>3)):
        i_index_1 = i_index-u_index/2
        i_index_2 = i_index+u_index/2
    else:
        i_index_1 = i_index
        i_index_2 = i_index
        u_index = np.zeros([box_size_y,box_size_x])

    if ((np.max(np.max(j_index+np.abs(v_index/2)))<Height-2) and
            (np.min(np.min(j_index-np.abs(v_index/2)))>3)):
        j_index_1 = j_index-v_index/2
        j_index_2 = j_index+v_index/2
    else:
        j_index_1 = j_index
        j_index_2 = j_index
        v_index = np.zeros([box_size_y,box_size_x])

    # Define pixels position on translated Sub_image1
    i_frac_1 = (i_index_1-np.floor(i_index_1))
    j_frac_1 = (j_index_1-np.floor(j_index_1))

    # Define pixels position on translated Sub_image2
    i_frac_2 = (i_index_2-np.floor(i_index_2))
    j_frac_2 = (j_index_2-np.floor(j_index_2))

    index_floor_1 = np.floor(j_index_1) + Height * np.floor(i_index_1)
    index_floor_2 = np.floor(j_index_2) + Height * np.floor(i_index_2)

    return (i_frac_1, j_frac_1, i_frac_2, j_frac_2, j_index_1, i_index_1,
            j_index_2, i_index_2)


def deform_image(Img1, Img2, Width, Height, box_origin_x, box_origin_y,
        i_matrix, j_matrix, box_size_x, box_size_y, u, v, du_dx, du_dy,
        dv_dx, dv_dy, i, j):

    # Translate pixels
    i_index=box_origin_x[j,i]+i_matrix
    j_index=box_origin_y[j,i]+j_matrix

    u_index = (u[j,i] * np.ones([box_size_y, box_size_x]) + du_dx[j,i] *
            i_matrix + du_dy[j,i] * j_matrix - (box_size_x)/2 *
            (du_dx[j,i]+du_dy[j,i])*np.ones([box_size_y,box_size_x]))

    v_index = (v[j,i] * np.ones([box_size_y, box_size_x]) + dv_dx[j,i] *
            i_matrix+dv_dy[j,i]*j_matrix - (box_size_y)/2 *
            (dv_dx[j,i]+dv_dy[j,i])*np.ones([box_size_y,box_size_x]))

    # Position of translated pixels of images 1 and 2 according to
    # velocity field
    [i_frac_1, j_frac_1, i_frac_2, j_frac_2, j_index_1,
            i_index_1, j_index_2,i_index_2] = translated_pixels(
            i_index, j_index, u_index, v_index, Width,Height,
            box_size_x, box_size_y)

    # Define sub_image1 deformed according to velocity field
    SubImg1 = (np.multiply(np.multiply(1-j_frac_1,1-i_frac_1),
        Img1[j_index_1.astype(int),i_index_1.astype(int)])
        + np.multiply(np.multiply(1-j_frac_1,i_frac_1),
        Img1[j_index_1.astype(int),i_index_1.astype(int)+1])
        + np.multiply(np.multiply(j_frac_1,1-i_frac_1),
        Img1[j_index_1.astype(int)+1,i_index_1.astype(int)])
        + np.multiply(np.multiply(j_frac_1,i_frac_1),
        Img1[j_index_1.astype(int)+1,i_index_1.astype(int)+1]))
    SubImg1 = SubImg1 - np.sum(np.sum(SubImg1))/(box_size_x*box_size_y)

    # Define sub_image2 deformed according to velocity field
    SubImg2 = (np.multiply(np.multiply(1-j_frac_2,1-i_frac_2),
        Img2[j_index_2.astype(int),i_index_2.astype(int)])
        + np.multiply(np.multiply(1-j_frac_2,i_frac_2),
        Img2[j_index_2.astype(int),i_index_2.astype(int)+1])
        + np.multiply(np.multiply(j_frac_2,1-i_frac_2),
        Img2[j_index_2.astype(int)+1,i_index_2.astype(int)])
        + np.multiply(np.multiply(j_frac_2,i_frac_2),
        Img2[j_index_2.astype(int)+1,i_index_2.astype(int)+1]))
    SubImg2 = SubImg2 - np.sum(np.sum(SubImg2))/(box_size_x*box_size_y)

    return SubImg1, SubImg2, u_index, v_index

def weight_function(i_matrix, j_matrix, box_size_x, box_size_y):

    # Define weighting function
    zeta = i_matrix/box_size_x-0.5-0.5/box_size_x
    eta = j_matrix/box_size_y-0.5-0.5/box_size_y
    Weighting_Func = (9*np.multiply(4*np.power(zeta,2)-4*np.abs(zeta)+1,
        4*np.power(eta,2)-4*np.abs(eta)+1))

    return Weighting_Func

def masking(Img1, Img2):

    Img1 = Parameters.Data.mask * Img1
    Img2 = Parameters.Data.mask * Img2

    return Img1, Img2

def change_mask(SubImg1, SubImg2):

    unmasked_pixels = np.count_nonzero(SubImg1)
    SubImg1[SubImg1 == 0] = sum(sum(SubImg1))/max(unmasked_pixels,1)

    unmasked_pixels = np.count_nonzero(SubImg1)
    SubImg2[SubImg2 == 0] = sum(sum(SubImg2))/max(unmasked_pixels,1)

    return SubImg1, SubImg2

def check_mask(u ,v, mask):
    """
    If the center of the correlation box is inside the mask set
    velocity to 0. Useful for no-slip condition and prevent bleeding
    crossing the mask edge due to median filter
    """
    u = u*mask
    v = v*mask

    return u, v

def load_images(name_img_1, name_img_2):
    """
    Load images for PIV processing. Add a little value to prevent
    any value to be 0 outside the mask.
    """

    Img1 = 0.001 + np.asarray(cv2.cvtColor(cv2.imread(name_img_1),
        cv2.COLOR_BGR2GRAY)).astype(np.float32)

    Img2 = 0.001 + np.asarray(cv2.cvtColor(cv2.imread(name_img_2),
        cv2.COLOR_BGR2GRAY)).astype(np.float32)

    return Img1, Img2

def save(x, y, u, v, filename, option='dpivsoft', Matlab=False, param=False):
    """
    save flow field to a file. Option indicates the saving
    format.

    dpivsof: save in python .npz file using the original
    formating of dpivsoft in matlab

    openpiv: save the field in an ascii file compatible
    with openpiv
    """

    # Scale results
    x = x * Parameters.calibration
    y = y * Parameters.calibration
    u = u * Parameters.calibration/Parameters.delta_t
    v = v * Parameters.calibration/Parameters.delta_t

    if Matlab:
        mdic = {"x":x*1.0,  "y":y*1.0,  "u":u*1.0, "v":v*1.0,
               "calibration": float(Parameters.calibration),
               "delta_t": float(Parameters.delta_t),
               "median_limit": float(Parameters.median_limit),
               "no_calculation_1": float(Parameters.no_iter_1),
               "no_calculation_2": float(Parameters.no_iter_2),
               "box_size_1_x": float(Parameters.box_size_1_x),
               "box_size_1_y": float(Parameters.box_size_1_y),
               "box_size_2_x": float(Parameters.box_size_2_x),
               "box_size_2_y": float(Parameters.box_size_2_y),
               "no_boxes_1_x": float(Parameters.no_boxes_1_x),
               "no_boxes_1_y": float(Parameters.no_boxes_1_y),
               "no_boxes_2_x": float(Parameters.no_boxes_2_x),
               "no_boxes_2_y": float(Parameters.no_boxes_2_y),
               "no_calculation": float(Parameters.no_iter_1),
               "direct_calculation": float(Parameters.direct_calc),
               "gaussian_size": float(Parameters.gaussian_size),
               "window_1_x": float(Parameters.window_1_x),
               "window_1_y": float(Parameters.window_1_y),
               "window_2_x": float(Parameters.window_2_x),
               "window_2_y": float(Parameters.window_2_y),
               "weighting": float(Parameters.weighting),
               "peak_ratio": float(Parameters.peak_ratio),
               "image_width": float(Parameters.Data.width),
               "image_height": float(Parameters.Data.height),
               "mask": float(Parameters.mask)}
        savemat(filename+'.mat', mdic)

    if option == 'dpivsoft':
        if param:
            np.savez(filename, x=x,  y=y,  u=u,  v=v,
                    calibration = Parameters.calibration,
                    delta_t = Parameters.delta_t,
                    median_limit = Parameters.median_limit,
                    gaussian_size = Parameters.gaussian_size,
                    no_calculation_1 = Parameters.no_iter_1,
                    no_calculation_2 = Parameters.no_iter_2,
                    box_size_1_x = Parameters.box_size_1_x,
                    box_size_1_y = Parameters.box_size_1_y,
                    box_size_2_x = Parameters.box_size_2_x,
                    box_size_2_y = Parameters.box_size_2_y,
                    window_1_x = Parameters.window_1_x,
                    window_1_y = Parameters.window_1_y,
                    window_2_x = Parameters.window_2_x,
                    window_2_y = Parameters.window_2_y,
                    weighting = Parameters.weighting,
                    peak_ratio = Parameters.peak_ratio,
                    mask = Parameters.mask,
                    direct_calc = Parameters.direct_calc
                    )
        else:
            np.savez(filename, x=x,  y=y,  u=u,  v=v,
                    calibration = Parameters.calibration)

    elif option == 'openpiv':
        fmt="%8.4f"
        delimiter="\t"

        # Build output array
        out = np.vstack([m.flatten() for m in [x, y, u, v, grid.mask_2]])

        np.savetxt(
            filename,
            out.T,
            fmt=fmt,
            delimiter=delimiter,
            header="x"
            + delimiter
            + "y"
            + delimiter
            + "u"
            + delimiter
            + "v"
            + delimiter
            + "mask",
            )
    else:
        sys.exit("Saving option not found")

