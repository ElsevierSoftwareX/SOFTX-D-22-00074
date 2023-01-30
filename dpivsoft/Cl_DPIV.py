import time
import cv2
import numpy as np
import importlib_resources

import reikna.cluda as cluda
import reikna.cluda.dtypes as dtypes
from reikna.core import Transformation, Parameter, Annotation, Type
from reikna.cluda import functions, dtypes
from reikna.fft import FFT, FFTShift
from reikna.cluda.tempalloc import ZeroOffsetManager

import dpivsoft.DPIV as DPIV      #Python PIV implementation
from dpivsoft.Classes  import Parameters
from dpivsoft.Classes  import grid
from dpivsoft.Classes  import GPU

def select_Platform(idx):
    """
    Selection of the device to run opencl calculations.

    idx: identifier int of platform in the computer.

    if idx is a string called "selection", the terminal shows a list
    of all available platforms in the computer to select one.
    """

    dtype = np.complex64
    api = cluda.ocl_api()
    if idx == "selection":
        thr = api.Thread.create(interactive=True)
    else:
        thr = api.Thread.create(idx)

    return thr

def compile_Kernels(thr):
    """
    Compiles all kernels needed for GPU calculation.
    Only needs to be called once.

    Kernels are by default in relative the path: ./GPU_Kernels.
    """

    # Package instalation path
    path = importlib_resources.files("dpivsoft")

    # Split image Kernel
    program = thr.compile(open(path/"GPU_Kernels/Slice.cl").read())
    GPU.Slice = program.Slice

    # Initialize u_index_1 and v_index_1
    program = thr.compile(open(path/"GPU_Kernels/ini_index.cl").read())
    GPU.ini_index = program.ini_index

    # Normalize image Kernel
    program = thr.compile(open(path/"GPU_Kernels/SubMean.cl").read())
    GPU.SubMean = program.SubMean

    program = thr.compile(open(path/"GPU_Kernels/Normalize_Img.cl").read())
    GPU.Normalize = program.Normalize

    # Multiplication Kernel
    program = thr.compile(open(path/"GPU_Kernels/multiply_them.cl").read(),
        render_kwds=dict(ctype1=dtypes.ctype(np.complex64),
        ctype2=dtypes.ctype(np.complex64),
        mul=functions.mul(np.complex64, np.complex64)))
    GPU.multiply_them = program.multiply_them

    # Apply mask Kernel
    program = thr.compile(open(path/"GPU_Kernels/multiply_them.cl").read(),
        render_kwds = dict(ctype1=dtypes.ctype(np.float32),
        ctype2 = dtypes.ctype(bool),
        mul = functions.mul(np.float32, bool)))
    GPU.masking = program.multiply_them

    # Find Maximun Kernel
    program = thr.compile(open(path/"GPU_Kernels/find_peak.cl").read())
    GPU.find_peak = program.find_peak

    # Interpolation Kernel
    program = thr.compile(open(path/"GPU_Kernels/interpolation.cl").read())
    GPU.interpolate = program.Interpolation

    # Jacobian Kernel
    program = thr.compile(open(path/"GPU_Kernels/Jacobian.cl").read())
    GPU.jacobian = program.Jacobian

    # Box blur filter
    program = thr.compile(open(path/"GPU_Kernels/box_blur.cl").read())
    GPU.box_blur = program.box_blur

    # Deform image kernel
    program = thr.compile(open(path/"GPU_Kernels/deform_image.cl").read())
    GPU.deform_image = program.Deform_image

    # Median Filter
    program = thr.compile(open(path/"GPU_Kernels/median_filter.cl").read())
    GPU.Median_Filter = program.Median_Filter

    # Weighting function
    program = thr.compile(open(path/"GPU_Kernels/Weighting.cl").read())
    GPU.Weighting = program.Weighting

    # Gaussian blur
    program = thr.compile(open(path/"GPU_Kernels/gaussian_filter.cl").read())
    GPU.gaussian_filter = program.gaussian_filter

    # directCorrelation
    program = thr.compile(open(path/"GPU_Kernels/directCorrelation.cl").read())
    GPU.directCorrelation = program.directCorrelation

    program = thr.compile(open(path/"GPU_Kernels/find_peak_direct.cl").read())
    GPU.find_peak_direct = program.find_peak_direct

    thr.synchronize()

    return 0

def initialization(width, height, thr):
    """
    Initialize variables in GPU memory
    """

    # Obtain PIV mesh
    grid.generate_mesh(width, height)
    grid.pixels = width*height
    Parameters.Data.height = height
    Parameters.Data.width = width

    # Total number of boxes for global size
    N_boxes_1 = Parameters.no_boxes_1_x*Parameters.no_boxes_1_y
    N_boxes_2 = Parameters.no_boxes_2_x*Parameters.no_boxes_2_y

    # subImg examples to compile fft kernel
    subImg1 = np.zeros((N_boxes_1, Parameters.box_size_1_y,
        Parameters.box_size_1_x), dtype = np.complex64)
    subImg2 = np.zeros((N_boxes_2, Parameters.box_size_2_y,
        Parameters.box_size_2_x), dtype = np.complex64)

    # Array of parameters data
    peak_ratio = int(Parameters.peak_ratio*1000)  #trick to use as int

    data1 = np.array((width, height,
        Parameters.box_size_1_x, Parameters.box_size_1_y,
        Parameters.no_boxes_1_x, Parameters.no_boxes_1_y,
        Parameters.window_1_x, Parameters.window_1_y,
        peak_ratio, Parameters.gaussian_size)).astype(np.int32)
    data2 = np.array((width, height,
        Parameters.box_size_2_x, Parameters.box_size_2_y,
        Parameters.no_boxes_2_x, Parameters.no_boxes_2_y,
        Parameters.window_2_x, Parameters.window_2_y,
        peak_ratio, Parameters.gaussian_size)).astype(np.int32)

    # Send mesh to gpu (only done once)
    GPU.box_origin_x_1 = thr.to_device(grid.box_origin_x_1)
    GPU.box_origin_y_1 = thr.to_device(grid.box_origin_y_1)
    GPU.box_origin_x_2 = thr.to_device(grid.box_origin_x_2)
    GPU.box_origin_y_2 = thr.to_device(grid.box_origin_y_2)

    GPU.x1 = thr.to_device(grid.x_1)
    GPU.y1 = thr.to_device(grid.y_1)
    GPU.x2 = thr.to_device(grid.x_2)
    GPU.y2 = thr.to_device(grid.y_2)

    if Parameters.mask:
        GPU.mask_1 = thr.to_device(grid.mask_1)
        GPU.mask_2 = thr.to_device(grid.mask_2)

    if Parameters.direct_calc:
        GPU.box_origin_x_d = thr.to_device(grid.box_origin_x_d)
        GPU.box_origin_y_d = thr.to_device(grid.box_origin_y_d)
        box_size_x_d = round(Parameters.window_1_x/2)*2+1;
        box_size_y_d = round(Parameters.window_1_y/2)*2+1;
        directCorr = np.zeros((N_boxes_1, box_size_y_d,
            box_size_x_d), dtype = np.float32)

        GPU.size_direct = np.prod(directCorr.shape)
        GPU.directCorr = thr.to_device(directCorr)

    # Send PIV parameters to gpu (only done once)
    GPU.data1 = thr.to_device(data1)
    GPU.data2 = thr.to_device(data2)
    GPU.median_limit = thr.to_device(np.float32(Parameters.median_limit))

    #Temp manager to reduce use of memory
    temp_manager = ZeroOffsetManager(
            thr, pack_on_alloc=True, pack_on_free=False)

    # Initialice all GPU variables for first iteration (only done once)
    GPU.subImg1_1 = temp_manager.array([N_boxes_1,
        Parameters.box_size_1_y, Parameters.box_size_1_x],
        np.complex64)

    GPU.subImg1_2 = temp_manager.array([N_boxes_1,
        Parameters.box_size_1_y, Parameters.box_size_1_x],
        np.complex64, dependencies = [GPU.subImg1_1])

    GPU.subMean1_1 = temp_manager.array(
        [Parameters.no_boxes_1_y, Parameters.no_boxes_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2])

    GPU.subMean1_2 = temp_manager.array(
        [Parameters.no_boxes_1_y, Parameters.no_boxes_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2,
        GPU.subMean1_1])

    GPU.u1 = temp_manager.array(
        [Parameters.no_boxes_1_y, Parameters.no_boxes_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2,
        GPU.subMean1_1, GPU.subMean1_2])

    GPU.v1 = temp_manager.array(
        [Parameters.no_boxes_1_y, Parameters.no_boxes_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2,
        GPU.subMean1_1, GPU.subMean1_2, GPU.u1])

    GPU.u1_f = temp_manager.array(
        [Parameters.no_boxes_1_y, Parameters.no_boxes_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2,
        GPU.subMean1_1, GPU.subMean1_2, GPU.u1, GPU.v1])

    GPU.v1_f = temp_manager.array(
        [Parameters.no_boxes_1_y, Parameters.no_boxes_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2,
        GPU.subMean1_1, GPU.subMean1_2, GPU.u1, GPU.v1, GPU.u1_f])

    GPU.du_dx_1 = temp_manager.array(
        [Parameters.no_boxes_1_y, Parameters.no_boxes_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2,
        GPU.subMean1_1, GPU.subMean1_2, GPU.u1, GPU.v1, GPU.u1_f,
        GPU.v1_f])

    GPU.du_dy_1 = temp_manager.array(
        [Parameters.no_boxes_1_y, Parameters.no_boxes_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2,
        GPU.subMean1_1, GPU.subMean1_2, GPU.u1, GPU.v1, GPU.u1_f,
        GPU.v1_f, GPU.du_dx_1])

    GPU.dv_dx_1 = temp_manager.array(
        [Parameters.no_boxes_1_y, Parameters.no_boxes_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2,
        GPU.subMean1_1, GPU.subMean1_2, GPU.u1, GPU.v1, GPU.u1_f,
        GPU.v1_f, GPU.du_dx_1, GPU.du_dy_1])

    GPU.dv_dy_1 = temp_manager.array(
        [Parameters.no_boxes_1_y, Parameters.no_boxes_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2,
        GPU.subMean1_1, GPU.subMean1_2, GPU.u1, GPU.v1, GPU.u1_f,
        GPU.v1_f, GPU.du_dx_1, GPU.du_dy_1, GPU.dv_dx_1])

    GPU.temp_dx_1 = temp_manager.array(
        [Parameters.no_boxes_1_y, Parameters.no_boxes_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2,
        GPU.subMean1_1, GPU.subMean1_2, GPU.u1, GPU.v1, GPU.u1_f,
        GPU.v1_f, GPU.du_dx_1, GPU.du_dy_1, GPU.dv_dx_1,
        GPU.dv_dy_1])

    GPU.temp_dy_1 = temp_manager.array(
        [Parameters.no_boxes_1_y, Parameters.no_boxes_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2,
        GPU.subMean1_1, GPU.subMean1_2, GPU.u1, GPU.v1, GPU.u1_f,
        GPU.v1_f, GPU.du_dx_1, GPU.du_dy_1, GPU.dv_dx_1,
        GPU.dv_dy_1, GPU.temp_dx_1])

    GPU.u_index_1 = temp_manager.array([N_boxes_1,
        Parameters.box_size_1_y, Parameters.box_size_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2,
        GPU.subMean1_1, GPU.subMean1_2, GPU.u1, GPU.v1,
        GPU.du_dx_1, GPU.du_dy_1, GPU.dv_dx_1,
        GPU.dv_dy_1, GPU.temp_dx_1, GPU.temp_dy_1])

    GPU.v_index_1 = temp_manager.array([N_boxes_1,
        Parameters.box_size_1_y, Parameters.box_size_1_x],
        np.float32, dependencies = [GPU.subImg1_1, GPU.subImg1_2,
        GPU.subMean1_1, GPU.subMean1_2, GPU.u1, GPU.v1,
        GPU.du_dx_1, GPU.du_dy_1, GPU.dv_dx_1,
        GPU.dv_dy_1, GPU.temp_dx_1, GPU.temp_dy_1,
        GPU.u_index_1])

    #Create temp images for gaussian filter if needed
    if (Parameters.gaussian_size):

        kernel = DPIV.gaussian_kernel(Parameters.gaussian_size)
        GPU.kernel = thr.to_device(np.float32(kernel))

        # Images filtered to be used only on first iteration
        GPU.img1_g = temp_manager.array([height, width],
            np.float32, dependencies = [GPU.subImg1_1,
            GPU.subImg1_2, GPU.u_index_1, GPU.v_index_1])
        GPU.img2_g = temp_manager.array([height, width],
            np.float32, dependencies = [GPU.subImg1_1,
            GPU.subImg1_2, GPU.u_index_1, GPU.v_index_1,
            GPU.img1_g])

    # Initialice all GPU variables for second iteration (only done once)
    GPU.subImg2_1 = temp_manager.array([N_boxes_2,
        Parameters.box_size_2_y, Parameters.box_size_2_x],
        np.complex64)

    GPU.subImg2_2 = temp_manager.array([N_boxes_2,
        Parameters.box_size_2_y, Parameters.box_size_2_x],
        np.complex64, dependencies = [GPU.subImg2_1])

    GPU.subMean2_1 = temp_manager.array(
        [Parameters.no_boxes_2_y, Parameters.no_boxes_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2])

    GPU.subMean2_2 = temp_manager.array(
        [Parameters.no_boxes_2_y, Parameters.no_boxes_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2,
        GPU.subMean2_1])

    GPU.u2 = temp_manager.array(
        [Parameters.no_boxes_2_y, Parameters.no_boxes_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2,
        GPU.subMean2_1, GPU.subMean2_2])

    GPU.v2 = temp_manager.array(
        [Parameters.no_boxes_2_y, Parameters.no_boxes_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2,
        GPU.subMean2_1, GPU.subMean2_2, GPU.u2])

    GPU.u2_f = temp_manager.array(
        [Parameters.no_boxes_2_y, Parameters.no_boxes_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2,
        GPU.subMean2_1, GPU.subMean2_2, GPU.u2, GPU.v2, GPU.u1])

    GPU.v2_f = temp_manager.array(
        [Parameters.no_boxes_2_y, Parameters.no_boxes_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2,
        GPU.subMean2_1, GPU.subMean2_2, GPU.u2, GPU.v2, GPU.u2_f, GPU.v1])

    GPU.du_dx_2 = temp_manager.array(
        [Parameters.no_boxes_2_y, Parameters.no_boxes_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2,
        GPU.subMean2_1, GPU.subMean2_2, GPU.u2, GPU.v2])

    GPU.du_dy_2 = temp_manager.array(
        [Parameters.no_boxes_2_y, Parameters.no_boxes_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2,
        GPU.subMean2_1, GPU.subMean2_2, GPU.u2, GPU.v2, GPU.du_dx_2])

    GPU.dv_dx_2 = temp_manager.array(
        [Parameters.no_boxes_2_y, Parameters.no_boxes_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2,
        GPU.subMean2_1, GPU.subMean2_2, GPU.u2, GPU.v2,
        GPU.du_dx_2, GPU.du_dy_2])

    GPU.dv_dy_2 = temp_manager.array(
        [Parameters.no_boxes_2_y, Parameters.no_boxes_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2,
        GPU.subMean2_1, GPU.subMean2_2, GPU.u2, GPU.v2,
        GPU.du_dx_2, GPU.du_dy_2, GPU.dv_dx_2])

    GPU.temp_dx_2 = temp_manager.array(
        [Parameters.no_boxes_2_y, Parameters.no_boxes_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2,
        GPU.subMean2_1, GPU.subMean2_2, GPU.u2, GPU.v2,
        GPU.du_dx_2, GPU.du_dy_2, GPU.dv_dx_2,
        GPU.dv_dy_2])

    GPU.temp_dy_2 = temp_manager.array(
        [Parameters.no_boxes_2_y, Parameters.no_boxes_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2,
        GPU.subMean2_1, GPU.subMean2_2, GPU.u2, GPU.v2,
        GPU.du_dx_2, GPU.du_dy_2, GPU.dv_dx_2,
        GPU.dv_dy_2, GPU.temp_dx_2])

    GPU.u_index_2 = temp_manager.array([N_boxes_2,
        Parameters.box_size_2_y, Parameters.box_size_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2,
        GPU.subMean2_1, GPU.subMean2_2, GPU.u2, GPU.v2,
        GPU.du_dx_2, GPU.du_dy_2, GPU.dv_dx_2,
        GPU.dv_dy_2, GPU.temp_dx_2, GPU.temp_dy_2])

    GPU.v_index_2 = temp_manager.array([N_boxes_2,
        Parameters.box_size_2_y, Parameters.box_size_2_x],
        np.float32, dependencies = [GPU.subImg2_1, GPU.subImg2_2,
        GPU.subMean2_1, GPU.subMean2_2, GPU.u2, GPU.v2,
        GPU.du_dx_2, GPU.du_dy_2, GPU.dv_dx_2,
        GPU.dv_dy_2, GPU.temp_dx_2, GPU.temp_dy_2,
        GPU.u_index_2])

    # Load mask if any
    if Parameters.mask:
        GPU.mask = thr.to_device(Parameters.Data.mask)
    else:
        temp = np.zeros(2)
        GPU.mask = thr.empty_like(temp)

    # Initialize GPU computations for the cross-correlation
    GPU.axes = (1,2)
    GPU.fft = FFT(subImg1, axes=GPU.axes).compile(thr)
    GPU.fftshift = FFTShift(subImg1, axes=GPU.axes).compile(thr)

    GPU.fft2 = FFT(subImg2, axes=GPU.axes).compile(thr)
    GPU.fftshift2 = FFTShift(subImg2, axes=GPU.axes).compile(thr)

    return 0


def processing(img1_name, img2_name, thr):
    """
    Perform a parallelized 2 pass PIV algorithm with window deformation
    executed on openCL.

    Developed by Jorge Aguilar-Cabello

    Inputs:
    -------
    Img1_name: str
        Path to the first Image of the next iteration to be loaded
        asynchronously during runtime

    Img2_name: str
        Path to the second Image of the next iteration to be loaded
        asynchronously during runtime

    thr: openCL object
        Platform where to perform the operations in openCL.

    Parameters: class
        Saved in  "Classes.py" file. It contains all PIV procesing parameters
        to be used in calculations. Parameters can be changed manually or
        loaded from external file by using the classmethod: "readParameters".
        Use $help class for more information about PIV parameters.

    Outputs:
    --------
    GPU: Class
        Saved in "Classes.py". Containing all data stored in GPU memory.
        Following outputs are all included inside this class.

    gx1: GPU 2d array
        x meshgrid on GPU from first sweep.

    gx2: GPU 2d array
        x meshgrid on GPU from second sweep.

    gy1: GPU 2d array
        y meshgrid on GPU from first sweep.

    gy2: GPU 2d array
        y meshgrid on GPU from second sweep.

    gu1: GPU 2d array
        velocity field in x direction on GPU from first sweep

    gu2: GPU 2d array
        velocity field in x direction on GPU from second sweep

    gv1: GPU 2d array
        velocity field in y direction on GPU from first sweep

    gv2: GPU 2d array
        velocity field in y direction on GPU from second sweep
    """

    if Parameters.weighting:
        Parameters.weighting = 0
        print("=======================================================================================")
        print("Warning: There is a bug in weighting function on the GPU. Parameter.weighting set to 0")
        print("=======================================================================================")

    N_boxes_1 = Parameters.no_boxes_1_x*Parameters.no_boxes_1_y
    N_boxes_2 = Parameters.no_boxes_2_x*Parameters.no_boxes_2_y
    N_pixels_1 = N_boxes_1*Parameters.box_size_1_x*Parameters.box_size_1_y
    N_pixels_2 = N_boxes_2*Parameters.box_size_2_x*Parameters.box_size_2_y

    # Set to zero the velocity index, as long as the can be initialized from
    # second pass due to shared memory
    GPU.ini_index(GPU.u_index_1, GPU.v_index_1, local_size = None,
            global_size = N_pixels_1)

    # Mask images if required
    if Parameters.mask:
        # Reserved to implement the mask
        GPU.masking(GPU.img1, GPU.img1, GPU.mask, local_size = None,
                global_size = grid.pixels)
        GPU.masking(GPU.img2, GPU.img2, GPU.mask, local_size = None,
                global_size = grid.pixels)

    # Apply gaussian blur only on first iteration if required
    if Parameters.gaussian_size:
        GPU.gaussian_filter(GPU.img1_g, GPU.img1, GPU.kernel, GPU.data1,
                local_size = None, global_size = grid.pixels)
        GPU.gaussian_filter(GPU.img2_g, GPU.img2, GPU.kernel, GPU.data1,
                local_size = None, global_size = grid.pixels)
        thr.synchronize()

        # Obtain SubImage
        GPU.Slice(GPU.subImg1_1, GPU.img1_g, GPU.box_origin_x_1, GPU.box_origin_y_1,
                GPU.data1, local_size = None, global_size = N_pixels_1)
        GPU.Slice(GPU.subImg1_2, GPU.img2_g, GPU.box_origin_x_1, GPU.box_origin_y_1,
                GPU.data1, local_size = None, global_size = N_pixels_1)

    else:
        # Obtain SubImage
        GPU.Slice(GPU.subImg1_1, GPU.img1, GPU.box_origin_x_1, GPU.box_origin_y_1,
                GPU.data1, local_size = None, global_size = N_pixels_1)
        GPU.Slice(GPU.subImg1_2, GPU.img2, GPU.box_origin_x_1, GPU.box_origin_y_1,
                GPU.data1, local_size = None, global_size = N_pixels_1)

    for i in range(0,Parameters.no_iter_1):
        #First iteration using direct cross correlation if needed
        if not i and Parameters.direct_calc:
            #Direct correlation
            GPU.directCorrelation(GPU.img1, GPU.img2, GPU.directCorr,
                GPU.box_origin_x_d, GPU.box_origin_y_d, GPU.data1,
                local_size = None, global_size = int(GPU.size_direct))

            #Find peak
            GPU.find_peak_direct(GPU.v1, GPU.u1, GPU.directCorr, GPU.data1,
                local_size = None, global_size = N_boxes_1)

            i += 1

        if i:
            # Median Filter
            GPU.Median_Filter(GPU.u1_f, GPU.v1_f, GPU.u1, GPU.v1,
                    GPU.median_limit, GPU.data1, local_size = None,
                    global_size = N_boxes_1)

            # Velocity=0 inside mask to prevent bleeding from median filter
            if Parameters.mask:
                GPU.masking(GPU.u1_f, GPU.u1_f, GPU.mask_1,
                        local_size = None, global_size = N_boxes_1)
                GPU.masking(GPU.v1_f, GPU.v1_f, GPU.mask_1,
                        local_size = None, global_size = N_boxes_1)

            # Jacobian matrix
            GPU.jacobian(GPU.temp_dx_1, GPU.temp_dy_1, GPU.u1_f, GPU.x1, GPU.y1,
                    GPU.data1, local_size = None, global_size = N_boxes_1)
            GPU.box_blur(GPU.du_dx_1, GPU.temp_dx_1, GPU.data1,
                    local_size = None, global_size = N_boxes_1)
            GPU.box_blur(GPU.du_dy_1, GPU.temp_dy_1, GPU.data1,
                    local_size = None, global_size = N_boxes_1)

            GPU.jacobian(GPU.temp_dx_1, GPU.temp_dy_1, GPU.v1_f, GPU.x1, GPU.y1,
                    GPU.data1, local_size = None, global_size = N_boxes_1)
            GPU.box_blur(GPU.dv_dx_1, GPU.temp_dx_1, GPU.data1,
                    local_size = None, global_size = N_boxes_1)
            GPU.box_blur(GPU.dv_dy_1, GPU.temp_dy_1, GPU.data1,
                    local_size = None, global_size = N_boxes_1)

            # Deformed image
            GPU.deform_image(GPU.subImg1_1, GPU.subImg1_2, GPU.img1, GPU.img2,
                    GPU.box_origin_x_1, GPU.box_origin_y_1, GPU.u1_f,
                    GPU.v1_f, GPU.du_dx_1, GPU.du_dy_1, GPU.dv_dx_1,
                    GPU.dv_dy_1, GPU.u_index_1, GPU.v_index_1, GPU.data1,
                    local_size = None, global_size = N_pixels_1)

        # Normalize
        GPU.SubMean(GPU.subMean1_1, GPU.subImg1_1, GPU.data1,
                local_size = None, global_size = N_boxes_1)
        GPU.SubMean(GPU.subMean1_2, GPU.subImg1_2, GPU.data1,
                local_size = None, global_size = N_boxes_1)

        GPU.Normalize(GPU.subImg1_1, GPU.subMean1_1, GPU.data1,
                local_size = None, global_size = N_pixels_1)
        GPU.Normalize(GPU.subImg1_2, GPU.subMean1_2, GPU.data1,
                local_size = None, global_size = N_pixels_1)

        # Weighting if required
        if Parameters.weighting:
            GPU.Weighting(GPU.subImg1_1, GPU.data1, local_size = None,
                    global_size = N_pixels_1)
            GPU.Weighting(GPU.subImg1_2, GPU.data1, local_size = None,
                    global_size = N_pixels_1)

        # FFT2D
        GPU.fft(GPU.subImg1_1, GPU.subImg1_1)
        GPU.fft(GPU.subImg1_2, GPU.subImg1_2)

        # Conjugate
        GPU.subImg1_1 = GPU.subImg1_1.conj()

        # Multiplication
        GPU.multiply_them(GPU.subImg1_1, GPU.subImg1_1, GPU.subImg1_2,
                local_size = None, global_size = N_pixels_1)

        # Inverse transform
        GPU.fft(GPU.subImg1_1, GPU.subImg1_1, inverse=True)

        # FFTShift
        GPU.fftshift(GPU.subImg1_1, GPU.subImg1_1)

        # Find peak
        GPU.find_peak(GPU.v1, GPU.u1, GPU.subImg1_1, GPU.u_index_1,
                GPU.v_index_1, GPU.data1, local_size = None,
                global_size = N_boxes_1)

    # Interpolate velocity results from first mesh
    GPU.interpolate(GPU.u2_f, GPU.u1, GPU.x2, GPU.y2, GPU.x1, GPU.y1,
            GPU.data1, local_size = None, global_size = N_boxes_2)
    GPU.interpolate(GPU.v2_f, GPU.v1, GPU.x2, GPU.y2, GPU.x1, GPU.y1,
            GPU.data1, local_size = None, global_size = N_boxes_2)


    for i in range(0,Parameters.no_iter_2):

        # Jacobian matrix
        GPU.jacobian(GPU.temp_dx_2, GPU.temp_dy_2, GPU.u2_f, GPU.x2, GPU.y2,
                GPU.data2, local_size = None, global_size = N_boxes_2)
        GPU.box_blur(GPU.du_dx_2, GPU.temp_dx_2, GPU.data2, local_size = None,
                global_size = N_boxes_2)
        GPU.box_blur(GPU.du_dy_2, GPU.temp_dy_2, GPU.data2, local_size = None,
                global_size = N_boxes_2)

        GPU.jacobian(GPU.temp_dx_2, GPU.temp_dy_2, GPU.v2_f, GPU.x2, GPU.y2,
                GPU.data2, local_size = None, global_size = N_boxes_2)
        GPU.box_blur(GPU.dv_dx_2, GPU.temp_dx_2, GPU.data2, local_size = None,
                global_size = N_boxes_2)
        GPU.box_blur(GPU.dv_dy_2, GPU.temp_dy_2, GPU.data2, local_size = None,
                global_size = N_boxes_2)

        # Deformed image
        GPU.deform_image(GPU.subImg2_1, GPU.subImg2_2, GPU.img1, GPU.img2,
                GPU.box_origin_x_2, GPU.box_origin_y_2, GPU.u2_f,
                GPU.v2_f, GPU.du_dx_2, GPU.du_dy_2, GPU.dv_dx_2,
                GPU.dv_dy_2, GPU.u_index_2, GPU.v_index_2, GPU.data2,
                local_size = None, global_size = N_pixels_2)

        # Normalize
        GPU.SubMean(GPU.subMean2_1,GPU.subImg2_1,GPU.data2,
               local_size = None, global_size = N_boxes_2)
        GPU.SubMean(GPU.subMean2_2,GPU.subImg2_2,GPU.data2,
                local_size = None, global_size = N_boxes_2)

        GPU.Normalize(GPU.subImg2_1,GPU.subMean2_1,GPU.data2,
                local_size = None, global_size = N_pixels_2)
        GPU.Normalize(GPU.subImg2_2,GPU.subMean2_2,GPU.data2,
                local_size = None, global_size = N_pixels_2)

        # Weighting if required
        if Parameters.weighting:
            GPU.Weighting(GPU.subImg2_1, GPU.data2, local_size = None,
                    global_size = N_pixels_2)
            GPU.Weighting(GPU.subImg2_2, GPU.data2, local_size = None,
                    global_size = N_pixels_2)

        # FFT2D
        GPU.fft2(GPU.subImg2_1, GPU.subImg2_1)
        GPU.fft2(GPU.subImg2_2, GPU.subImg2_2)

        # Conjugate
        GPU.subImg2_1 = GPU.subImg2_1.conj()

        # Multiplication
        GPU.multiply_them(GPU.subImg2_1, GPU.subImg2_1, GPU.subImg2_2,
                local_size=None, global_size = N_pixels_2)

        # Inverse transform
        GPU.fft2(GPU.subImg2_1, GPU.subImg2_1, inverse=True)

        # FFTShift
        GPU.fftshift2(GPU.subImg2_1, GPU.subImg2_1)

        # Find peak
        GPU.find_peak(GPU.v2, GPU.u2, GPU.subImg2_1, GPU.u_index_2,
                GPU.v_index_2, GPU.data2, local_size = None,
                global_size = N_boxes_2)

        # Median Filter
        GPU.Median_Filter(GPU.u2_f, GPU.v2_f, GPU.u2, GPU.v2,
                GPU.median_limit, GPU.data2, local_size = None,
                global_size = N_boxes_2)

        if Parameters.mask:
            # Check if inside mask to prevent bleeding from median filter
            GPU.masking(GPU.u2_f, GPU.u2_f, GPU.mask_2,
                    local_size = None, global_size = N_boxes_2)
            GPU.masking(GPU.v2_f, GPU.v2_f, GPU.mask_2,
                    local_size = None, global_size = N_boxes_2)

    # Load Images of next iteration during runtime
    Img1, Img2 = DPIV.load_images(img1_name, img2_name)

    thr.synchronize()

    #Send next iteration images to the GPU
    GPU.img1 = thr.to_device(Img1)
    GPU.img2 = thr.to_device(Img2)

    return 0
