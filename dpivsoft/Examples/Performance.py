"""
This example test the performance of DPIV algorithm as long as the images size increase comparing
the processing time of python CPU and openCL implementations.

The correlation windows used are of 32x32 and 16x16 pixels in first and second pass respectively.

The correlation windows are placed in order to mantain a window overlap of 50% so that the number of
correlation windows is proportional to the Image size.

The syntetic images used for the test are generated from John Howkings CFD simulation of bouyancy
driven mixing flow.
"""

#Standar libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os

#DPIVSoft libraries
import dpivsoft.DPIV as DPIV  #Python PIV implementation
import dpivsoft.Cl_DPIV as Cl_DPIV   #OpenCL PIV implementation
import dpivsoft.SyIm as SyIm  #Syntetic images generator

from dpivsoft.Classes  import Parameters
from dpivsoft.Classes  import grid
from dpivsoft.Classes  import GPU
from dpivsoft.Classes  import Synt_Img

#=============================================================================
#WORKING FOLDERS
#=============================================================================
dirCode = os.getcwd()   #Current path
dirCFD = dirCode + "/CFD"
dirImg = dirCode + "/Images/Performance"   #Images folder
if not os.path.exists(dirImg):
    os.makedirs(dirImg)

#============================================================================
#SYNTETIC IMAGES FOR TEST (BOUYANCY SIMULATION FROM JOHN HOPKINGS UNIVERSITY)
#============================================================================
CFD_name = "Mixing_Flow.npy"
Img_dimension = np.array([128,256,512,1024,2048])

limits = [0,6.28,0,6.28]
factor = float(limits[1])/Img_dimension
dt = 0.02

print("Generating Syntetic images ....")
for i in range(0,len(Img_dimension)):
    print("...Img",str(i))
    Synt_Img.width = Img_dimension[i]
    Synt_Img.height = Img_dimension[i]

    dirSave = dirImg + "/" + str(i)

    if not os.path.exists(dirSave):
        os.makedirs(dirSave)

    SyIm.Custom_Syntetic(dirSave, dirSave, dirCFD, CFD_name,
            factor[i], limits, dt)

#=============================================================================
#PROCESSING ON GPU
#=============================================================================
#Number of iterations to obtain mean processing time
N_GPU = 50
N_CPU = 5    #The CPU iterations are lower because CPU processing is too slow

print("PIV processing......")
Parameters.readParameters(dirCode+'/Performance_parameters.yaml')

temp_GPU = np.zeros(len(Img_dimension))
temp_CPU = np.zeros(len(Img_dimension))
for i in range(0,len(Img_dimension)):
    dirSave = dirImg + "/" + str(i)

    #Number of boxes to obtain 50% overlap
    Parameters.no_boxes_1_x = int(2*Img_dimension[i]/32)
    Parameters.no_boxes_1_y = int(2*Img_dimension[i]/32)
    Parameters.no_boxes_2_x = int(2*Img_dimension[i]/16)
    Parameters.no_boxes_2_y = int(2*Img_dimension[i]/16)

    #Platform selection
    thr = Cl_DPIV.select_Platform(0)
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    #Load first pair of images to start the computation and set arrays sizes
    name_img_1 = dirSave + '/Mixing_Flow_1.png'
    name_img_2 = dirSave + '/Mixing_Flow_2.png'

    Img1, Img2 = DPIV.load_images(name_img_1, name_img_2)
    [height, width] = Img1.shape

    #Send first pair of images to GPU
    GPU.img1 = thr.to_device(Img1)
    GPU.img2 = thr.to_device(Img2)

    #Compile kernels and initialize variables (only needed once)
    Cl_DPIV.compile_Kernels(thr)
    Cl_DPIV.initialization(width, height, thr)

    #GPU performance test
    start = time.time()
    for j in range(0,N_GPU):

        #Name of the images of the next iteration
        name_img_1 = dirSave + '/Mixing_Flow_1.png'
        name_img_2 = dirSave + '/Mixing_Flow_2.png'

        # Process images. (Next iteration Images path is send to be loaded
        # in parallel during runtime)
        Cl_DPIV.processing(name_img_1, name_img_2, thr)

        gx2 = GPU.x2.get()
        gy2 = GPU.y2.get()
        gu2 = GPU.u2_f.get()
        gv2 = GPU.v2_f.get()

    temp_GPU[i] = (time.time()-start)/N_GPU
    print("OpenCl processing time per Image:", temp_GPU[i])
    thr.release()

    #CPU performance test
    start = time.time()
    for j in range(0,N_CPU):
        #Name of the images
        name_img_1 = dirSave + '/Mixing_Flow_1.png'
        name_img_2 = dirSave + '/Mixing_Flow_2.png'

        #Load images
        Img1, Img2 = DPIV.load_images(name_img_1, name_img_2)

        #PIV processing
        x, y, u ,v = DPIV.processing(Img1, Img2)

    temp_CPU[i] = (time.time()-start)/N_CPU
    print("Python processing time per Image: ", temp_CPU[i])
    print("======================================================")

fig1, ax1 = plt.subplots()
ax1.plot(Img_dimension**2,temp_CPU,'s-',Img_dimension**2,temp_GPU,'^-')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend(['python','openCL'])
ax1.set_xlabel('Image number of pixels')
ax1.set_ylabel('Image pair processing time [s]')

fig2, ax2 = plt.subplots()
ax2.plot(Img_dimension**2,temp_CPU/temp_GPU,'s-')
ax2.set_xlabel('Image number of pixels')
ax2.set_ylabel('OpenCL speedup')
ax2.set_xscale('log')
plt.show()
