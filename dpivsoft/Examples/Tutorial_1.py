"""
This Tutorial cover most of basic functions of DPIVSoft.

    1-  To generate a pair of Syntetic_Images from Analytical flow solutions.

    2-  Set the processing parameters (In this simple tutorial the parameters file
        needs to be in same path than the code, this can be easily changed)

    3-  To process the images using the cpu python implementation and the much more
        faster GPU openCL implementation.

    4-  Load and save results in different formats.
"""

# Standar libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os

# DPIVSoft libraries
import dpivsoft.DPIV as DPIV      #Python PIV implementation
import dpivsoft.Cl_DPIV as Cl_DPIV   #OpenCL PIV implementation
import dpivsoft.SyIm as SyIm  #Syntetic images generator

from dpivsoft.Classes  import Parameters
from dpivsoft.Classes  import grid
from dpivsoft.Classes  import GPU
from dpivsoft.Classes  import Synt_Img

#=============================================================================
# WORKING FOLDERS
#=============================================================================
dirCode = os.getcwd()   #Current path
dirImg = dirCode + "/Images/Tutorial_1"   #Images folder
dirRes = dirCode + "/Results/Tutorial_1"  #Results folder

if not os.path.exists(dirImg):
    os.makedirs(dirImg)
if not os.path.exists(dirRes):
    os.makedirs(dirRes)

#=============================================================================
# SYNTETIC IMAGES TO PERFORM TEST
#=============================================================================
SyIm.Analytic_Syntetic(dirImg, "Test_Img_")

#=============================================================================
# LIST OF IMAGES TO PROCESS
#=============================================================================
files = os.listdir(dirImg)
files = sorted([i for i in files if i.endswith('.png')])
print(files)

#=============================================================================
# SET PIV PARAMETERS
#=============================================================================
# 1: Set parameters manually (see Classes.py for more details):
Parameters.box_size_2_x = 32

# 2: Arternateively parameters can be load from a file using readParamters():
Parameters.readParameters(dirCode+'/Tutorial_1_parameters.yaml')


#=============================================================================
# PYTHON PROCESSING
#=============================================================================
start = time.time()
# Loop for load all images (only one in the example)
for i in range(0,len(files),2):

    # Name of the images
    name_img_1 = dirImg+'/'+files[i]
    name_img_2 = dirImg+'/'+files[i+1]

    # Load images
    Img1, Img2 = DPIV.load_images(name_img_1, name_img_2)

    [height, width] = Img1.shape

    # PIV processing python
    [x2, y2, u2, v2] = DPIV.processing(Img1, Img2)
    # Save results in ASCII file compatible with openPIV format
    saveName = dirRes + '/cpu_field_' + format(int(i/2), '03d')
    DPIV.save(x2, y2, u2, v2, saveName, 'openpiv')

print("Python algorithm finished. Time = ", time.time()-start, "s")

#=============================================================================
# OPENCL PROCESSING (The same but much faster)
#=============================================================================

# Select platform (only needed once). If more than one platform is installed
# use "selection"
thr = Cl_DPIV.select_Platform(0)
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# Load first pair of images to start the computation and set arrays sizes
name_img_1 = dirImg+'/'+files[0]
name_img_2 = dirImg+'/'+files[1]
Img1, Img2 = DPIV.load_images(name_img_1, name_img_2)
[height, width] = Img1.shape

# Send first pair of images to GPU
GPU.img1 = thr.to_device(Img1)
GPU.img2 = thr.to_device(Img2)

# Compile kernels and initialize variables (only needed once)
Cl_DPIV.compile_Kernels(thr)
Cl_DPIV.initialization(width, height, thr)

start = time.time()
# Loop for load all images (only one in the example)
for i in range(0,len(files),2):

    # Change the name of next iteration Images only if needed
    if i<len(files)-3:
        #Name of next iteration Images
        name_img_1 = dirImg + '/' + files[i+2]
        name_img_2 = dirImg + '/' + files[i+3]

    # Process images. (Next iteration Images path is send to be loaded
    # in parallel during runtime)
    Cl_DPIV.processing(name_img_1, name_img_2, thr)

    # Get final results from GPU
    x2 = GPU.x2.get()
    y2 = GPU.y2.get()
    u2 = GPU.u2_f.get()
    v2 = GPU.v2_f.get()

    # Save results in numpy file compatible with DPIVSoft format
    saveName = dirRes + '/gpu_field_' +  format(int(i/2), '03d')
    DPIV.save(x2, y2, u2, v2, saveName)

print("OpenCl algorithm finished. Time = ", time.time()-start, "s")

#=============================================================================
# WORK WITH RESULTS
#=============================================================================
# Load PIV results
Data = np.load(dirRes+'/gpu_field_000.npz')
x = Data['x']
y = Data['y']
u = Data['u']
v = Data['v']

fig, ax1 = plt.subplots()
ax1.quiver(x, y, u, v, scale=1 / 0.003)
ax1.set_xlabel('x (pixels)',fontsize=18)
ax1.set_ylabel('y (pixels)',fontsize=18)
plt.show()
