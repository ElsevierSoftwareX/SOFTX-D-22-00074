#Syntethic image Generation
import numpy as np
import cv2
import random
import os
#import vtk
import shapely.geometry as shapely
from scipy import interpolate

import dpivsoft.meshTools as mt

#Image Parameteres import. Needs to be changed on the class
from dpivsoft.Classes import Synt_Img

# Generate images from a given analytical flow field
def Analytic_Syntetic(dirSave, Name):
    """
    Generates a pair of imnages where the trazers particles moves
    accordly to an analytical flow velocity field. The parameters of
    the generated images are defined on the class Synt_Img (see class
    Synt_Img for more info).
    """

    #Generate trazers
    [trazers_shine, trazers_D, img_noise] = Gen_trazers()

    #Velocity profile function
    [xv, yv, Uv, Vv] = Velocity_Profile()

    N_pixel = Synt_Img.height*Synt_Img.width
    u = Uv.reshape(N_pixel,order = 'C')  #X velocity write on 1-D mode
    v = Vv.reshape(N_pixel,order = 'C')  #Y velocity write on 1-D mode

    [img1,img2] = Img_Generation(u,v,xv,yv,trazers_shine,trazers_D,img_noise)

    #Save Images
    SaveImg(img1,img2,dirSave,Name)
    #Save Data
    np.savez(dirSave+'/Py_profile_' + Name,  x=xv,  y=yv,  u=Uv,  v=Vv)

    return 1

# Generation of the selected analytical velocity field to create the
# PIV image pairs
def Velocity_Profile():

    N_pixel = Synt_Img.width*Synt_Img.height
    vel = Synt_Img.vel

    #Mesh generation
    x = np.linspace(0, Synt_Img.width-1,Synt_Img.width)
    y = np.linspace(0, Synt_Img.height-1,Synt_Img.height)
    xv, yv = np.meshgrid(x, y)

    x = xv.reshape((N_pixel),order = 'C')  #X axis write on 1-D mode
    y = yv.reshape((N_pixel),order = 'C')  #Y axis write on 1-D mode

    if Synt_Img.vel_profile == 'Constant':
        Vv = np.zeros([Synt_Img.height, Synt_Img.width])
        Uv = vel*np.ones([Synt_Img.height, Synt_Img.width])
    elif Synt_Img.vel_profile== 'Couette':
        Vv = np.zeros([height,width])
        Uv = vel*yv/np.max(np.max(yv))
    elif Synt_Img.vel_profile== 'Poiseuille':
        h = np.max(np.max(yv))
        Vv = np.zeros([Synt_Img.height, Synt_Img.width])
        Uv = vel*yv*(h-yv)/h/h
    elif Synt_Img.vel_profile == 'Vortex':
        xv = xv-np.mean(np.mean(xv))
        yv = yv-np.mean(np.mean(yv))
        R0 = 200
        r = np.sqrt(xv**2+yv**2)
        Uv = -vel*(r/R0)/(1+np.power(r/R0,2))*np.sin(np.arcsin(yv/r))
        Vv = vel*(r/R0)/(1+np.power(r/R0,2))*np.cos(np.arccos(xv/r))
    elif Synt_Img.vel_profile == 'Frequency':
        Uv = 2*np.sin(2*np.pi*yv/vel)
        Vv = np.zeros([Synt_Img.height, Synt_Img.width])
    else:
        sys.exit("Velocity profile not found")

    return xv, yv, Uv, Vv

def Custom_Syntetic(dirSave, dirSave_field, dirRes, Name, factor, limits, dt):
    """
    Generates a pair of syntetic images where the trazers particles move
    accordly a custom velocity field loaded from a numpy array.

    The numpy array must be saved in colums as: | x | y | u | v |
    """

    Data = np.load(dirRes+'/'+Name)
    xx = Data[:,0]
    yy = Data[:,1]
    uu = Data[:,3]
    vv = Data[:,4]

    #Limits of the mesh
    x_min = limits[0]
    x_max = limits[1]
    y_min = limits[2]
    y_max = limits[3]

    #Size of the Images
    Synt_Img.width = int((x_max-x_min)/factor)
    Synt_Img.height = int((y_max-y_min)/factor)
    N_pixel = Synt_Img.height*Synt_Img.width

    #Generate trazers
    [trazers_shine, trazers_D, img_noise] = Gen_trazers()

    #Image pixels mesh
    xv = np.linspace(x_min, x_max, Synt_Img.width)
    yv = np.linspace(y_min, y_max, Synt_Img.height)
    [xv,yv] = np.meshgrid(xv, yv)
    xv = xv/factor
    yv = yv/factor

    #Interpolate velocity into each pixel
    Uv = interpolate.griddata((xx/factor, yy/factor), uu/factor*dt,
            (xv, yv), method='linear')
    Vv = interpolate.griddata((xx/factor, yy/factor), vv/factor*dt,
            (xv, yv), method='linear')

    u = Uv.reshape(N_pixel,order = 'C')  #X velocity write on 1-D mode
    v = Vv.reshape(N_pixel,order = 'C')  #Y velocity write on 1-D mode

    [img1,img2] = Img_Generation(u,v,xv,yv,trazers_shine,trazers_D,img_noise)

    #Save Images
    SaveImg(img1,img2,dirSave,Name[0:-4])
    #Save Data
    np.savez(dirSave_field + '/' + Name[0:-4] + 'VelProfile',
            x=xv, y=yv, u=Uv, v=Vv)

    return 1

#Generation of the trazers
def Gen_trazers():

    N_trazers = round(Synt_Img.width*Synt_Img.height*Synt_Img.trazers_density)

    #particles light
    trazers_shine = np.random.triangular(Synt_Img.Shine_m-Synt_Img.d_Shine,
            Synt_Img.Shine_m, Synt_Img.Shine_m+Synt_Img.d_Shine,N_trazers)
    #particles dimaeter
    trazers_D = np.random.triangular(Synt_Img.D_m-Synt_Img.d_D, Synt_Img.D_m,
            Synt_Img.D_m+Synt_Img.d_D, N_trazers)
    #image noise
    img_noise = np.random.triangular(Synt_Img.noise_m-Synt_Img.d_noise,
            Synt_Img.noise_m, Synt_Img.noise_m+Synt_Img.d_noise,
            (Synt_Img.height, Synt_Img.width))

    return trazers_shine, trazers_D, img_noise


#Create the image pair with particles from the velocity field in 1-D
def Img_Generation(u, v, xv, yv, trazers_shine, trazers_D, img_noise):

    #Variables
    width = Synt_Img.width
    height = Synt_Img.height
    trazers_density = Synt_Img.trazers_density

    #randomly distributed trazers inside an image
    trazer = np.array(random.sample(range(0,width*height),
        round(width*height*trazers_density)))
    x_1 = (trazer)%width-1
    y_1 = (trazer)//width-1
    u_1 = u[trazer]
    v_1 = v[trazer]

    #position on second image
    x_2 = x_1 + u_1
    y_2 = y_1 + v_1

    #image inicializate
    img1 = np.zeros([height,width])
    img2 = np.zeros([height,width])

    #particles added to images
    for i in range(0,round(width*height*trazers_density)):
        #Rounded center of particle on first image
        yp = y_1[i].astype(int)
        xp = x_1[i].astype(int)

        #Gaussian distribution bright for each particle on the first image
        img1[yp-4:yp+5,xp-4:xp+5] = (img1[yp-4:yp+5, xp-4:xp+5]+
                trazers_shine[i]*np.exp(-((xv[yp-4:yp+5, xp-4:xp+5]
                -xv[yp,xp])**2+(yv[yp-4:yp+5,xp-4:xp+5]-yv[yp,xp])**2)
                *4/trazers_D[i]))

        #Interpolated position of particle for second image
        x2_p = xv[yp,xp]+u_1[i]
        y2_p = yv[yp,xp]+v_1[i]

        #Rounded center of particle on second image
        yp = y_2[i].astype(int)
        xp = x_2[i].astype(int)

        # Detect if a particle left the image after displacement to
        # enter throught the other side
        if xp < width and yp < height and yp>0 and xp>0:
            # Gaussian distribution bright for each particle on the
            # second image
            img2[yp-4:yp+5, xp-4:xp+5] = (img2[yp-4:yp+5, xp-4:xp+5]+
                    trazers_shine[i]*np.exp(-((xv[yp-4:yp+5, xp-4:xp+5]
                    -x2_p)**2 + (yv[yp-4:yp+5, xp-4:xp+5]-y2_p)**2)
                    *4/trazers_D[i]))
        else:
            if x2_p >= width:
                xp = xp-width
                x2_p = x2_p-width
            if y2_p >= height:
                yp = yp-height
                y2_p = y2_p-width
            if x2_p < 0:
                xp = xp+width
                x2_p = x2_p+width
            if y2_p < 0:
                yp = yp+height
                y2_p = y2_p-width

            # Gaussian distribution bright for each particle on the
            # second image
            img2[yp-4:yp+5,xp-4:xp+5] = (img2[yp-4:yp+5, xp-4:xp+5]+
                    trazers_shine[i]*np.exp(-((xv[yp-4:yp+5, xp-4:xp+5]
                    - x2_p)**2 + (yv[yp-4:yp+5,xp-4:xp+5]-y2_p)**2)
                    * 4/trazers_D[i]))

    #final image adding random noise distribution
    img1 = img1 + img_noise
    img2 = img2 + img_noise

    img1[img1>255] = 255
    img2[img2>255] = 255

    return img1, img2

# This funciton transform velocity field used to generate the testing images
# to avarage value inside the deformation windows introduced from the PIV
def Pix2PIV(Xv, Yv, Uv, Vv, no_boxes_x, no_boxes_y, box_size_1_x,
        box_size_1_y, box_size_2_x, box_size_2_y):

    Ui = np.zeros([no_boxes_y,no_boxes_x])
    Vi = np.zeros([no_boxes_y,no_boxes_x])
    Xi = np.zeros([no_boxes_y,no_boxes_x])
    Yi = np.zeros([no_boxes_y,no_boxes_x])
    box_origin_x = np.zeros([no_boxes_y,no_boxes_x])
    box_origin_y = np.zeros([no_boxes_y,no_boxes_x])

    [Height,Width] = Xv.shape

    x_margin = 3/2*np.amax([box_size_1_x, box_size_2_x])
    y_margin = 3/2*np.amax([box_size_1_y, box_size_2_y])

    Xi = 2 + np.round(np.arange(0, no_boxes_x) *
            (Width-x_margin-6)/(no_boxes_x-1) + x_margin/2)
    Yi = 2 + np.round(np.arange(0, no_boxes_y) *
            (Height-y_margin-6)/(no_boxes_y-1) + y_margin/2)

    box_origin_x = Xi - box_size_2_x/2;
    box_origin_y = Yi - box_size_2_y/2;

    for i in range(0,no_boxes_x):
        for j in range(0,no_boxes_y):

            #Define sub images to work
            Sub_u =(Uv[int(box_origin_y[j]):int(box_origin_y[j])+box_size_2_y,
                int(box_origin_x[i]):int(box_origin_x[i])+box_size_2_x])
            Sub_v =(Vv[int(box_origin_y[j]):int(box_origin_y[j])+box_size_2_y,
                int(box_origin_x[i]):int(box_origin_x[i])+box_size_2_x])

            Ui[j, i] = np.mean(Sub_u)
            Vi[j, i] = np.mean(Sub_v)

    return Xi, Yi, Ui, Vi

#Save Images
def SaveImg(Img1, Img2, dirSave, Name):
    """
    Save the syntetic Images
    """

    cv2.imwrite(dirSave+'/'+Name+'_1'+Synt_Img.ext,Img1)
    cv2.imwrite(dirSave+'/'+Name+'_2'+Synt_Img.ext,Img2)

#============================================================================
# Load velocity fields from VTK files is temporarily disabled due to
# python3.9 incompatibilities
#============================================================================

"""
#Generate images from a given CFD field
def CFD_Syntetic(dirLoad, dirSave, Name, no_Images, limits, factor, dt,
        time, t_idx, object_points):

    #Limits of the mesh
    x_min = limits[0]
    x_max = limits[1]
    y_min = limits[2]
    y_max = limits[3]

    #Size of the Images
    Synt_Img.width = int((x_max-x_min)/factor)
    Synt_Img.height = int((y_max-y_min)/factor)
    N_pixel = Synt_Img.height*Synt_Img.width

    #Folder to load first
    timeName = time[t_idx[0]]
    file_folder = dirLoad+'/'+timeName+'/U_xy.vtk'

    #function to load the CFD mesh
    [x,y] = mt.load_vtk_mesh(file_folder)

    #Create mesh of the image (pixels)
    xv = np.linspace(x_min, x_max, Synt_Img.width)
    yv = np.linspace(y_min, y_max, Synt_Img.height)
    [xv,yv] = np.meshgrid(xv, yv)
    Xv = xv/factor
    Yv = yv/factor

    #Get the pixels position of black object if exists
    if object_points:
        polygon = shapely.Polygon(object_points)

        mesh = np.zeros((Synt_Img.height,Synt_Img.width))
        for j in range(0,len(yv[:,0])):
            for i in range(0,len(xv[0,:])):
                point = shapely.Point((xv[j,i],yv[j,i]))
                mesh[j,i] = polygon.intersects(point)
        pos = np.where(mesh)
        del mesh


    for i in range(0,len(time)):
        #direc = dirSave+'/'+str(timeName[i])
        #os.makedirs(direc, exist_ok=True)
        timeName = time[t_idx[i]]
        print(timeName)
        for j in range(0,no_Images):

            #Generate trazers
            [trazers_shine, trazers_D, img_noise] = Gen_trazers()

            #load data from CFD
            file_folder = dirLoad+'/'+timeName+'/U_xy.vtk'
            ux, uy = mt.load_vtk_velocity(file_folder)

            #function ti fix that velocity profile into the image
            Uv = interpolate.griddata((x, y), ux, (xv, yv), method='linear')
            Vv = interpolate.griddata((x, y), uy, (xv, yv), method='linear')

            if object_points:
                #Velociti on the object set to null
                Uv[pos] = 0
                Vv[pos] = 0

            #u and v changed to 1D
            u = Uv.reshape(N_pixel,order = 'C')*factor/dt
            v = Vv.reshape(N_pixel,order = 'C')*factor/dt

            #Generate Image
            [img1,img2] = Img_Generation(u, v, Xv, Yv, trazers_shine,
                    trazers_D,img_noise)

            if object_points:
                #Black body if exist
                img1[pos] = 0
                img2[pos] = 0

            #Save Images
            SaveImg(img1,img2,dirSave,format(i,'05d')+Name+str(j))

    return 1
"""
