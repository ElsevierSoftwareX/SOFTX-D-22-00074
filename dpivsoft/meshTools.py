import numpy as np
import time
import os
#import vtk
from scipy import interpolate

import dpivsoft.SyIm as SyIm

def CFD2mesh(dirName,dirSave,xmin,xmax,ymin,ymax,dx,dy):
    #Interpolate OpenFoam results into a meshgrid for several calculations
    os.chdir(dirName)
    time = sorted(os.listdir())
    p_ = np.argsort([float(i) for i in time])

    x, y = load_vtk_mesh(fileName_u)
    for i in range(0,len(time)):
        timeName = time[p_[i]]
        fileName_u = dirName+'/'+timeName+'/U_xy.vtk'
        fileName_p = dirName+'/'+timeName+'/p_xy.vtk'
        ux, uy = load_vtk_velocity(fileName_u)
        dum,dum,p = load_vtk_pressure(fileName_p)
        xv = np.arange(xmin,xmax+dx,dx)
        yv = np.arange(ymin,ymax+dy,dy)
        [xv,yv] = np.meshgrid(xv, yv)
        pos = np.where((xv>-0.5) & (xv<0.5) & (yv>-0.5) & (yv<0.5))
        uv = interpolate.griddata((x, y), ux, (xv, yv), method = 'cubic')
        vv = interpolate.griddata((x, y), uy, (xv, yv), method = 'cubic')
        p = interpolate.griddata((x, y), p, (xv, yv), method = 'cubic')
        uv[pos] = 0
        vv[pos] = 0
        p[pos] = 0
        np.savez(dirSave+'/Field'+format(i,'04d'),x=xv,y=yv,u=uv,v=vv,p=p)
        print('Generatin meshgrid:',i,'/',len(time))

    return 1


def readForces(Name):
#Read forces output from openfoam
    fid = open(Name)
    nL = TextLines(Name)
    fid.readline().rstrip()
    line = fid.readline().rstrip()
    line = fid.readline().rstrip()
    line = fid.readline().rstrip()
    line = fid.readline().rstrip()

    Fp = np.zeros([nL-5,3])
    Ff = np.zeros([nL-5,3])
    Mp = np.zeros([nL-5,3])
    Mf = np.zeros([nL-5,3])
    t = np.zeros([nL-5])

    cont=0
    for i in fid:
        line = i.rstrip()
        idx = findOcurrences(line,'(')
        jdx = findOcurrences(line,')')
        t[cont] = float(line[0:int(idx[0])])
        string = line[idx[1]+1:jdx[0]-1].split()
        Fp[cont,:] = [float(i) for i in string]
        string = line[idx[2]+1:jdx[1]-1].split()
        Ff[cont,:] = [float(i) for i in string]
        string = line[idx[5]+1:jdx[4]-1].split()
        Mp[cont,:] = [float(i) for i in string]
        string = line[idx[6]+1:jdx[5]-1].split()
        Mf[cont,:] = [float(i) for i in string]
        cont = cont+1
    return t, Fp, Ff, Mp, Mf

def TextLines(Name):
    fid = open(Name)
    cont = 0
    for i in fid:
        cont = cont+1
    return cont

def findOcurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def genMesh(fileName, dirRes, body, radius, nlayers, N):
    #Generate a non structured finite elements mesh surrounding body

    dirCod = os.getcwd()
    os.chdir(dirRes)
    aux = 10
    with open(fileName+'.geo', 'w') as the_file:

        #Boundary
        the_file.write('Point(1) = {0, 0, 0, '+str(N)+'};\n')
        the_file.write('Point(2) = {'+str(radius)+', 0, 0, '+str(N)+'};\n')
        the_file.write('Point(3) = {'+str(-radius)+', 0, 0, '+str(N)+'};\n')
        the_file.write('Point(4) = {0, '+str(-radius)+', 0, '+str(N)+'};\n')
        the_file.write('Point(5) = {0, '+str(radius)+', 0, '+str(N)+'};\n')

        the_file.write('\n')

        the_file.write('Circle(1) = {5, 1, 2};\n')
        the_file.write('Circle(2) = {2, 1, 4};\n')
        the_file.write('Circle(3) = {4, 1, 3};\n')
        the_file.write('Circle(4) = {3, 1, 5};\n')

        the_file.write('Line Loop(1) = {1, 2, 3, 4};\n')
        the_file.write('Transfinite Line{1, 2, 3, 4} = '+str(nlayers)+'Using Progression 1;')

        #Body
        for i in range(0,len(body)):
            the_file.write('Point('+str(i+aux)+') = {' +
                    str(body[i])[1:-1]+'};\n')
        the_file.write('\n')
        for i in range(0,len(body)-1):
            the_file.write('Line('+str(i+aux)+') = {'+str(i+aux) +
                    ','+str(i+aux+1)+'};\n')
        the_file.write('Line('+str(i+aux+1)+') = {'+str(i+aux+1) +
                ','+str(aux)+'};\n')
        the_file.write('\n')

        the_file.write('Line Loop(' + str(aux) + ') = {')
        for i in range(aux,len(body)+aux-1):
            the_file.write(str(i)+', ')
        the_file.write(str(i+1)+'};\n')
        the_file.write('\n')

        the_file.write('Plane Surface(1) = {1, '+str(aux)+'};\n')
        the_file.write('\n')
        the_file.write('Physical Curve("boundary") = {1,2,3,4};\n')
        the_file.write('\n')
        the_file.write('Physical Curve("down") = {'+str(0+aux)+'};\n')
        the_file.write('\n')
        the_file.write('Physical Curve("right") = {'+str(1+aux)+'};\n')
        the_file.write('\n')
        the_file.write('Physical Curve("up") = {'+str(2+aux)+'};\n')
        the_file.write('\n')
        the_file.write('Physical Curve("left") = {'+str(3+aux)+'};\n')
        the_file.write('\n')
        the_file.write('Physical Surface("My surface") = {'+str(1)+'};')

    os.chdir(dirCod)
    return 1



#=============================================================================
# Load velocity fields from VTK files is temporarily disabled due to
# python3.9 incompatibilities
#=============================================================================
"""
#Import the velocity field from a VTKfile (from Alexey Matveichev)
def load_vtk_velocity(filename):
    if not os.path.exists(filename):
        print('Missing file')
        return None

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.Update()

    data = reader.GetOutput()

    # Mapping data: cell -> point
    mapper = vtk.vtkCellDataToPointData()
    mapper.AddInputData(data)
    mapper.Update()
    mapped_data = mapper.GetOutput()

    # Extracting interpolate point data
    udata = mapped_data.GetPointData().GetArray(0)

    nvls = udata.GetNumberOfTuples()

    ux = np.zeros(int(nvls))
    uy = np.zeros(int(nvls))

    for i in range(0, nvls):
        U = udata.GetTuple(i)
        ux[i] = U[0]
        uy[i] = U[1]

    return ux, uy

#Import the mesh field from a VTKfile (from Alexey Matveichev)
def load_vtk_mesh(filename):
    if not os.path.exists(filename):
        return None
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    data = reader.GetOutput()

    # Extracting triangulation information
    points = data.GetPoints()

    # Extracting interpolate point data
    npts = points.GetNumberOfPoints()

    x = np.zeros(int(npts))
    y = np.zeros(int(npts))

    for i in range(npts):
        pt = points.GetPoint(i)
        x[i] = pt[0]
        y[i] = pt[1]

    return x, y

#Import the pressure field from a VTKfile (from Alexey Matveichev)
def load_vtk_pressure(filename):
    if not os.path.exists(filename):
        print('Missing file')
        return None
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.ReadAllScalarsOn()
    reader.Update()

    data = reader.GetOutput()

    # Extracting triangulation information
    triangles = data.GetPolys().GetData()
    points = data.GetPoints()

    # Mapping data: cell -> point
    mapper = vtk.vtkCellDataToPointData()
    mapper.AddInputData(data)
    mapper.Update()
    mapped_data = mapper.GetOutput()

    # Extracting interpolate point data
    udata = mapped_data.GetPointData().GetArray(0)

    npts = points.GetNumberOfPoints()
    nvls = udata.GetNumberOfTuples()

    x = np.zeros(int(npts))
    y = np.zeros(int(npts))
    p = np.zeros(int(nvls))

    for i in range(npts):
        pt = points.GetPoint(i)
        x[i] = pt[0]
        y[i] = pt[1]

    for i in range(0, nvls):
        temp = udata.GetTuple(i)
        p[i] = temp[0]

    return x, y, p

# To be used in future
def Read_Mesh(dirRes):
    #Load all the data
    os.chdir(dirRes)
    Name = sorted(os.listdir())
    Data = np.load(Name[0])
    [a,b] = Data['x'].shape
    U = np.zeros([a-2,b-2,len(Name)])
    V = np.zeros([a-2,b-2,len(Name)])
    P = np.zeros([a-2,b-2,len(Name)])
    Omega = np.zeros([a-2,b-2,len(Name)])
    for i in range(1,len(Name)):
        Data = np.load(Name[i])
        x = Data['x']
        y = Data['y']
        u = Data['u']
        v = Data['v']
        p = Data['p']
        [X,Y,omega] = post.Vorticity(x,y,u,v,'circulation')
        pos = np.where((X>=-0.5) & (X<=0.5) & (Y>=-0.5) & (Y<=0.5))
        omega[pos] = 0
        u = u[1:len(u[:,1])-1,1:len(u[1,:])-1]
        v = v[1:len(v[:,1])-1,1:len(v[1,:])-1]
        p = p[1:len(p[:,1])-1,1:len(p[1,:])-1]
        U[:,:,i] = u
        V[:,:,i] = v
        P[:,:,i] = p
        Omega[:,:,i] = omega
        print('loading results:',i,'/',len(Name))

    return X,Y,U,V,Omega,P,Name
"""
