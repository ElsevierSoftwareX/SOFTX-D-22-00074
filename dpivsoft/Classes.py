import numpy as np
import yaml
import cv2

class Parameters:
    """
    Contains all parameters options to perform PIV

    Default values are defined here. There are two options to change the
    values for a specific DPIV run:

    - Change the specific value manually on python shell or on the running
      script.

            Example:  Parameters.box_size_1_x = 128

    - Load all the parameters included in a yaml file by using readPArameters
      classmethod.

            Example: Parameters.readParameters('folder/filename')
    """

    #Default first step parameters
    box_size_1_x = 64   #Cross-correlation box1
    box_size_1_y = 64   #Cross-correlation box1
    no_boxes_1_x = 64   #Number of x-windows
    no_boxes_1_y = 32   #Number of y-windows
    window_x_1 = 48
    window_y_1= 48

    #Default second step parameters
    box_size_2_x = 32   #Cross-correlation box 2
    box_size_2_y = 32   #Cross-correlation box 2
    no_boxes_2_x = 128   #number of x-Windows
    no_boxes_2_y = 64   #Number of y windows
    window_x_2 = 32
    window_y_2 = 32

    #Number of pass of first step
    no_iter_1 = 1
    #Number of pass of second step
    no_iter_2 = 2

    #Direct calculation or FFT
    direct_calc = 0  #1=direct; 0=FFT

    #default general parameters
    mask = 0
    peak_ratio = 1
    weighting = 0
    gaussian_size = 0
    median_limit = 0.5
    calibration = 1
    delta_t = 1

    #Extra data needed in some cases
    class Data:
        #path of mask images
        path_mask = "none"

    @classmethod
    def readParameters(self,fileName):
        """
        Read parameters from a yaml file
        """
        with open(fileName) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

            try:
                #Default first step parameters
                self.box_size_1_x = data['box_size_1_x']
                self.box_size_1_y = data['box_size_1_y']
                self.no_boxes_1_x = data['no_boxes_1_x']
                self.no_boxes_1_y = data['no_boxes_1_y']
                self.window_1_x =   data['window_1_x']
                self.window_1_y =   data['window_1_y']

                #Number of pass of first step
                self.no_iter_1 = data['no_iter_1']
                self.no_iter_2 = data['no_iter_2']

                #Direct calculation or FFT
                self.direct_calc = data['direct_calc']

                #Default first step parameters
                self.box_size_2_x = data['box_size_2_x']
                self.box_size_2_y = data['box_size_2_y']
                self.no_boxes_2_x = data['no_boxes_2_x']
                self.no_boxes_2_y = data['no_boxes_2_y']
                self.window_2_x =   data['window_2_x']
                self.window_2_y =   data['window_2_y']

                #default general parameters
                self.mask = data['mask']
                self.peak_ratio = data['peak_ratio']
                self.weighting = data['weighting']
                self.gaussian_size = data['gaussian_size']
                self.median_limit = data['median_limit']
                self.calibration = data['calibration']
                self.delta_t = data['delta_t']

                #Extra data
                if self.mask:
                    if data['path_mask'].endswith('.np'):
                        self.Data.mask = bool(np.load(data['path_mask']))
                    else:
                        self.Data.mask = np.asarray(cv2.cvtColor(cv2.imread(
                            data['path_mask']), cv2.COLOR_BGR2GRAY)).astype(bool)

            except:
                print('Error: Wrong parameters definition. Some Parameters'
                      ' have been renamed in order to make them'
                      ' more consistent with the matlab DPIVSoft version.'
                      ' Please rename the following parameters as indicated:')
                print(" - Window_axis_number --> window_number_axis")
                print(" - gaussian_filter --> gaussian_size")
                print(" - factor --> calibration")
                print(" - dt --> delta_t")
                print("You can also check the new parameters definition in "
                        "Classes.py file or copy them from the examples. "
                        "Sorry for the inconvenience.")
                exit()
    def introParameters():
        """
        Introduce a parameter manually, not implemented yet (probably
        needed for a GUI)
        """
        pass

class grid:

    @classmethod
    def generate_mesh(cls,width,height):

        """
        Generates the meshgrid of x and y position for the correlation
        windows in the two passes, according to the piv parameters selected.
        """
        pixels = width*height
        no_boxes_1_x = Parameters.no_boxes_1_x
        no_boxes_1_y = Parameters.no_boxes_1_y
        box_size_1_x = Parameters.box_size_1_x
        box_size_1_y = Parameters.box_size_1_y
        no_boxes_2_x = Parameters.no_boxes_2_x
        no_boxes_2_y = Parameters.no_boxes_2_y
        box_size_2_x = Parameters.box_size_2_x
        box_size_2_y = Parameters.box_size_2_y


        #Obtain PIV Mesh for calculationsFp_Top + Fp_Down
        box_origin_x_1 = (1+np.round((np.arange(0,no_boxes_1_x)
            *(width-box_size_1_x-2))/(no_boxes_1_x-1)).astype(np.int32))
        box_origin_y_1 = (1+np.round((np.arange(0,no_boxes_1_y)
            *(height-box_size_1_y-2))/(no_boxes_1_y-1)).astype(np.int32))

        x_1 = (box_origin_x_1-1+box_size_1_x/2).astype(np.int32)
        y_1 = (box_origin_y_1-1+box_size_1_y/2).astype(np.int32)

        [x_1, y_1] = np.meshgrid(x_1,y_1)
        [box_origin_x_1, box_origin_y_1] = np.meshgrid(
                box_origin_x_1, box_origin_y_1)

        # Obtain special mesh for direct calculation if needed
        if Parameters.direct_calc:
            window_x = Parameters.window_x_1
            window_y = Parameters.window_y_1

            x_1 = (1 + np.round((window_x/2+box_size_1_x)/2) +
                np.round(np.arange(0, no_boxes_1_x) *
                (width-box_size_1_x-window_x/2-4)/(no_boxes_1_x-1))
            )
            y_1 = (1 + np.round((window_y/2+box_size_1_y)/2) +
                np.round(np.arange(0, no_boxes_1_y) *
                (height-box_size_1_y-window_y/2-4)/(no_boxes_1_y-1))
            )
            [x_1, y_1] = np.meshgrid(x_1,y_1)

            box_origin_x_d = x_1+1-round(box_size_1_x/2)
            box_origin_y_d = y_1+1-round(box_size_1_y/2)

            cls.box_origin_x_d = box_origin_x_d.astype(np.int32)
            cls.box_origin_y_d = box_origin_y_d.astype(np.int32)


        x_margin = 3/2 * np.amax([box_size_1_x, box_size_2_x])
        y_margin = 3/2 * np.amax([box_size_1_y, box_size_2_y])

        #second grid is placed completely inside first one
        x_2 = np.int32(2 + np.round(np.arange(0, no_boxes_2_x) *
            (width-x_margin-6)/(no_boxes_2_x-1)+x_margin/2))
        y_2 = np.int32(2 + np.round(np.arange(0, no_boxes_2_y) *
            (height-y_margin-6)/(no_boxes_2_y-1)+y_margin/2))

        [x_2, y_2] = np.meshgrid(x_2, y_2)

        box_origin_x_2 = (x_2-box_size_2_x/2).astype(np.int32)
        box_origin_y_2 = (y_2-box_size_2_y/2).astype(np.int32)

        cls.x_1 = x_1.astype(np.int32)
        cls.y_1 = y_1.astype(np.int32)
        cls.x_2 = x_2.astype(np.int32)
        cls.y_2 = y_2.astype(np.int32)
        cls.box_origin_x_1 = box_origin_x_1.astype(np.int32)
        cls.box_origin_y_1 = box_origin_y_1.astype(np.int32)
        cls.box_origin_x_2 = box_origin_x_2.astype(np.int32)
        cls.box_origin_y_2 = box_origin_y_2.astype(np.int32)


        #Create mask mesh if needed
        cls.mask_1 = np.full(x_1.shape, False)
        cls.mask_2 = np.full(x_2.shape, False)
        if Parameters.mask:
            for j in range(0,len(x_1[:,0])):
                for i in range(0,len(x_1[0,:])):
                    if Parameters.Data.mask[int(y_1[j,i]),int(x_1[j,i])]:
                        cls.mask_1[j,i] = True
            for j in range(0,len(x_2[:,0])):
                for i in range(0,len(x_2[0,:])):
                    if Parameters.Data.mask[int(y_2[j,i]),int(x_2[j,i])]:
                        cls.mask_2[j,i] = True

    def read_mesh(self,height,width):
        """
        Read the positions of a custon mesh from a file (not implemented yet)
        """
        pass

class Synt_Img():
    """
    Contains parameters of to generate a pair of sintetic images following
    a certain flow velocity field. The default values can be changed manually.

    Variables:
    ----------
    Width: int
        Number of pixels on the x-direction of the generated image.

    height: int
        Number of pixels on the y-direction of the generated image.

    trazers_density: float
        Number of particles generated per pixel. Needs to be some float
        between 0 and 1

    vel: float
        Caracteristic velocity of the canonical flow used for generate the
        particles displacement.

        Note: It has a diferent definition on each flow.

    Shine_m: int
        Mean shine of the trazers core (from 0 to 255).

    d_Shine: int
        Triangular variability of the trazers shine.

    D_m: int or float
        Mean diameter of the trazers.

    d_D: int or float
        triangular variability of the trazers diameter.

    noise_m: int
        mean of the random noise.

    d_noise: int
        triangular variability of the random noise.

    vel_profile: str
        Name of the flow velocity field used. The following options are
        avaiable:

            Constant: Flow with a constant displacement of "vel" pixels on
                      the x-direction

            Couette: Couette flwo in x-direction using a moving condition
                     of "vel" pixels between images on top wall. Bottom
                     limit of the images is not moving.

            Poiseuille: Poiseuille flow in x direction with no-slip condition
                        on top and bottom of the images. The parameter "vel"
                        indicates de maximun velicity of the flow in pixels.

            Vortex: Flow generated by a Scully vortex (see Scully 1975). Vel
                    is the maximun velocity of the vortex in pixel/frame.

            Frequency: Spatial frequency wave along y-direction. This flow
                       allows to test the PIV frequency response (see akfda
                       1992). In this case, the maximun pixel displacement
                       on the wave is always 2 pixels, and the parameter
                       "vel" indicates the wavelength.

    ext: str
        Extension of the images to be saved

    """

    width = 1024               #Width of generated image
    height = 1024              #height of generated image
    vel_profile = 'Vortex'     #(Constant, Couette, Poiseuille, Rankine)
    vel = 8                    #Velocity in pixels/fotogram
    trazers_density = 0.05     #Number of trazers/ pixel
    Shine_m = 230              #Mean shine of trazers
    d_Shine = 80               #Variability for randon shine of trazers
    D_m = 6                    #Mean diameter of trazers (in pixels)
    d_D = 3                    #Variability for random diameter of trazers
    noise_m = 1                #Mean white noise of the image
    d_noise = 1                #Variability of the noise of the image
    ext = '.png'               #Image save format

class GPU():

    def gpu_data(self,thr):
        pass
