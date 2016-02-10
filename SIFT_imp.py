#===============Scale Invariant Feature Transform=============================#
import math
from scipy import ndimage
from PIL import Image
from numpy import *
from matplotlib import pyplot as plt
from pylab import *
import scipy.stats
import cv2
#1278
#----------------------Find the ocatve---------------------------start-------#
def sift_imp(I):    
    s = 1.6 # set the standard deviation
    k = sqrt(2) # set the constant multiplicative factor
    row = len(I[0,:])
    clm = len(I[:,0])
    def find_octave (I, # Input image
    s, # Standard Deviation Value
    k # Constant multiplicative factor value
    ):
   	L5 = cv2.GaussianBlur(I,(5,5),s*(k**4)) # Laplace of Gaussian Layer 5
   	L4 = cv2.GaussianBlur(I,(5,5),s*(k**3)) # Laplace of Gaussian Layer 4
   	L3 = cv2.GaussianBlur(I,(5,5),s*(k**2)) # Laplace of Gaussian Layer 3
   	L2 = cv2.GaussianBlur(I,(5,5),s*k) # Laplace of Gaussian Layer 2
   	L1 = cv2.GaussianBlur(I,(5,5),s) # Laplace of Gaussian Layer 1
   	DOG4 = array(L5-L4) # Difference of Gaussian layer 4
   	DOG3 = array(L4-L3) # Difference of Gaussian layer 3
   	DOG2 = array(L3-L2) # Difference of Gaussian layer 2
   	DOG1 = array(L2-L1) # Difference of Gaussian layer 1
   	
   	return DOG1,DOG2,DOG3,DOG4
    #----------------------Find the ocatve---------------------------end-------#
    
    #---------------------Find the Extrema----------------------start----------#
    def find_ext( # the function to find the extrema's og gaussian
    DOG3, # The Difference of Guassian layer above the interested one
    DOG2, # The Difference of Gaussian layer that we are interested in
    DOG1 # The Difference of Guassian layer below the interested one 
    ):
   	
   	M2 = DOG3 # copy GoA3
   	M = DOG2 # copy GoA2
   	M1 = DOG1 # copy GoA1
   	x_ex_oc1 = []
   	y_ex_oc1 = []
   	IDoG = []
   	#print len(range(1,len(M[:-1,0]))), len(range(1,len(M[0,:-1])))#
   	for i in range(1,len(M[:-1,0])):
  		for j in range(1,len(M[0,:-1])):
  		    # Now compare it with (the neighbouring pixels, the pixels of interest from layer above and the pixels 
  		    #of interest from layer below)
 			compare = (M[i-1,j+1],M[i,j+1],M[i+1,j+1],M[i-1,j],M[i+1,j],M[i-1,j-1],M[i,j-1],M[i+1,j-1],
   					M1[i-1,j+1],M1[i,j+1],M1[i+1,j+1],M1[i-1,j],M1[i,j],M1[i+1,j],M1[i-1,j-1],M1[i,j-1],M1[i+1,j-1],
   					M2[i-1,j+1],M2[i,j+1],M2[i+1,j+1],M2[i-1,j],M2[i,j],M2[i+1,j],M2[i-1,j-1],M2[i,j-1],M2[i+1,j-1])
 			if M[i,j] > max(compare) or M[i,j] < min(compare):
    				# note the pixel locations if the pixel of interest is an extrema
    				x_ex_oc1.append(i) # note x vertex
    				y_ex_oc1.append(j) # note y vertex
    				IDoG.append(M[i,j]) # note the pixel value 
   	return x_ex_oc1,y_ex_oc1,IDoG
    #---------------------Find the Extrema----------------------end----------#
    #Step1 complete....
    #--------------------- Elimination of extremas --> Keypoints ----------------------Start----------#
    def find_key (DOG2_extx_1,DOG2_exty_1,ID2_1,DOG2_1): # the function discards the extremas that are not required....
        for i in range(len(DOG2_extx_1)): # discarding the low contrast extremas    
            nx = DOG2_extx_1[i] # x index of extrema...
            ny = DOG2_exty_1[i] # y index of extrema...
            '''----- To remove the low contrast features-----------'''
            if abs(DOG2_1[nx+1,ny+1]) <= 0:    # 0 here is the contrast threshold below which we eliminate the pixels...
                ID2_1[i]=0 #eliminating the extrema....
            else:
                '''----- To remove the edges using tailor's expansion series-----------'''
                rx,ry = nx+1,ny+1 # rx,ry are the indecies of the neighboring pixels of extremas....
                if rx+1<DOG2_1.shape[0] and ry+1<DOG2_1.shape[1]: # avoiding index error on the below operation
                    fxx= DOG2_1[rx-1,ry]+DOG2_1[rx+1,ry]-2*DOG2_1[rx,ry]# double derivate in x direction
                    fyy= DOG2_1[rx,ry-1]+DOG2_1[rx,ry+1]-2*DOG2_1[rx,ry]; # double derivate in y direction
                    fxy= DOG2_1[rx-1,ry-1]+DOG2_1[rx+1,ry+1]-DOG2_1[rx-1,ry+1]-DOG2_1[rx+1,ry-1]; # derivate inx and y direction
                trace=fxx+fyy    # trace(H) = Dxx+Dyy 
                deter=fxx*fyy-fxy*fxy # Determinant(H) = DxxDyy - Dxy^2
                r=trace*trace/deter # curvature 
                th= ((r+1)^2)/r # curvature threshold...
                # If the current keypoint id poorly localized, its rejected...
                if (deter < 0 or r > th): # if the pixel belongs to an edge....
                    ID2_1[i]=0 # discard that pixel...
        index = array(nonzero(ID2_1)) # updating the list of keypoints after elimination...
        ID_new,x_new,y_new = [],[],[]
        for i in range(size(index)):
            ID_new.append(ID2_1[index[0,i]])
            x_new.append(DOG2_extx_1[index[0,i]])
            y_new.append(DOG2_exty_1[index[0,i]])
        return x_new,y_new,ID_new
    #--------------------- Elimination of extremas --> Keypoints ----------------------End----------#
    #Step2 complete...
    #--------------------- Keypoints Orientation ----------------------Start----------#
    # the function finds the orientation for the given keypoints....
    def find_ori(DOG2_extx_1key,DOG2_exty_1key,ID2_1key,DOG2_1,s):
        #mag--> magnitude, theta --> orientation in degrees
        mag,theta,bins,indexes = zeros(shape = (10,10)),zeros(shape = (10,10)),[[0] * 36 for z in range(len(ID2_1key))],[]
        for i in range(len(ID2_1key)):
            Iblur = cv2.GaussianBlur(DOG2_1,(5,5),1.5) # blurring the difference of gaussian....
            nx = DOG2_extx_1key[i] # x index of extrema...
            ny = DOG2_exty_1key[i] # y index of extrema...
            x1,x2,y1,y2 = nx - 5,nx + 5,ny - 5,ny + 5
            l = -1
            for j in range(x1,x2): # creating a box of 10x10 around the keypoints....
                l+=1
                m = -1
                if j<2:                            #
                    j = 2                          #
                if j>len(Iblur[:,0])-2:            # Avoiding the Index error....
                    j= len(Iblur[:,0])-2           #
                for k in range(y1,y2):             #    
                    m+=1                           #
                    if k<2:                        #
                        k = 2                      #
                    if k>len(Iblur[0,:])-2:        #
                        k = len(Iblur[0,:])-2      #
                    mag[l,m] = sqrt((Iblur[j+1,k] - Iblur[j-1,k])**2 + (Iblur[j,k+1] - Iblur[j,k-1])**2) # Magnitude calculation
                    theta[l,m] = math.degrees(math.atan((Iblur[j,k+1] - Iblur[j,k-1])/(Iblur[j+1,k] - Iblur[j-1,k]))) # Orientation calculation
                    if (theta[l,m] < 0):
                        theta[l,m] += 360 
            bins[i] = histogram(theta,36,[0,360])[0].tolist() # Histogram of orientation
            indexes.append(max(bins[i]))
        return indexes
    #--------------------- Keypoints Orientation ----------------------End----------#
    #Step3 completed....
    #--------------------- Plotting Keypoints ----------------------Start----------#
    def drawsitf(x,y,theta,I):
        figure()
        ax = plt.gca()
        ax.imshow(I,cmap=cm.gray)
        for i in range(len(x)):
            if not(x[i]>160 and x[i]<450 and y[i]>625 and y[i]<800):
                if not(x[i]>0 and x[i]<450 and y[i]>0 and y[i]<115):    
                    ax.add_patch(mpl.patches.Rectangle((y[i], x[i]), 20, 20,theta[i],joinstyle='bevel', ec="red", fill=False,axes=0))
        plt.draw()
        show()
    #--------------------- Plotting Keypoints ----------------------End----------#
    #--------------------- Keypoints descriptor ----------------------Start----------#
    def key_desc (DOG2_extx_1key,DOG2_exty_1key,ID2_1key,DOG2_1):
        key_desc_vector = [] # For each keypoint, stores 16 vectors of length 8 each---> 128 values
        for i in range(len(ID2_1key)):
            Iblur = cv2.GaussianBlur(DOG2_1,(5,5),1.5) # blurring the difference of gaussian....
            nx = DOG2_extx_1key[i] # x index of keypoint...
            ny = DOG2_exty_1key[i] # x index of keypoint...
            area = [[] * 16 for z in range(16)] # taking an area of 16*16 around the keypoint...
            area[0] = [(nx-7,ny-7),(nx-6,ny-7),(nx-5,ny-7),(nx-4,ny-7),(nx-7,ny-6),(nx-6,ny-6),(nx-5,ny-6),(nx-4,ny-6),
            (nx-7,ny-5),(nx-6,ny-5),(nx-5,ny-5),(nx-4,ny-5),(nx-7,ny-4),(nx-6,ny-4),(nx-5,ny-4),(nx-4,ny-4)]
            # Divding the area into 16 4x4 smaller areas....
            area[1] = [(nx-3,ny-7),(nx-2,ny-7),(nx-1,ny-7),(nx,ny-7),(nx-3,ny-6),(nx-2,ny-6),(nx-1,ny-6),(nx,ny-6),
            (nx-3,ny-5),(nx-2,ny-5),(nx-1,ny-5),(nx,ny-5),(nx-3,ny-4),(nx-2,ny-4),(nx-1,ny-4),(nx,ny-4)]
            area[2] = [(nx+3,ny-7),(nx+2,ny-7),(nx+1,ny-7),(nx,ny-7),(nx+3,ny-6),(nx+2,ny-6),(nx+1,ny-6),(nx,ny-6),
            (nx+3,ny-5),(nx+2,ny-5),(nx+1,ny-5),(nx,ny-5),(nx+3,ny-4),(nx+2,ny-4),(nx+1,ny-4),(nx,ny-4)]
            area[3] = [(nx+4,ny-7),(nx+5,ny-7),(nx+6,ny-7),(nx+7,ny-7),(nx+4,ny-6),(nx+5,ny-6),(nx+6,ny-6),(nx+7,ny-6),
            (nx+4,ny-5),(nx+5,ny-5),(nx+6,ny-5),(nx+7,ny-5),(nx+4,ny-4),(nx+5,ny-4),(nx+6,ny-4),(nx+7,ny-4)]
            area[4] = [(nx-7,ny-3),(nx-6,ny-3),(nx-5,ny-3),(nx-4,ny-3),(nx-7,ny-2),(nx-6,ny-2),(nx-5,ny-2),(nx-4,ny-2),
            (nx-7,ny-1),(nx-6,ny-1),(nx-5,ny-1),(nx-4,ny-1),(nx-7,ny),(nx-6,ny),(nx-5,ny),(nx-4,ny)]
            area[5] = [(nx-3,ny-3),(nx-2,ny-3),(nx-1,ny-3),(nx,ny-3),(nx-3,ny-2),(nx-2,ny-2),(nx-1,ny-2),(nx,ny-2),
            (nx-3,ny-1),(nx-2,ny-1),(nx-1,ny-1),(nx,ny-1),(nx-3,ny),(nx-2,ny),(nx-1,ny),(nx,ny)]
            area[6] = [(nx+3,ny-3),(nx+2,ny-3),(nx+1,ny-3),(nx,ny-3),(nx+3,ny-2),(nx+2,ny-2),(nx+1,ny-2),(nx,ny-2),
            (nx+3,ny-1),(nx+2,ny-1),(nx+1,ny-1),(nx,ny-1),(nx+3,ny),(nx+2,ny),(nx+1,ny),(nx,ny)]
            area[7] = [(nx+4,ny-3),(nx+5,ny-3),(nx+6,ny-3),(nx+7,ny-3),(nx+4,ny-2),(nx+5,ny-2),(nx+6,ny-2),(nx+7,ny-2),
            (nx+4,ny-1),(nx+5,ny-1),(nx+6,ny-1),(nx+7,ny-1),(nx+4,ny),(nx+5,ny),(nx+6,ny),(nx+7,ny)]
            area[8] = [(nx-7,ny),(nx-6,ny),(nx-5,ny),(nx-4,ny),(nx-7,ny+1),(nx-6,ny+1),(nx-5,ny+1),(nx-4,ny+1),
            (nx-7,ny+2),(nx-6,ny+2),(nx-5,ny+2),(nx-4,ny+2),(nx-7,ny+3),(nx-6,ny+3),(nx-5,ny+3),(nx-4,ny+3)]
            area[9] = [(nx-3,ny),(nx-2,ny),(nx-1,ny),(nx,ny),(nx-3,ny+1),(nx-2,ny+1),(nx-1,ny+1),(nx,ny+1),
            (nx-3,ny+2),(nx-2,ny+2),(nx-1,ny+2),(nx,ny+2),(nx-3,ny+3),(nx-2,ny+3),(nx-1,ny+3),(nx,ny+3)]       
            area[10] = [(nx+3,ny),(nx+2,ny),(nx+1,ny),(nx,ny),(nx+3,ny+1),(nx+2,ny+1),(nx+1,ny+1),(nx,ny+1),
            (nx+3,ny+2),(nx+2,ny+2),(nx+1,ny+2),(nx,ny+2),(nx+3,ny+3),(nx+2,ny+3),(nx+1,ny+3),(nx,ny+3)]
            area[11] = [(nx+4,ny),(nx+5,ny),(nx+6,ny),(nx+7,ny),(nx+4,ny+1),(nx+5,ny+1),(nx+6,ny+1),(nx+7,ny+1),
            (nx+4,ny+2),(nx+5,ny+2),(nx+6,ny+2),(nx+7,ny+2),(nx+4,ny+3),(nx+5,ny+3),(nx+6,ny+3),(nx+7,ny+3)]
            area[12] = [(nx-7,ny+4),(nx-6,ny+4),(nx-5,ny+4),(nx-4,ny+4),(nx-7,ny+5),(nx-6,ny+5),(nx-5,ny+5),(nx-4,ny+5),
            (nx-7,ny+6),(nx-6,ny+6),(nx-5,ny+6),(nx-4,ny+6),(nx-7,ny+7),(nx-6,ny+7),(nx-5,ny+7),(nx-4,ny+7)] 
            area[13] = [(nx-3,ny+4),(nx-2,ny+4),(nx-1,ny+4),(nx,ny+4),(nx-3,ny+5),(nx-2,ny+5),(nx-1,ny+5),(nx,ny+5),
            (nx-3,ny-1),(nx-2,ny-1),(nx-1,ny-1),(nx,ny-1),(nx-3,ny+7),(nx-2,ny+7),(nx-1,ny+7),(nx,ny+7)]
            area[14] = [(nx+3,ny+4),(nx+2,ny+4),(nx+1,ny+4),(nx,ny+4),(nx+3,ny+5),(nx+2,ny+5),(nx+1,ny+5),(nx,ny+5),
            (nx+3,ny+6),(nx+2,ny+6),(nx+1,ny+6),(nx,ny+6),(nx+3,ny+7),(nx+2,ny+7),(nx+1,ny+7),(nx,ny+7)]
            area[15] = [(nx+4,ny+4),(nx+5,ny+4),(nx+6,ny+4),(nx+7,ny+4),(nx+4,ny+5),(nx+5,ny+5),(nx+6,ny+5),(nx+7,ny+5),
            (nx+4,ny+6),(nx+5,ny+6),(nx+6,ny+6),(nx+7,ny+6),(nx+4,ny+7),(nx+5,ny+7),(nx+6,ny+7),(nx+7,ny+7)]
            #initialization....
            word,bins = [],[[0] * 8 for z in range(16)]
            for m in range(16):
                m = -1
                mag,theta = [0]*16,[0]*16 # initialization of magnitude and orientation...
                for j,k in area[z]:
                    if j<2:                       #
                        j = 2                      #
                    if j>len(Iblur[:,0])-2:       # Avoiding the Index error....
                        j= len(Iblur[:,0])-2       #
                    if k<2:                       #
                        k = 2                      #
                    if k>len(Iblur[0,:])-2:       #
                        k = len(Iblur[0,:])-2      #
                    m+=1
                    mag[m] = sqrt((Iblur[j+1,k] - Iblur[j-1,k])**2 + (Iblur[j,k+1] - Iblur[j,k-1])**2) # Magnitude calculation
                    theta[m] = round(math.degrees(math.atan((Iblur[j,k+1] - Iblur[j,k-1])/(Iblur[j+1,k] - Iblur[j-1,k])))) # Orientation calculation
                    if (theta[m] < 0):
                        theta[m] += 360 
                bins[m] = histogram(theta,8,[0,360])[0].tolist() # Histogram of orientation with 8 bins
            normbin = (bins/linalg.norm(bins)) # normalizing after collecting all 16 histograms for a keypoint...
            a = scipy.stats.threshold(normbin,0,0.2,0.2) # the threshold used here is 0.2 although this value could be changed as per requirement....
            bin_final = around(a, decimals=2) # rounding to 2 decimal points for ease
            key_desc_vector.append([bin_final]) # key descriptions of all keypoints...
        return key_desc_vector
    #--------------------- Keypoints descriptor ----------------------End----------#
    #Step4 completed....
    '''-----------------------Function Definitions Complete------------------------'''
    #%%%%%%%%%%%%%%%%%%%%Step1 --> Scale-space extrema detection%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #---------------------First Octave----------------start--------#
    (DOG1_1,DOG2_1,DOG3_1,DOG4_1) = find_octave(I,s,k) # DoGs for OCtave 1 #DOG1_1 --> DOG1 of OCTAVE1
    (DOG2_extx_1,DOG2_exty_1,ID2_1) = find_ext(DOG3_1,DOG2_1,DOG1_1) # extrema of DoG2 for OCtave 1
    (DOG3_extx_1,DOG3_exty_1,ID3_1) = find_ext(DOG4_1,DOG3_1,DOG2_1) # extrema of DoG3 for OCtave 1
    #---------------------First Octave----------------END--------#
    I1 = cv2.resize(I,(row/2,clm/2)) # Down sample the input image
    #---------------------Second Octave----------------start--------#
    (DOG1_2,DOG2_2,DOG3_2,DOG4_2) = find_octave(I1,s,k) # DoGs for OCtave 2
    (DOG2_extx_2,DOG2_exty_2,ID2_2) = find_ext(DOG3_2,DOG2_2,DOG1_2) # extrema of DoG2 for OCtave 2
    (DOG3_extx_2,DOG3_exty_2,ID3_2) = find_ext(DOG4_2,DOG3_2,DOG2_2) # extrema of DoG3 for OCtave 2
    #---------------------Second Octave----------------END--------#
    I2 = cv2.resize(I,(row/4,clm/4)) # Down sample the input image again
    #---------------------third Octave----------------start--------#
    (DOG1_3,DOG2_3,DOG3_3,DOG4_3) = find_octave(I2,s,k) # DoGs for OCtave 3
    (DOG2_extx_3,DOG2_exty_3,ID2_3) = find_ext(DOG3_3,DOG2_3,DOG1_3) # extrema of DoG2 for OCtave 3
    (DOG3_extx_3,DOG3_exty_3,ID3_3) = find_ext(DOG4_3,DOG3_3,DOG2_3) # extrema of DoG3 for OCtave 3
    #---------------------third Octave----------------END--------#
    I3 = cv2.resize(I,(row/8,clm/8)) # Down sample the input image again
    #---------------------fourth Octave----------------start--------#
    (DOG1_4,DOG2_4,DOG3_4,DOG4_4) = find_octave(I3,s,k) # DoGs for OCtave 4
    (DOG2_extx_4,DOG2_exty_4,ID2_4) = find_ext(DOG3_4,DOG2_4,DOG1_4) # extrema of DoG2 for OCtave 4
    (DOG3_extx_4,DOG3_exty_4,ID3_4) = find_ext(DOG4_4,DOG3_4,DOG2_4) # extrema of DoG3 for OCtave 4
    #---------------------fourth Octave----------------END--------#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Step2 --> Keypoint localization%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #-------------------------Accurate Keypoints Localization---------------------------Start---------#
    (DOG2_extx_1key,DOG2_exty_1key,ID2_1key) = find_key(DOG2_extx_1,DOG2_exty_1,ID2_1,DOG2_1) # keypoints of DoG2 for OCtave 1
    (DOG3_extx_1key,DOG3_exty_1key,ID3_1key) = find_key(DOG3_extx_1,DOG3_exty_1,ID3_1,DOG3_1) # keypoints of DoG3 for OCtave 1
    (DOG2_extx_2key,DOG2_exty_2key,ID2_2key) = find_key(DOG2_extx_2,DOG2_exty_2,ID2_2,DOG2_2) # keypoints of DoG2 for OCtave 2
    (DOG3_extx_2key,DOG3_exty_2key,ID3_2key) = find_key(DOG3_extx_2,DOG3_exty_2,ID3_2,DOG3_2) # keypoints of DoG3 for OCtave 2
    (DOG2_extx_3key,DOG2_exty_3key,ID2_3key) = find_key(DOG2_extx_3,DOG2_exty_3,ID2_3,DOG2_3) # keypoints of DoG2 for OCtave 3
    (DOG3_extx_3key,DOG3_exty_3key,ID3_3key) = find_key(DOG3_extx_3,DOG3_exty_3,ID3_3,DOG3_3) # keypoints of DoG3 for OCtave 3
    (DOG2_extx_4key,DOG2_exty_4key,ID2_4key) = find_key(DOG2_extx_4,DOG2_exty_4,ID2_4,DOG2_4) # keypoints of DoG2 for OCtave 4
    (DOG3_extx_4key,DOG3_exty_4key,ID3_4key) = find_key(DOG3_extx_4,DOG3_exty_4,ID3_4,DOG3_4) # keypoints of DoG3 for OCtave 4
    #-------------------------Accurate Keypoints Localization---------------------------End---------#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''Step3 --> Orientation assignment'''#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #-------------------------Accurate Keypoints Localization---------------------------Start---------#
    theta2_1 = find_ori(DOG2_extx_1,DOG2_exty_1,ID2_1,DOG2_1,s) # octave1
    theta3_1 = find_ori(DOG2_extx_1,DOG2_exty_1,ID3_1,DOG3_1,s)# octave1
    theta2_2 = find_ori(DOG2_extx_2,DOG2_exty_2,ID2_2,DOG2_2,s)# octave2
    theta3_2 = find_ori(DOG2_extx_2,DOG2_exty_2,ID3_2,DOG3_2,s)# octave2
    theta2_3 = find_ori(DOG2_extx_3,DOG2_exty_3,ID2_3,DOG2_3,s)# octave3
    theta3_3 = find_ori(DOG2_extx_3,DOG2_exty_3,ID3_3,DOG3_3,s)# octave3
    theta2_4 = find_ori(DOG2_extx_4,DOG2_exty_4,ID2_4,DOG2_4,s)# octave4
    theta3_4 = find_ori(DOG2_extx_4,DOG2_exty_4,ID3_4,DOG3_4,s)# octave4
    #-------------------------Accurate Keypoints Localization---------------------------End---------#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''Step4 --> Feature Description'''#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #-------------------------Keypoints Orientation---------------------------Start---------#
    # since we dont used the key descriptors any further... i have called the below function only for a few keypoints in order to minimize
    #the computation time, we can uncomment the below lines in order to get the key descriptors for all the key points... 
    key_2_1=key_desc(DOG2_extx_1,DOG2_exty_1,ID2_1,DOG2_1)
    #key_3_1=key_desc(DOG3_extx_1,DOG3_exty_1,ID3_1,DOG3_1)
    #key_2_2=key_desc(DOG2_extx_2,DOG2_exty_2,ID2_2,DOG2_2)
    #key_3_2=key_desc(DOG3_extx_2,DOG3_exty_2,ID3_2,DOG3_2)
    #key_2_3=key_desc(DOG2_extx_3,DOG2_exty_3,ID2_3,DOG2_3)
    #key_3_3=key_desc(DOG3_extx_3,DOG3_exty_3,ID3_3,DOG3_3)
    #-------------------------Keypoints Orientation---------------------------End---------#
    x = DOG2_extx_1+DOG3_extx_1+DOG2_extx_2+DOG3_extx_2+DOG2_extx_3+DOG3_extx_3+DOG2_extx_4+DOG3_extx_4
    y = DOG2_exty_1+DOG3_exty_1+DOG2_exty_2+DOG3_exty_2+DOG2_exty_3+DOG3_exty_3+DOG2_exty_4+DOG3_exty_4
    theta = theta2_1+theta3_1+theta2_2+theta3_2+theta2_3+theta3_3+theta2_4+theta3_4
    drawsitf(x,y,theta,I)
#=============================== Call Function ================================#
inp = 'SIFT-input1.png'
I = array(Image.open(inp).convert('L')) # read the input image1
sift_imp(I)
inp = 'SIFT-input2.png'
I = array(Image.open(inp).convert('L')) # read the input image2
sift_imp(I)
#=============================== END ================================#
'''
            OOOOOOOOOOO
         OOOOOOOOOOOOOOOOOOO
      OOOOOO  OOOOOOOOO  OOOOOO
    OOOOOO      OOOOO      OOOOOO
  OOOOOOOO  @   OOOOO  @   OOOOOOOO
 OOOOOOOOOO    OOOOOOO    OOOOOOOOOO
OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
OOOO  OOOOOOOOOOOOOOOOOOOOOOOOO  OOOO
 OOOO  OOOOOOOOOOOOOOOOOOOOOOO  OOOO
  OOOO   OOOOOOOOOOOOOOOOOOOO  OOOO
    OOOOO   OOOOOOOOOOOOOOO   OOOO
      OOOOOO   OOOOOOOOO   OOOOOO
         OOOOOO         OOOOOO
             OOOOOOOOOOOO
'''