
# coding: utf-8

# Camera Calibration with OpenCV
# ===
# Run All Cells should work. The final Main cell invokes the calibration and runs undistort on the cal input files and displays and saves them to file.
# 

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os
import pickle


# In[2]:


get_ipython().run_cell_magic('HTML', '', '<style> code {background-color : orange !important;} </style>\nfrom IPython.core.display import display, HTML\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# In[3]:


# For debug only. These are default to false at start of module, but overriden in the final Main testing cell
g_doDrawCorners = False # This is used in cal calculation: Overlays the calImage with the found corners
g_doCreateBothImage = False # Creates composite of input and output images in CameraCal_Undistort()


# In[4]:


def OpenImages(imgFileNames):
    imgList = []
    for imgFileName in imgFileNames:
        img = mpimg.imread(imgFileName)
        imgList.append(img)
    return(imgList)

def SaveImages(imgFileNames, images):
    for imgFileName, img in zip(imgFileNames, images):
        mpimg.imsave(imgFileName, img, format='jpg')


# In[5]:


def DrawText(img, text, posLL = (10,40)):
    font                   = cv2.FONT_HERSHEY_PLAIN    
    fontScale              = 2
    fontColor              = (255, 0, 0) # cv.BGR Blue but saved np.RGB red
    lineType               = 2
    cv2.putText(img, text, posLL, font, fontScale, fontColor, lineType)
 


# In[6]:


get_ipython().run_line_magic('matplotlib', 'qt4')
def AnimateImages(imgList, pauseMs=1000):
    for index, imgIn in enumerate(imgList):
        windowName = "img" # 'img'+str(index), imgIn) # Use same window name to reuse same window. Different names for different windows
        cv2.imshow(windowName, imgIn)
        cv2.waitKey(pauseMs)

    cv2.destroyAllWindows()


# In[7]:


def CameraCal_FindCorners(calImages, numCornersXY):
    """
    Calculates the chessboard trueCorners and foundCorners of a list of chessboard cal images.
    """
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    numCornersTotal = numCornersXY[0] * numCornersXY[1]
    trueCornerPointsSingle = np.zeros((numCornersTotal, 3), np.float32)
    trueCornerPointsSingle[:,:2] = np.mgrid[0:numCornersXY[0], 0:numCornersXY[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    trueCornerPoints = [] # 3d points in real world space
    foundCornerPoints = [] # 2d points in image plane.

    cornersFoundStatus = []

    # Step through the list and search for chessboard corners
    for index, imgCalIn in enumerate(calImages):
        imgGray = cv2.cvtColor(imgCalIn, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        cornersFound, curImgCornerPoints = cv2.findChessboardCorners(imgGray, numCornersXY, None)
        cornersFoundStatus.append(cornersFound)
        
        print("calImageIndex={} cornersFound={}".format(index+1, cornersFound))
        # If found, append the true and found corners to retval lists
        if cornersFound == True:
            trueCornerPoints.append(trueCornerPointsSingle)
            foundCornerPoints.append(curImgCornerPoints)
            
            # For dev/debug draw the corners on the input image
            if (g_doDrawCorners):
                cv2.drawChessboardCorners(imgCalIn, numCornersXY, curImgCornerPoints, True) 

        # For dev/debug draw failure notice input image        
        elif (g_doDrawCorners):
            DrawText(imgCalIn, 'CornersNotFound!')            
        
    return(cornersFoundStatus, trueCornerPoints, foundCornerPoints)


# In[8]:


def CameraCal_CalcCalValsFromPoints(trueCornerPoints, foundCornerPoints, imgSample):
    """
    Do camera calibration given object points and image points
    """
    
    imgSize = (imgSample.shape[1], imgSample.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(trueCornerPoints, foundCornerPoints, imgSize, None, None)
    dictCameraCalVals = {}
    dictCameraCalVals["mtx"] = mtx
    dictCameraCalVals["dist"] = dist
    dictCameraCalVals["rvecs"] = rvecs
    dictCameraCalVals["tvecs"] = tvecs
    return(dictCameraCalVals)

def CameraCal_CalcCalValsFromImages(calImages, numCornersXY):
    """
    Do camera calibration given a list of numpy checkerboard images
    """
    
    cornersFoundStatus, trueCornerPoints, foundCornerPoints = CameraCal_FindCorners(calImages, numCornersXY)
    dictCameraCalVals = CameraCal_CalcCalValsFromPoints(trueCornerPoints, foundCornerPoints, calImages[0])
    return(dictCameraCalVals)
 
def CameraCal_SaveCalFile(calFileName, dictCameraCalVals):
    pickle.dump(dictCameraCalVals, open(calFileName, "wb" ) )

def CameraCal_LoadCalFile(calFileName):
    dictCameraCalVals = pickle.load( open(calFileName, "rb" ) )
    return(dictCameraCalVals)
    


# In[9]:


def CameraCal_Undistort(dictCameraCalVals, imgIn):
    """
    Perform image undistortion on a numpy image
    """
    imgOut = cv2.undistort(imgIn, dictCameraCalVals["mtx"], dictCameraCalVals["dist"], None, dictCameraCalVals["mtx"])
    return(imgOut)

def CameraCal_UndistortList(dictCameraCalVals, imagesIn):
    """
    Perform image undistortion on a list of numpy images
    """
    imagesOut = []
    imagesBothOut = []
    for index, imgIn in enumerate(imagesIn):
        imgOut = CameraCal_Undistort(dictCameraCalVals, imgIn)
        imagesOut.append(imgOut)        
        if (g_doCreateBothImage):
            imgBoth = np.concatenate((imgIn, imgOut), axis=1)
            imagesBothOut.append(imgBoth)        
    return(imagesBothOut, imagesOut)


# ## Main routine
# ### Creates a camera distortion cal object from a list of checkerboard cal files and saves the cal object to disk
# Or optionally loads the cal object from disk.
# Then applies the cal object to test images and displays (and optionally saves) the resulting images

# In[10]:


# For debug only. These are default to false at start of module
g_doDrawCorners=True # This is used in cal calculation: Overlays the calImage with the found corners
g_doCreateBothImage = True # Creates composite of input and output images in CameraCal_Undistort()

g_outputDir = "camera_caloutput/"
g_calFileName = g_outputDir + "cameracalval.pypickle"
if (not os.path.exists(g_outputDir)):
    os.makedirs(g_outputDir)
    
g_numCornersXY = (9,6)

g_calFileInNames = glob.glob('camera_cal/cal*.jpg')
#g_calFileInNames = ['camera_cal/calibration2.jpg']

# Get natural sort of files... only works for this exact environment and is useful only for dev debug
g_doSort=True
if (g_doSort):
    g_baseLen = len("camera_cal/calibration")
    g_calFileInNames = sorted(g_calFileInNames, key=lambda fileName: int(fileName[g_baseLen:-4]))
    
#print("Infiles: ", g_calFileInNames)
g_calImages = OpenImages(g_calFileInNames)

g_useCalFile = False
if (g_useCalFile):
    g_dictCameraCalVals = CameraCal_LoadCalFile(g_calFileName)
else:
    g_dictCameraCalVals = CameraCal_CalcCalValsFromImages(g_calImages, g_numCornersXY)
    CameraCal_SaveCalFile(g_calFileName, g_dictCameraCalVals)
    #AnimateImages(g_calImages, 2000)

# Run undistort test on the same images used for calibration
g_testImagesIn = g_calImages
g_testImagesBothOut, g_testImagesOut = CameraCal_UndistortList(g_dictCameraCalVals, g_testImagesIn)  

# Animate the combined test output to see side by side comparison
AnimateImages(g_testImagesBothOut, 500)

g_saveOutFiles = True
if (g_saveOutFiles):
    g_outputFileNames = [(g_outputDir + os.path.splitext(os.path.basename(fileName))[0] + ".out.jpg") for fileName in g_calFileInNames]
    #print("Outfiles: ", g_outputFileNames)
    SaveImages(g_outputFileNames, g_testImagesBothOut)


# ## Sample invocation for use in image proc pipeline

# In[11]:


def SampleInvocation():
    imgFileName = 'camera_cal/calibration2.jpg'
    imgIn = mpimg.imread(imgFileName)

    dictCameraCalVals = CameraCal_LoadCalFile(g_calFileName)
    imgUndistorted = CameraCal_Undistort(dictCameraCalVals, imgIn)
    
    get_ipython().run_line_magic('matplotlib', 'inline')
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(imgIn)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(imgUndistorted)
    ax2.set_title('Undistorted Image', fontsize=30)
    
SampleInvocation()

