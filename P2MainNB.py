
# coding: utf-8

# # **Proj2: Advanced Lane Finding** 
# ### ChrisL
# 

# In[1]:


import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import math
import os
import pickle

from moviepy.editor import VideoFileClip
from IPython.display import HTML
import datetime


# In[2]:


get_ipython().run_cell_magic('HTML', '', '<style> code {background-color : orange !important;} </style>\nfrom IPython.core.display import display, HTML\njnk = display(HTML("<style>.container { width:100% !important; }</style>"))')


# ## README
# #### Check the cell below for global variable switches you may want to manipulate.
# They are currently set so that if you do run all cells the notebook will 
# process project_video.mp4 (stored in a subfolder) and create project_video_out in the root.
# 

# In[3]:


########################################### Globals #################

# I most often used qt4 backend, but I've selected the safest option for a clean run-all
#%matplotlib qt4
#%matplotlib inline
get_ipython().run_line_magic('matplotlib', 'notebook')

# Determines if run-all-cells should also run the entire movie process
g_doAutorunMovieProcess = True

# DEVDEBUG IMAGE DISPLAY
# This slows down the movie process alot, and doesn't work well for multiframe.
# If you want to see all the pipeline images enable this and run the P2Main cell
g_doDisplayDebugImages = False
# If g_doDisplayDebugImages==True and this is true 
# the pipeline images will also be save to file.
# This was very useful for dev/debug but not so much for code review.
g_doSaveDebugImages = False


# Specify filenames that contain the camera calibration information
# These should stay as is
g_cameraDistortionCalValsFileName = "CameraDistortionCalVals.pypickle"
g_cameraPerspectiveWarpMatrixFileName = "CameraPerspectiveWarpMatrix.pypickle"


# ### Display/Plotting Utilities

# In[4]:


#-------------------------------------------------------------------
#g_plotFigIndex = 0
def PlotImageRecords(imgRecords, doSaveDebugImages):
    #global g_plotFigIndex
    fig = plt.figure(0)#g_plotFigIndex)
    g_plotFigIndex+=1
    fig.set_size_inches(18,12)

    numImages = len(imgRecords)
    numCols = 3
    numRows = math.ceil(numImages/numCols)
    for recIndex, imgRecord in enumerate(imgRecords):
        numFields = len(imgRecord)
        img = imgRecord[0]
        if (numFields >= 2):
            imgName =  imgRecord[1]
        else:
            imgName =  "img_" + str(recIndex)
            
        if (numFields >= 3):
            kwArgs =  imgRecord[2]
        else:
            kwArgs =  {}
                
        plt.subplot(numRows, numCols, recIndex+1)
        plt.title(imgName)
        plt.imshow(img, **kwArgs)
       
    plt.show()
    fig.tight_layout()
    
    if doSaveDebugImages:
        firstPlotName = imgRecords[0][1]
        PlotSaveFigure(fig, firstPlotName)
        
 #-------------------------------------------------------------------
def PlotSaveFigure(fig, plotName):
    dirOut = "ImagesOut/ImagesOut/PipeLineFigures/"
    if (not os.path.exists(dirOut)):
        os.makedirs(dirOut)

    pfigsDT = "pfig_{:%Y-%m-%dT%H:%M:%S}_".format(datetime.datetime.now())

    baseext =  os.path.basename(plotName)
    fileNameBase =  os.path.splitext(baseext)[0]
    fileNameOut = dirOut + pfigsDT  + fileNameBase + ".png"
    print("    Saving fig:", fileNameOut)
    fig.savefig(fileNameOut, bbox_inches='tight')    


# In[5]:


def DrawText(img, text, posLL = (10,40), colorFont=(255, 255, 255), fontScale = 2):
    font                   = cv2.FONT_HERSHEY_PLAIN    
    lineType               = 2
    cv2.putText(img, text, posLL, font, fontScale, colorFont, lineType)


# In[6]:


def PlotSelectedAndPolynomial(polyCoeff, inPixelsX, inPixelsY, imgInBW):    
    imageHeight = imgInBW.shape[0]
    imageWidth = imgInBW.shape[1]

    # Set unselected pixels to grey
    imgInGrey = imgInBW * 64
    imgOut = np.dstack((imgInGrey, imgInGrey, imgInGrey))
    
    # Display the selected pixels
    imgOut[inPixelsY, inPixelsX, 1] = 255 # Set selected pixels to red
    
    lineSegmentPoints = EvalPolyToLineSegments(polyCoeff, imageHeight, doReverseSegments=False)
    OverlayLineSegments(imgOut, lineSegmentPoints, isClosed=False)
    return imgOut

#g_imgPlotOut = PlotSelectedAndPolynomial(g_testpolyCoeff, g_inPixelsX, g_inPixelsY, g_testimgPlotIn)
#%matplotlib qt4
#plt.imshow(g_imgPlotOut)
#plt.imshow(g_testimgPlotIn, cmap="gray")


# ### Camera Calibration Utilities and Processing

# In[7]:


def CameraCal_GetCalVals(cameraDistortionCalValsFileName, cameraPerspectiveWarpMatrixFileName):
    dictCameraCalVals = pickle.load( open(cameraDistortionCalValsFileName, "rb" ) )
    dictCameraCalVals["warpMatrix"] = pickle.load( open(cameraPerspectiveWarpMatrixFileName, "rb" ) )

    # These scaling values were provided in the modules.
    # Obviously they should be a variable value determined 
    # by some perCamera calibration process with respect to the warp matrix calibration
    # Also, road slope curvature would be another factor in real solution
    dictCameraCalVals["metersPerPixX"] = 3.7/700 # 0.005285714285714286 meters per pixel in x dimension 189.189 p/m
    dictCameraCalVals["metersPerPixY"] = 30/720 # 0.041666666666666664 meters per pixel in y dimension 24 p/m

    return(dictCameraCalVals)

def CameraCal_Undistort(imgIn, dictCameraCalVals):
    """
    Perform image undistortion on a numpy image
    """
    imgOut = cv2.undistort(imgIn, dictCameraCalVals["mtx"], dictCameraCalVals["dist"], None, dictCameraCalVals["mtx"])
    return(imgOut)

def CameraCal_DoPerspectiveTransform(imgIn, matTransform, doInverse = False):
    img_size = (imgIn.shape[1], imgIn.shape[0])
    flags = cv2.INTER_CUBIC
    if doInverse:
        flags |= cv2.WARP_INVERSE_MAP    
    imgOut = cv2.warpPerspective(imgIn, matTransform, img_size, flags=flags)
    return imgOut


# ### Formula for Radius for curvature
# See (http://www.intmath.com/applications-differentiation/8-radius-curvature.php)
# 
# $$
# \Large
# \begin{align}
# R_{curve} &=  \frac{(1 + (2Ay+B)^2 )^{3/2}}{|2A|}
# \end{align}
# $$

# In[8]:


# See http://www.intmath.com/applications-differentiation/8-radius-curvature.php
def CalcRadiusOfCurvatureParabola(funcCoeffients, evalAtpixY, metersPerPixX=1, metersPerPixY=1):
   # Convert parabola from pixels to meters
   coA = funcCoeffients[0] *  metersPerPixX/(metersPerPixY**2)
   coB = funcCoeffients[1] * metersPerPixX/metersPerPixY
   #coC = funcCoeffients[2] * metersPerPixX # Not used

   mY = evalAtpixY * metersPerPixY

   radiusNumerator = pow((1 + (2 * coA * mY + coB)**2), 3/2)
   radiusDenominator = abs(2*coA)
   radiusMeters = round(radiusNumerator/radiusDenominator, 1)                           
   return(radiusMeters)

def TestCalcRadiusOfCurvatureParabola():
   testPolyCoeffLeft = np.array([  3.07683280e-04 , -4.44713275e-01 ,  3.59067572e+02])
   testPolyCoeffRight = np.array([  2.53380891e-04,  -3.96103079e-01  , 1.04908806e+03])
   #testPolyCoeffLeft = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
   #testPolyCoeffRight = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])
   metersPerPixX = 3.7/700 # 0.005286 meters per pixel in x dimension 189.2 pix/m
   metersPerPixY = 30/720  # 0.041667 meters per pixel in y dimension 24.0 pix/m

   evalAtpixY = 600
   curveX = CalcRadiusOfCurvatureParabola(testPolyCoeffLeft, evalAtpixY, metersPerPixX, metersPerPixY)
   curveY = CalcRadiusOfCurvatureParabola(testPolyCoeffRight, evalAtpixY, metersPerPixX, metersPerPixY)
   
   print("curveX m", curveX) # expect 533.8m 1625.1pix
   print("curveY m", curveY) # expect 648.2m 1976.3pix
   
TestCalcRadiusOfCurvatureParabola ()                               


# ### Image Preprocessing Pipeline Functions 
# 

# In[9]:


def SelectPixelsUsingSlidingWindow(imgIn, whichLine):    
    margin = 100 # Set the width of the windows +/- margin    
    minpix = 100 # Set minimum number of pixels found to recenter window    
    numWindows = 9 # Number of sliding windows

    # Various image display variables
    lineWidth = 3
    colorWinRect = (0,255,0)
    colorNextCenterXLine = (0, 130, 255)
    
    imageHeight = imgIn.shape[0]
    imageWidth = imgIn.shape[1]
    
    # Create winInfo struct for bookkeeping. Reused each loop
    win = type('WindowInfo', (object,), {})  
    win.height = np.int(imageHeight//numWindows)
    win.width = margin * 2
   
    # Choose which side of the image to evaluate the histogram 
    # depending on which lane line we are looking for
    if (whichLine == "LEFT"):
        histLeft = 0
        histRight = imageWidth//2
    else:
        histLeft = imageWidth//2
        histRight = imageWidth

    # Take a histogram of the bottom half of the image
    # and find the peak. This will be the initial window center
    histogram = np.sum(imgIn[imageHeight//2:, : ], axis=0)
    win.centerX = histLeft + np.argmax(histogram[histLeft:histRight])
    
    # Identify the x and y positions of all nonzero pixels in the image    
    pixelsNon0 = imgIn.nonzero()
    pixelsNon0X = np.array(pixelsNon0[1])
    pixelsNon0Y = np.array(pixelsNon0[0])
 
    # Create empty lists to receive left and right lane pixel indices
    selectedPixelIndicesAccum = []

    # Create an output image to draw on for visualizing the result
    imgSelectionOut = np.dstack((imgIn, imgIn, imgIn))

    # Loop thru windows
    for windowIndex in range(numWindows):
        # Identify window boundaries 
        win.bottom = imageHeight - windowIndex * win.height
        win.top = win.bottom - win.height
        win.left = win.centerX -  margin
        win.right = win.centerX +  margin

        # Draw the windows on the visualization image
        cv2.rectangle(imgSelectionOut, (win.left, win.bottom), (win.right, win.top), colorWinRect, lineWidth)


        # Identify the nonzero pixels in x and y within the window 
        selectedPixelIndicesCur =  ((pixelsNon0X >= win.left) & (pixelsNon0X < win.right) 
                                   &(pixelsNon0Y >= win.top) & (pixelsNon0Y < win.bottom)).nonzero()[0]

        # Append these indices to the accumulated selected pixels list
        selectedPixelIndicesAccum.append(selectedPixelIndicesCur)

        # Get sub image corresponding to the cur window
        imgWindow = imgIn[win.top : win.bottom, win.left : win.right]
        pixelsWindow0 = imgWindow.nonzero()
        pixelsWindow0X = pixelsWindow0[1]

        # If there are enough nonzero pixels for a meaningful center shift
        if (len(pixelsWindow0X) > minpix):
            # Find the X center of all the nonzero pixels in this window, use that for next window center
            pixelsWindow0XAvg = int(np.mean(pixelsWindow0X))
        else:
            # Otherwise it may be a gap in the lane line, just keep the same center for next window
            pixelsWindow0XAvg = margin

        nextWindowCenterX = pixelsWindow0XAvg + win.left
        win.centerX = nextWindowCenterX

        # Draw a line thru this window indicating the found centerX that will be used for the next window
        pt0, pt1 = (nextWindowCenterX, win.bottom), (nextWindowCenterX, win.top)
        cv2.line(imgSelectionOut, pt0, pt1, colorNextCenterXLine, lineWidth)

 
    # Get the pixel indices from all of the windows
    selectedPixelIndices = np.concatenate(selectedPixelIndicesAccum)
    # Get the non0 pixels for those indices
    selectedPixelsX = pixelsNon0X[selectedPixelIndices]
    selectedPixelsY = pixelsNon0Y[selectedPixelIndices]

    return selectedPixelsX, selectedPixelsY, imgSelectionOut

def Test_SelectPixelsUsingSlidingWindow():
    selectedPixelsX, selectedPixelsY, imgSelectionOut = SelectPixelsUsingSlidingWindow(g_imgTest, "RIGHT")
    plt.figure(figsize=(12,8))
    imgSelectionOut[selectedPixelsY, selectedPixelsX] = [255, 0, 0]
    plt.imshow(imgSelectionOut)
    plt.tight_layout()

#=========== Test invocation
#g_testImgFileName = "ImagesIn/TestImagesIn/PipelineStages/warped_example.jpg"
#g_imgTest = mpimg.imread(g_testImgFileName)
#Test_SelectPixelsUsingSlidingWindow()


# In[10]:


def ImageProc_HSLThreshold(imgHLSIn, channelNum, threshMinMax=(0, 255)):
    imgOneCh = imgHLSIn[:,:,channelNum]
    imgBWMask = np.zeros_like(imgOneCh)
    imgBWMask[(imgOneCh > threshMinMax[0]) & (imgOneCh <= threshMinMax[1])] = 1
    return imgOneCh, imgBWMask


# In[11]:


#-------------------------------------------------------------------
def SobelThesholdMag(imgIn, sobel_kernel=3, threshMinMax=(0, 255)):
    """
    Calculates the Sobel XY magnitude value and applies a threshold.
    :param img: input image as np.array
    """    
    # 1) Convert to grayscale
    imgGray = cv2.cvtColor(imgIn,cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*gradmag/np.max(gradmag))
    #scale_factor = np.max(gradmag)/255 
    #scaled_sobel = (gradmag/scale_factor).astype(np.uint8) 
    
    # 5) Create a binary mask where mag thresholds are met
    imgBWMask = np.zeros_like(scaled_sobel)
    imgBWMask[(scaled_sobel >= threshMinMax[0]) & (scaled_sobel <= threshMinMax[1])] = 1
    return imgBWMask


# In[12]:


def EvalPolyToLineSegments(polyCoeff, imageHeight, doReverseSegments=False):
    coA, coB, coC = polyCoeff[0], polyCoeff[1], polyCoeff[2]
    
    #if doReverseSegments:
    #    yStart, yStop, yStep = imageHeight-4, 4, -8
    #else:
    #    yStart, yStop, yStep = 4, imageHeight-4, 8
    yStart, yStop, yStep = 8, imageHeight, 8
    polyInY = np.array([y for y in range(yStart, yStop, yStep) ]) # Start at 4 so top line segments are sure to render
    polyOutXf = coA * polyInY**2 + coB * polyInY + coC  
    polyOutX = polyOutXf.astype(int)

    lineSegmentPoints = np.array(list(zip(polyOutX, polyInY)))
    if doReverseSegments:
        lineSegmentPoints = lineSegmentPoints[::-1]

    return lineSegmentPoints

def OverlayLineSegments(imgIn, lineSegmentPoints, isClosed=False, colorLine=(255, 0, 0)): 
    cv2.polylines(imgIn, [lineSegmentPoints], isClosed, colorLine, thickness=4)

def OverlayLineSegmentsFill(imgIn, lineSegmentPoints, isClosed=False, colorLine=(255, 0, 0), colorFill=(0, 128, 0)): 
    cv2.fillConvexPoly(imgIn, lineSegmentPoints, colorFill)
    cv2.polylines(imgIn, [lineSegmentPoints], isClosed, colorLine, thickness=5)


# In[13]:


def ImageProc_PreProcPipeline(imgRaw, dictCameraCalVals):
    imageRecsPreProc = []

    imgUndistort = CameraCal_Undistort(imgRaw, dictCameraCalVals)
    imageRecsPreProc.append( (imgUndistort, "imgUndistort") ) 
    
    imgDePerspect = CameraCal_DoPerspectiveTransform(imgUndistort, dictCameraCalVals["warpMatrix"])
    imageRecsPreProc.append( (imgDePerspect, "imgDePerspect") ) 
    
    imgHSL =  cv2.cvtColor(imgDePerspect, cv2.COLOR_RGB2HLS)
    #imgHSL_Hue, imgHSL_HueThr = HSLThreshold(imgHSL, 0, (20, 25))
    #imgHSL_Lit, imgHSL_LitThr = HSLThreshold(imgHSL, 1, (90, 100))
    imgHSL_Sat, imgHSL_SatThr = ImageProc_HSLThreshold(imgHSL, 2, (120, 255))
    imageRecsPreProc.append( (imgHSL_Sat, "imgHSL_Sat", {"cmap":"gray"}) ) 
    imageRecsPreProc.append( (imgHSL_SatThr, "imgHSL_SatThr", {"cmap":"gray"}) ) 

    sobel_kernel = 5
    imgSobelMagThr = SobelThesholdMag(imgDePerspect, sobel_kernel=3, threshMinMax=(30, 100))
    imageRecsPreProc.append( (imgSobelMagThr, "imgSobelMagThr", {"cmap":"gray"}) ) 

    imgSatThrOrSobelMagThr = np.zeros_like(imgSobelMagThr)
    imgSatThrOrSobelMagThr[(imgSobelMagThr==1) | (imgHSL_SatThr==1)] = 1
    imageRecsPreProc.append( (imgSatThrOrSobelMagThr, "imgSatThrOrSobelMagThr", {"cmap":"gray"}) ) 

    return imageRecsPreProc


# In[14]:



class CImageLine(object):
    """
    This class holds state information of each line (LEFT, RIGHT) 
    because prior state is a factor in future line determinations.
    """

    def __init__(self, name, dictCameraCalVals):
        self.name = name
        self.dictCameraCalVals = dictCameraCalVals
        
        self.prevPolyLine = None
        self.polyCoeff = None
        self.curveRadiusMeters = 1
        
    def HasPolyFit(self):
        hasPolyFit = (self.polyCoeff != None)
        return(hasPolyFit)
    
    def SelectPixelsUsingPolynomial(self, argPolyCoeff, imgIn, selectionWidth = 100):
        """
        Use the supplied parabola coeffients (from the previous frame) to create a curved mask
        for selecting pixels to use in the next poly fit
        """
        # Coeffient vars. For readability
        coA, coB, coC = argPolyCoeff[0],argPolyCoeff[1],argPolyCoeff[2],

        imgSelectionOut = np.dstack((imgIn, imgIn, imgIn))

        # Find image NonZero pixels
        pixelsNon0 = imgIn.nonzero()
        pixelsNon0X = np.array(pixelsNon0[1])
        pixelsNon0Y = np.array(pixelsNon0[0])

        # Filter in all pixels within +/- margin of the polynomial
        selectedPixelIndices = ((pixelsNon0X > (coA * (pixelsNon0Y**2) + coB * pixelsNon0Y + coC - selectionWidth)) 
                              & (pixelsNon0X < (coA * (pixelsNon0Y**2) + coB * pixelsNon0Y + coC + selectionWidth)))


        # Get the selected pixels
        selectedPixelsX = pixelsNon0X[selectedPixelIndices]
        selectedPixelsY = pixelsNon0Y[selectedPixelIndices] 

        return selectedPixelsX, selectedPixelsY, imgSelectionOut

    def CalcPolyFit(self, imgIn):
        imageHeight = imgIn.shape[0]
        minpix = 100

        # Select pixels for next poly fit
        if (self.HasPolyFit() == True):
            selectedPixelsX, selectedPixelsY, imgSelectedPixels = self.SelectPixelsUsingPolynomial(self.polyCoeff, imgIn)
        else:
            # selectedCoordsY, selectedCoordsX = Random_PolyPoints(selectedCoordsY)
            print("    No previous polynomial. Searching with sliding window...", end='', flush=True)
            selectedPixelsX, selectedPixelsY, imgSelectedPixels = SelectPixelsUsingSlidingWindow(imgIn, self.name)

        if (len(selectedPixelsY) > minpix):
            # Do the poly fit on the selected pixels
            polyCoeffNew = np.polyfit(selectedPixelsY, selectedPixelsX, 2)
        else:
            # Just reuse the old poly
            polyCoeffNew = self.polyCoeff
            
        self.polyCoeff = polyCoeffNew
        
        # LANE LINE RADIUS CALCULATION
        evalRadiusAtpixY = imageHeight - 100
        curveRadiusMetersNew = CalcRadiusOfCurvatureParabola(polyCoeffNew, evalRadiusAtpixY, self.dictCameraCalVals["metersPerPixX"], self.dictCameraCalVals["metersPerPixY"])
        self.curveRadiusMeters = curveRadiusMetersNew
 
        imgSelection = PlotSelectedAndPolynomial(polyCoeffNew, selectedPixelsX, selectedPixelsY, imgIn)

        return polyCoeffNew, imgSelection
    


# In[15]:


def GetImageIteratorFromDir():
    """ 
    Creates and returns an iterator that supplies 'imageRecord' tuples (image, imageName)
    """
    dirImagesIn = "ImagesIn/TestImagesIn/POVRaw/"
    #dirImagesIn = "ImagesIn/VideosIn/project_video_frames/"

    #imgInFileNames = [dirImagesIn + "straight_lines1.jpg", dirImagesIn + "straight_lines2.jpg"]
    #imgInFileNames = [dirImagesIn + "project_video_f1192.jpg"]
    imgInFileNames = [dirImagesIn + "straight_lines1.jpg", dirImagesIn + "straight_lines1.jpg"]
    #imgInFileNames = [dirImagesIn + "project_video_f1192.jpg", dirImagesIn + "project_video_f1194.jpg", dirImagesIn + "project_video_f1196.jpg", dirImagesIn + "project_video_f1198.jpg"]
    #imgInFileNames = [dirImagesIn + "project_video_f0611.jpg", dirImagesIn + "project_video_f0612.jpg"]
    #imgInFileNames = [dirImagesIn + "project_video_f0609.jpg", dirImagesIn + "project_video_f0610.jpg", dirImagesIn + "project_video_f0611.jpg", dirImagesIn + "project_video_f0612.jpg", dirImagesIn + "project_video_f0613.jpg", dirImagesIn + "project_video_f0614.jpg"]
    #imgInFileNames = [dirImagesIn + "project_video_f1026.jpg", dirImagesIn + "project_video_f1027.jpg", dirImagesIn + "project_video_f1028.jpg", dirImagesIn + "project_video_f1029.jpg", dirImagesIn + "project_video_f1030.jpg", dirImagesIn + "project_video_f1031.jpg"]
    #imgInFileNames = [dirImagesIn + "project_video_f1030.jpg", dirImagesIn + "project_video_f1031.jpg"]
    #imgInFileNames = [dirImagesIn + "project_video_f0800.jpg", dirImagesIn + "project_video_f0801.jpg"]
    imageIter = ((mpimg.imread(imgInFileName), imgInFileName) for imgInFileName in imgInFileNames)

    return(imageIter)


# # ==================== P2Main() ===================
# ### This is the main cell/entry point for this notebook. 
# ## This cell autoruns!

# In[16]:



def P2Main(rawImgSrcIter, dictCameraCalVals):
    """
    Run the full lane finding process on each image provided by
    rawImgSrcIter iterator
    """
    frameIndex = 0  
    imageFrames = []   
    
    # Create 2 expected LaneLine structs
    imageLines = [CImageLine("LEFT", dictCameraCalVals), CImageLine("RIGHT", dictCameraCalVals)]
  
    # For every raw POV image coming from the camera
    for inputImageObj in rawImgSrcIter:
        # My file iterator supply an 'imageRecord' tuple (image, imgNameString)
        # The moviePy iterator only supplies an image. 
        # This if-else handles the two cases 
        if type(inputImageObj) is tuple:
            (imgRawPOV, imgRawPOVName) = inputImageObj
        else:
            imgRawPOV = inputImageObj
            imgRawPOVName = "frame{:04}".format(frameIndex)
            
        imageHeight = imgRawPOV.shape[0]
        imageWidth = imgRawPOV.shape[1]

        imageRecsCurFrame = [] # A list image records generated at every step of processing
        imageRecsCurFrame.append( (imgRawPOV, imgRawPOVName) ) 
        
        # IMAGE PREPROCESSING
        # Perform preprocessing on the raw image
        # Returns all image records of each stage of the pipeline
        # The last image is a suitable image for lane line detection/polynomial calculation
        print("P2Main->PreProc(f{:04} {}): =====================".format(frameIndex, imgRawPOVName))
        imageRecsPreProc = ImageProc_PreProcPipeline(imgRawPOV, dictCameraCalVals)
        
        # Append PreProc image recs to the dev Images display container for this frame
        imageRecsCurFrame.extend(imageRecsPreProc)
        
        # Get the most recent image from the image preprocess image records to use for lane finding
        imgLineFindSrc = imageRecsPreProc[-1][0]

        # For each of the expected lane lines
        for curImageLine in imageLines:
            
            # IMAGE LANE LINE DETECTION/POLYNOMIAL CALCULATION
            print("    P2Main->CalcPolyFit({}):".format(curImageLine.name), end='', flush=True)
            polyCoeffNew, imgLanePixSelection = curImageLine.CalcPolyFit(imgLineFindSrc)
            
            imageRecsCurFrame.append( (imgLanePixSelection, "imgLanePixSelection" + curImageLine.name, {"cmap":"gray"} ))       
 
            print("    Radius_{} = {}m".format(curImageLine.name, curImageLine.curveRadiusMeters), end='', flush=True)

            print(" ")

        # AVERAGE LANE LINE RADIUS ANALYSIS
        curveRadiusMetersRatio = abs(imageLines[0].curveRadiusMeters/imageLines[0].curveRadiusMeters)
        if (curveRadiusMetersRatio > 1.5 or curveRadiusMetersRatio < 0.66):
            radiusSuspicion = "SUSPICIOUS radius difference!!!"
        else:
             radiusSuspicion = ""
                
        curveRadiusMetersAvg = np.mean([imageLines[0].curveRadiusMeters, imageLines[0].curveRadiusMeters])
        print("    P2Main->Radius_{} = {}m {}".format("AVG", curveRadiusMetersAvg, radiusSuspicion))
        
        # CREATE COMPOSITE LANE POLYNOMIAL IMAGE   
        doReverseSegments = False

        polynomialLineSegmentsAccum = []
        for curImageLine in imageLines:
            polynomialLineSegmentsCur = EvalPolyToLineSegments(curImageLine.polyCoeff, imageHeight, doReverseSegments)
            polynomialLineSegmentsAccum.append(polynomialLineSegmentsCur)
            doReverseSegments = not doReverseSegments # To make the polygon correctly closed, need to reverse order
            
        polynomialLineSegmentsCombo = np.concatenate(polynomialLineSegmentsAccum)
        imgPolyCombo = np.zeros_like(imgRawPOV)
        isClosed = True
        OverlayLineSegmentsFill(imgPolyCombo, polynomialLineSegmentsCombo, isClosed)
        imageRecsCurFrame.append( (imgPolyCombo, "imgPolyCombo"))    

        # OVERLAY LANE POLYNOMIAL ON POV PERSPECTIVE VIEW
        # Get the first image from the image preprocess image records to use imgFinal overlay
        imgLineUndistort = imageRecsPreProc[0][0]

        imgRePerspect = CameraCal_DoPerspectiveTransform(imgPolyCombo, dictCameraCalVals["warpMatrix"], doInverse = True)          
        imageRecsCurFrame.append( (imgRePerspect, "imgRePerspect"))
        
        imgFinal = cv2.addWeighted(imgRePerspect, 1, imgLineUndistort, 0.7, 0)
        textFrameIndex = "f{:04} Radius L,R ={:5.0f}m, {:5.0f}m".format(frameIndex, imageLines[0].curveRadiusMeters, imageLines[1].curveRadiusMeters)
        textRadius = "RadiusAvg= {:5.0f}m {}".format(curveRadiusMetersAvg, radiusSuspicion)

        DrawText(imgFinal, textFrameIndex, posLL = (10,25), colorFont=(255, 255, 255), fontScale = 2)
        DrawText(imgFinal, textRadius, posLL = (10,70), colorFont=(255, 255, 255), fontScale = 2)
        imageRecsCurFrame.append( (imgFinal, "imgFinal"))
        
        imageFrames.append(imgFinal)
 
        if g_doDisplayDebugImages:
            PlotImageRecords(imageRecsCurFrame, doSaveDebugImages)
            #key = cv2.waitKey(5000)
            
        frameIndex+=1
        print(" ")
            
    return imageFrames
#============================= Main Invocation Prep =================
g_dictCameraCalVals = CameraCal_GetCalVals(g_cameraDistortionCalValsFileName, g_cameraPerspectiveWarpMatrixFileName)
g_imageIter = GetImageIteratorFromDir() # Provide a source of raw camera POV images for processing

#================= P2Main() invocation
g_imageFrames = P2Main(g_imageIter, g_dictCameraCalVals)  


# In[17]:


def GetMovieIterator(fileNameBase):
    fileExtIn = ".mp4"
    dirIn = "ImagesIn/VideosIn/"
    fileNameIn  = dirIn + fileNameBase + fileExtIn
    
    #movieClipIn = VideoFileClip(fileNameIn).subclip(39, 43) difficult bridge for project_video
    movieClipIn = VideoFileClip(fileNameIn)
    imageIter = movieClipIn.iter_frames()
    return imageIter
    


# In[18]:


import moviepy.editor as mp
from moviepy.editor import VideoFileClip

def MakeMovie():
    #fileNameBase = "challenge_video"
    fileNameBase = "project_video"
    
    imageIter = GetMovieIterator(fileNameBase) # Provide a source of raw camera POV images for processing
    imageFrames =  P2Main(imageIter, g_dictCameraCalVals)
    movieClipOut = mp.ImageSequenceClip(imageFrames, fps=25)
    
    #dirOut = "ImagesOut/VideosOut/"
    dirOut = "./" # For the submission output to root
    if (not os.path.exists(dirOut)):
        os.makedirs(dirOut)

    strDT = "{:%Y-%m-%dT%H:%M:%S}".format(datetime.datetime.now())
    #fileOutName = dirOut + fileNameBase + strDT + ".mp4"
    fileOutName = "project_video_out.mp4"
    movieClipOut.write_videofile(fileOutName, fps=25, codec='mpeg4')

#g_imageIter = GetImageIteratorFromDir() # Provide a source of raw camera POV images for processing
if g_doAutorunMovieProcess:
    MakeMovie()


# In[19]:


def Random_PolyPoints(ploty):
    '''
    Generates fake data to use for calculating lane curvature.
    In your own project, you'll ignore this function and instead
    feed in the output of your lane detection algorithm to
    the lane curvature calculation.
    '''
    # Set random seed number so results are consistent for grader
    # Comment this out if you'd like to see results on different random data!
    np.random.seed(0)

    quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
    offsetX = 200
    
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([offsetX + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) for y in ploty])
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    return ploty, leftx


# ## Snippets & Junk Code

# In[20]:


def Polynomial_Eval(polyCoeff, inVals):
    outVals = self.polyCoeff[0] * inVals**2 + self.polyCoeff[1] * inVals + self.polyCoeff[2]
    return inVals, outVals

def PlotSelectedAndPolynomialOld(argPolyCoeff, inDomainCoords, xRangeCoords):
        coA, coB, coC = argPolyCoeff[0], argPolyCoeff[1], argPolyCoeff[2],
        polyEvalOutCoords = coA * inDomainCoords**2 + coB * inDomainCoords + coC
        
        fig, ax = plt.subplots()      
        plt.plot(xRangeCoords, inDomainCoords, 'o', color='red', markersize=1)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(polyEvalOutCoords, inDomainCoords, color='green', linewidth=2)
        plt.gca().invert_yaxis() # to visualize as we do the images
        plt.show()
        


# In[21]:


def MakeMovieTest(rawImgSrcIter):
    frameIndex = 0
    movieFramesAccum = []
    
    for (imgRawPOV, imgRawPOVName) in rawImgSrcIter:
        print("Frame{:04} {}".format(frameIndex, imgRawPOVName))
        #movieFramesAccum.append(mp.ImageClip(imgRawPOV))
        movieFramesAccum.append(imgRawPOV)
        frameIndex+=1

    movieClip = mp.ImageSequenceClip(movieFramesAccum, fps=25)
    #movieClips = mp.concatenate_videoclips(movieFramesAccum, method="compose")
    movieClip.write_videofile("test.mp4", fps=25, codec='mpeg4')

import pprint
pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(threshold=1000)   
np.set_printoptions(threshold=np.nan)
g_testpolyCoeff = None
g_testimgPlotIn = None
g_inPixelsX = None
g_inPixelsY = None
np.set_printoptions(threshold=np.nan)

def PlotSelectedAndPolynomial(polyCoeff, inPixelsX, inPixelsY, imgInBW):
    global g_testpolyCoeff 
    global g_testimgPlotIn
    global g_inPixelsX 
    global g_inPixelsY 

    g_testpolyCoeff = polyCoeff
    g_testimgPlotIn = imgInBW
    g_inPixelsX = inPixelsX
    g_inPixelsY = inPixelsY
    
    imageHeight = imgInBW.shape[0]
    imageWidth = imgInBW.shape[1]

    imgInGrey = imgInBW * 64
    imgOut = np.dstack((imgInGrey, imgInGrey, imgInGrey))
    
    imgOut[inPixelsY, inPixelsX] = [0, 255, 0]
    
    coA, coB, coC = polyCoeff[0], polyCoeff[1], polyCoeff[2]

    polyInY = np.array([y for y in range(imageHeight) if y % 8 == 0])
    polyOutXf = coA * polyInY**2 + coB * polyInY + coC  
    polyOutX = polyOutXf.astype(int)
    #imgOut[polyInY, polyOutX] = [255]#, 255, 255]

    points = np.array(list(zip(polyOutX, polyInY)))
    #print("points", points)
    isClosed=False
    cv2.polylines(imgOut, [points], isClosed, (255, 0, 0), thickness=4)

    return imgOut

#pp.pprint("coeff",g_testpolyCoeff )
#print("g_testimgPlotIn",g_testimgPlotIn )
#pp.pprint("g_testimgPlotIn",g_testimgPlotIn )

#pprint.pprint(g_testpolyCoeff )
#pprint.pprint(g_testinDomainCoords )
#pprint.pprint(g_testimgPlotIn )

#print("g_testinDomainCoords",g_testinDomainCoords )

g_imgPlotOut = PlotSelectedAndPolynomial(g_testpolyCoeff, g_inPixelsX, g_inPixelsY, g_testimgPlotIn)
get_ipython().run_line_magic('matplotlib', 'qt4')
plt.imshow(g_imgPlotOut)
#plt.imshow(g_testimgPlotIn, cmap="gray")

#    if doReverseSegments:
#        lineSegmentPoints = lineSegmentPoints[::-1]

#np.set_printoptions(threshold=np.nan)
#import pprint
#pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(threshold=1000)   
np.set_printoptions(threshold=np.nan)
g_testpolyCoeff = None
g_testimgPlotIn = None
g_inPixelsX = None
g_inPixelsY = None

def PlotSelectedAndPolynomial(polyCoeff, inPixelsX, inPixelsY, imgInBW):
    global g_testpolyCoeff 
    global g_testimgPlotIn
    global g_inPixelsX 
    global g_inPixelsY 

    #g_testpolyCoeff = polyCoeff
    #g_testimgPlotIn = imgInBW
    #g_inPixelsX = inPixelsX
    #g_inPixelsY = inPixelsY
    
    imageHeight = imgInBW.shape[0]
    imageWidth = imgInBW.shape[1]

    # Set unselected pixels to grey
    imgInGrey = imgInBW * 64
    imgOut = np.dstack((imgInGrey, imgInGrey, imgInGrey))
    
    # Display the selected pixels
    #imgOut[inPixelsY, inPixelsX] = [0, 255, 0]
    imgOut[inPixelsY, inPixelsX, 1] = 255 # Set selected pixels to red
    
    #coA, coB, coC = polyCoeff[0], polyCoeff[1], polyCoeff[2]

    # Evaluate the polynomial
    #polyInY = np.array([y for y in range(imageHeight) if y % 8 == 0])
    #polyOutXf = coA * polyInY**2 + coB * polyInY + coC  
    #polyOutX = polyOutXf.astype(int)

    #lineSegmentPoints = np.array(list(zip(polyOutX, polyInY)))
    lineSegmentPoints = EvalPolyToLineSegments(polyCoeff, imageHeight, doReverseSegments=False)
    #isClosed=False
    #cv2.polylines(imgOut, [lineSegmentPoints], isClosed, (255, 0, 0), thickness=4)
    PlotLineSegments(imgOut, lineSegmentPoints, isClosed=False)
    return imgOut

#g_imgPlotOut = PlotSelectedAndPolynomial(g_testpolyCoeff, g_inPixelsX, g_inPixelsY, g_testimgPlotIn)
#%matplotlib qt4
#plt.imshow(g_imgPlotOut)
#plt.imshow(g_testimgPlotIn, cmap="gray")
        #selectedCoordsY = np.linspace(0, imageHeight-1, num=imageHeight)
#"RadiusAvg = {:0.0f}m {}".format(curveRadiusMetersAvg, radiusSuspicion)



    # DevDebug Get sample lineFindSrc image
    #dirImagesIn = "ImagesIn/TestImagesIn/PipelineStages/"
    #imgInFileNameBase = "warped_example.jpg"
    #imgInFileName = dirImagesIn + imgInFileNameBase
    #imgLineFindSrcDebug = mpimg.imread(imgInFileName)  
    #mageRecsPreProc.append( (imgLineFindSrcDebug, "imgLineFindSrcDebug", {"cmap":"gray"}) )       


# In[ ]:


def GlobalCandidates():
    fileNameBase = "project_video"
    fileExtIn = ".mp4"
    fileExtOut = "jpg"
    dirIn = "ImagesIn/VideosIn/"
    dirOut = "ImagesIn/VideosIn/" + fileNameBase + "_frames/"
    fileNameIn  = dirIn + fileNameBase + fileExtIn
    fileNameOutFmt = dirOut + fileNameBase + "_f{:04}." + fileExtOut
    
    #if (not os.path.exists(dirOut)):
    #    os.makedirs(dirOut)


# In[ ]:



def PlotFigureToRGBArray(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

def PlotSelectedAndPolynomialOld(polyCoeff, inDomainCoords, xRangeCoords, imgPlotIn):
    imageWidth = imgPlotIn.shape[1]
    imageHeight = imgPlotIn.shape[0]
    coA, coB, coC = polyCoeff[0], polyCoeff[1], polyCoeff[2]

    polyEvalOutCoords = coA * inDomainCoords**2 + coB * inDomainCoords + coC

    fig, ax = plt.subplots()

    plt.imshow(imgPlotIn)
    plt.xlim(0, imageHeight)
    plt.ylim(0, imageHeight)
    plt.axis('off')
    plt.plot(xRangeCoords, inDomainCoords, 'o', color='red', markersize=1)
    plt.plot(polyEvalOutCoords, inDomainCoords, color='green', linewidth=2)
    plt.gca().invert_yaxis() # to visualize as we do the images
    imgPlotOut = PlotFigureToRGBArray(fig)
    plt.clf()
    return imgPlotOut
    
#-------------------------------------------------------------------
def PlotImageRecordsOld(imgRecords):
    #fig = plt.gcf()
    fig = plt.figure()
    fig.set_size_inches(18,12)
    #fig.set_dpi(180)

    numImages = len(imgRecords)
    numCols = 3
    numRows = math.ceil(numImages/numCols)
    for recIndex, imgRecord in enumerate(imgRecords):
        numFields = len(imgRecord)
        img = imgRecord[0]
        if (numFields >= 2):
            imgName =  imgRecord[1]
        else:
            imgName =  "img_" + str(recIndex)
            
        if (numFields >= 3):
            kwArgs =  imgRecord[2]
        else:
            kwArgs =  {}
                
        plt.subplot(numRows, numCols, recIndex+1)
        plt.title(imgName)
        plt.imshow(img, **kwArgs)
       
    plt.show()
    fig.tight_layout()
    

