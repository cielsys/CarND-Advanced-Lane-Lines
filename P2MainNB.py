
# coding: utf-8

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


get_ipython().run_cell_magic('HTML', '', '<style> code {background-color : orange !important;} </style>\nfrom IPython.core.display import display, HTML\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# In[3]:


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


# In[4]:


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


# ### Display/Plotting Utilities

# In[5]:



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
def PlotImageRecords(imgRecords):
    fig = plt.gcf()
    fig.set_size_inches(12,9)
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
        
    fig.tight_layout()
    


# In[ ]:


import pprint
pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(threshold=1000)   
np.set_printoptions(threshold=np.nan)
g_testpolyCoeff = None
g_testimgPlotIn = None
g_inPixelsX = None
g_inPixelsY = None


# In[53]:


#np.set_printoptions(threshold=np.nan)

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

    imgInGrey = imgInBW * 64
    imgOut = np.dstack((imgInGrey, imgInGrey, imgInGrey))
    
    imgOut[inPixelsY, inPixelsX] = [0, 255, 0]
    
    coA, coB, coC = polyCoeff[0], polyCoeff[1], polyCoeff[2]

    polyInY = np.array([y for y in range(imageHeight) if y % 8 == 0])
    polyOutXf = coA * polyInY**2 + coB * polyInY + coC  
    polyOutX = polyOutXf.astype(int)

    points = np.array(list(zip(polyOutX, polyInY)))
    isClosed=False
    cv2.polylines(imgOut, [points], isClosed, (255, 0, 0), thickness=4)

    return imgOut

#g_imgPlotOut = PlotSelectedAndPolynomial(g_testpolyCoeff, g_inPixelsX, g_inPixelsY, g_testimgPlotIn)
#%matplotlib qt4
#plt.imshow(g_imgPlotOut)
#plt.imshow(g_testimgPlotIn, cmap="gray")


# In[8]:


import cv2
import numpy as np

def testPolyLine():
    img = np.zeros((768, 1024, 3), dtype='uint8')

    points = np.array([[910, 641], [206, 632], [696, 488], [458, 485]])
    cv2.polylines(img, [points], 1, (255,255,255),thickness=1)

    winname = 'example'
    cv2.namedWindow(winname)
    cv2.imshow(winname, img)
    cv2.waitKey(3000)
    cv2.destroyWindow(winname)
#testPolyLine()


# ### Camera Calibration Utilities and Processing

# In[9]:


def CmaeraCal_GetCalVals(cameraDistortionCalValsFileName, cameraPerspectiveWarpMatrixFileName):
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

def CameraCal_DoPerspectiveTransform(imgIn, matTransform):
    img_size = (imgIn.shape[1], imgIn.shape[0])
    interpolation = cv2.INTER_CUBIC
    imgOut = cv2.warpPerspective(imgIn, matTransform, img_size, flags=interpolation)
    return imgOut


# In[10]:


# See http://www.intmath.com/applications-differentiation/8-radius-curvature.php
def CalcRadiusOfCurvatureParabola(funcCoeffients, evalAtpixY, metersPerPixX=1, metersPerPixY=1):
   # Convert parabola from pixels to meters
   coA = funcCoeffients[0] * metersPerPixX / (metersPerPixY**2)
   coB = funcCoeffients[1] * metersPerPixY
   #coC = funcCoeffients[2] * mPerPixY # Not used

   mY = evalAtpixY * metersPerPixY

   radiusNumerator = pow((1 + (2 * coA * mY + coB)**2), 3/2)
   radiusDenominator = abs(2*coA)
   radius = radiusNumerator/radiusDenominator                           
   return(radius)


# In[11]:


def SelectPixelsUsingSlidingWindow(imgIn, whichLine):    
    margin = 100 # Set the width of the windows +/- margin    
    minpix = 50 # Set minimum number of pixels found to recenter window    
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
    #%matplotlib qt4
    #%matplotlib inline
    #%matplotlib notebook
    selectedPixelsX, selectedPixelsY, imgSelectionOut = SelectPixelsUsingSlidingWindow(g_imgTest, "RIGHT")
    plt.figure(figsize=(12,8))
    imgSelectionOut[selectedPixelsY, selectedPixelsX] = [255, 0, 0]
    plt.imshow(imgSelectionOut)
    plt.tight_layout()

#=========== Test invocation
g_testImgFileName = "ImagesIn/TestImagesIn/PipelineStages/warped_example.jpg"
g_imgTest = mpimg.imread(g_testImgFileName)
#Test_SelectPixelsUsingSlidingWindow()


# In[12]:


#g_testPolyCoeffLeft = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
#g_testPolyCoeffRight = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])

class CImageLine(object):
    
    def __init__(self, name):
        self.prevPolyLine = None
        self.name = name
        self.polyCoeff = None # g_testPolyCoeffLeft
            
    def HasPolyFit(self):
        hasPolyFit = (self.polyCoeff != None)
        return(hasPolyFit)
    
    def SelectPixelsUsingPolynomial(self, argPolyCoeff, imgIn, selectionWidth = 100):
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
        #selectedCoordsY = np.linspace(0, imageHeight-1, num=imageHeight)

        if (self.HasPolyFit() == True):
            selectedPixelsX, selectedPixelsY, imgSelection = self.SelectPixelsUsingPolynomial(self.polyCoeff, imgIn)
        else:
            # selectedCoordsY, selectedCoordsX = Random_PolyPoints(selectedCoordsY)
            print("    No previous polynomial. Searching with sliding window...")
            selectedPixelsX, selectedPixelsY, imgSelection = SelectPixelsUsingSlidingWindow(imgIn, self.name)

        polyCoeffNew = np.polyfit(selectedPixelsY, selectedPixelsX, 2)       
        self.polyCoeff = polyCoeffNew

        imgPlot = imgIn.copy()
        imgSelection = PlotSelectedAndPolynomial(polyCoeffNew, selectedPixelsX, selectedPixelsY, imgPlot)

        return polyCoeffNew, imgSelection
    
#%matplotlib notebook
#P2Main(g_imageIter)


# In[13]:


def ImageProc_HSLThreshold(imgHLSIn, channelNum, threshMinMax=(0, 255)):
    imgOneCh = imgHLSIn[:,:,channelNum]
    imgBWMask = np.zeros_like(imgOneCh)
    imgBWMask[(imgOneCh > threshMinMax[0]) & (imgOneCh <= threshMinMax[1])] = 1
    return imgOneCh, imgBWMask


# ### Image Preprocessing Pipeline

# In[14]:


def ImageProc_PreProcPipeline(imgRaw, dictCameraCalVals):
    imageRecsPreProc = []

    imgUndistort = CameraCal_Undistort(imgRaw, dictCameraCalVals)
    imageRecsPreProc.append( (imgUndistort, "imgUndistort") ) 
    
    imgPerspect = CameraCal_DoPerspectiveTransform(imgUndistort, dictCameraCalVals["warpMatrix"])
    imageRecsPreProc.append( (imgPerspect, "imgPerspect") ) 
    
    imgHSL =  cv2.cvtColor(imgPerspect, cv2.COLOR_RGB2HLS)
    #imgHSL_Hue, imgHSL_HueThr = HSLThreshold(imgHSL, 0, (20, 25))
    #imgHSL_Lit, imgHSL_LitThr = HSLThreshold(imgHSL, 1, (90, 100))
    imgHSL_Sat, imgHSL_SatThr = ImageProc_HSLThreshold(imgHSL, 2, (90, 255))
    imageRecsPreProc.append( (imgHSL_Sat, "imgHSL_Sat", {"cmap":"gray"}) ) 
    imageRecsPreProc.append( (imgHSL_SatThr, "imgHSL_SatThr", {"cmap":"gray"}) ) 

    # DevDebug Get sample lineFindSrc image
    dirImagesIn = "ImagesIn/TestImagesIn/PipelineStages/"
    imgInFileNameBase = "warped_example.jpg"
    imgInFileName = dirImagesIn + imgInFileNameBase
    imgLineFindSrcDebug = mpimg.imread(imgInFileName)  
    #mageRecsPreProc.append( (imgLineFindSrcDebug, "imgLineFindSrcDebug", {"cmap":"gray"}) )       

    return imageRecsPreProc


# In[55]:


#calFileInNames = glob.glob('camera_cal/cal*.jpg')
#calFileInNames = ['camera_cal/calibration2.jpg']

def GetImageIteratorFromDir():
    dirImagesIn = "ImagesIn/TestImagesIn/POVRaw/"
    dirImagesIn = "ImagesIn/VideosIn/project_video_frames/"
    fileNameBase = "straight_lines1.jpg"

    #imgInFileNames = [dirImagesIn + "straight_lines1.jpg", dirImagesIn + "straight_lines2.jpg"]
    imgInFileNames = [dirImagesIn + "straight_lines1.jpg", dirImagesIn + "straight_lines1.jpg"]
    imgInFileNames = [dirImagesIn + "project_video_f1192.jpg", dirImagesIn + "project_video_f1194.jpg", dirImagesIn + "project_video_f1196.jpg", dirImagesIn + "project_video_f1198.jpg"]
    #imgInFileNames = [dirImagesIn + "project_video_f1192.jpg"]
    imageIter = ((mpimg.imread(imgInFileName), imgInFileName) for imgInFileName in imgInFileNames)

    return(imageIter)


# In[56]:



def P2Main(rawImgSrcIter, dictCameraCalVals):
    
    # Create 2 expected LaneLine structs
    imageLines = [CImageLine("LEFT"), CImageLine("RIGHT")]
    
    # For every raw POV image coming from the camera
    for (imgRawPOV, imgRawPOVName) in rawImgSrcIter:
        curveRadiusMetersList = []
        imageRecsCurFrame = [] # A list image records generated at every step of processing
        imageRecsCurFrame.append( (imgRawPOV, imgRawPOVName) ) 
        
        # IMAGE PREPROCESSING
        # Perform preprocessing on the raw image
        # Returns all image records of each stage of the pipeline
        # The last image is a suitable image for lane line detection/polynomial calculation
        print("P2Main->PreProcPipeline({}): =====================".format(imgRawPOVName))
        imageRecsPreProc = ImageProc_PreProcPipeline(imgRawPOV, dictCameraCalVals)
        
        # Append PreProc image recs to the dev Images display container for this frame
        imageRecsCurFrame.extend(imageRecsPreProc)
        
        # Get the most recent image from the image preprocess image records to use for lane finding
        imgLineFindSrc = imageRecsPreProc[-1][0]

        # For each of the expected lane lines
        for curImageLine in imageLines:
            
            # IMAGE LANE LINE DETECTION/POLYNOMIAL CALCULATION
            print("    P2Main->CalcPolyFit({}):".format(curImageLine.name))
            polyCoeffNew, imgLanePixSelection = curImageLine.CalcPolyFit(imgLineFindSrc)
            
            imageRecsCurFrame.append( (imgLanePixSelection, "imgLanePixSelection" + curImageLine.name, {"cmap":"gray"} ))       
 
            # LANE LINE RADIUS CALCULATION
            imgHeight = imgRawPOV.shape[0]
            evalRadiusAtpixY = imgHeight - 20
            curveRadiusMetersCur = CalcRadiusOfCurvatureParabola(polyCoeffNew, evalRadiusAtpixY, dictCameraCalVals["metersPerPixX"], dictCameraCalVals["metersPerPixY"])
            print("    P2Main->Radius_{} = {:0.0f}m".format(curImageLine.name, curveRadiusMetersCur))
            curveRadiusMetersList.append(curveRadiusMetersCur)
            print(" ")
        
        # AVERAGE LANE LINE RADIUS ANALYSIS
        #curveRadiusMetersRatio = abs(curveRadiusMetersList[0]/curveRadiusMetersList[1])
        curveRadiusMetersRatio = 1
        if (curveRadiusMetersRatio > 1.5 or curveRadiusMetersRatio < 0.66):
            radiusSuspicion = "SUSPICIOUS radius difference!!!"
        else:
             radiusSuspicion = ""
                
        curveRadiusMetersAvg = np.mean(curveRadiusMetersList)
        print("    P2Main->Radius_{} = {:0.0f}m {}".format("AVG", curveRadiusMetersAvg, radiusSuspicion))
        print(" ")

        # DEVDEBUG IMAGE DISPLAY
        doDisplayDebugImages = True
        if doDisplayDebugImages:
            get_ipython().run_line_magic('matplotlib', 'qt4')
            PlotImageRecords(imageRecsCurFrame)
            key = cv2.waitKey(2000)


#============================= Main Invocation Prep =================
# Specify filenames that contain the camera calibration information
g_cameraDistortionCalValsFileName = "CameraDistortionCalVals.pypickle"
g_cameraPerspectiveWarpMatrixFileName = "CameraPerspectiveWarpMatrix.pypickle"

g_dictCameraCalVals = CmaeraCal_GetCalVals(g_cameraDistortionCalValsFileName, g_cameraPerspectiveWarpMatrixFileName)
g_imageIter = GetImageIteratorFromDir() # Provide a source of raw camera POV images for processing

#================= P2Main() invocation
P2Main(g_imageIter, g_dictCameraCalVals)


# ## Snippets & Junk Code

# In[ ]:


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
        


# In[ ]:


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

