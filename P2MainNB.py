
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


# In[6]:


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


# In[37]:


def Polynomial_Eval(polyCoeff, inVals):
    outVals = self.polyCoeff[0] * inVals**2 + self.polyCoeff[1] * inVals + self.polyCoeff[2]
    return inVals, outVals

def PlotSelectedAndPolynomialOld(argPolyCoeff, inDomainCoords, xRangeCoords):
        coA, coB, coC = argPolyCoeff[0], argPolyCoeff[1], argPolyCoeff[2],
        polyEvalOutCoords = coA*inDomainCoords**2 + coB*inDomainCoords + coC
        
        fig, ax = plt.subplots()      
        plt.plot(xRangeCoords, inDomainCoords, 'o', color='red', markersize=1)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(polyEvalOutCoords, inDomainCoords, color='green', linewidth=2)
        plt.gca().invert_yaxis() # to visualize as we do the images
        plt.show()

def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

def PlotSelectedAndPolynomial(argPolyCoeff, inDomainCoords, xRangeCoords, imgPlotIn):
        coA, coB, coC = argPolyCoeff[0], argPolyCoeff[1], argPolyCoeff[2],
        polyEvalOutCoords = coA*inDomainCoords**2 + coB*inDomainCoords + coC
        
        fig, ax = plt.subplots()
      
        plt.imshow(imgPlotIn)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(xRangeCoords, inDomainCoords, 'o', color='red', markersize=1)
        plt.plot(polyEvalOutCoords, inDomainCoords, color='green', linewidth=2)
        plt.gca().invert_yaxis() # to visualize as we do the images
        imgPlotOut = fig2rgb_array(fig)
        plt.clf()
        return imgPlotOut

#ImageRecord = namedtuple('Img', 'val left right')
#def Node(val, left=None, right=None):
#  return NodeT(val, left, right)
#        defaultargs = (None, "img_" + str(recIndex), {})
#        img, imgName, kwargs = tuple(map(lambda x, y: y if y is not None else x, defaultargs, imgRecord))

    
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
        #plt.axis('off')
        plt.imshow(img, **kwArgs)
        
    fig.tight_layout()
    


# In[30]:


def SelectPixelsUsingSlidingWindow(imgIn, whichLine):    
    margin = 100 # Set the width of the windows +/- margin    
    minpix = 50 # Set minimum number of pixels found to recenter window    
    numWindows = 9 # Number of sliding windows
    
    imageHeight = imgIn.shape[0]
    imageWidth = imgIn.shape[1]
    
    win = type('WindowInfo', (object,), {})  
    win.height = np.int(imageHeight//numWindows)
    win.width = margin * 2
   
    # Take a histogram of the bottom half of the image
    # and find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    if (whichLine == "LEFT"):
        histLeft = 0
        histRight = imageWidth//2
    else:
        histLeft = imageWidth//2
        histRight = imageWidth

    #print("histLR", histLeft, histRight)
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
    lineWidth = 3
    colorWinRect = (0,255,0)
    colorNextCenterXLine = (0, 130, 255)

    # Loop thru windows
    for windowIndex in range(numWindows):
        #print("======================= processing window: ", window)

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

        # Append these indices to the lists
        selectedPixelIndicesAccum.append(selectedPixelIndicesCur)

        #imgWindow = imgIn[win_y_low : win_y_high, win_x_low : win_x_high]
        imgWindow = imgIn[win.top : win.bottom, win.left : win.right]
        pixelsWindow0 = imgWindow.nonzero()
        pixelsWindow0X = pixelsWindow0[1]

        if (len(pixelsWindow0X) > minpix):
            pixelsWindow0XAvg = int(np.mean(pixelsWindow0X))
        else:
            pixelsWindow0XAvg = margin

        nextWindowCenterX = pixelsWindow0XAvg + win.left
        win.centerX = nextWindowCenterX

        Pt0 = (nextWindowCenterX, win.bottom)
        Pt1 = (nextWindowCenterX, win.top)
        cv2.line(imgSelectionOut, Pt0, Pt1, colorNextCenterXLine, lineWidth)

 
    # Extract left and right line pixel positions
    selectedPixelIndices = np.concatenate(selectedPixelIndicesAccum)
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

g_testImgFileName = "ImagesIn/TestImagesIn/PipelineStages/warped_example.jpg"
g_imgTest = mpimg.imread(g_testImgFileName)
Test_SelectPixelsUsingSlidingWindow()


# In[35]:


#left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
#right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])

class CImageLine(object):
    
    def __init__(self, name):
        self.prevPolyLine = None
        self.name = name
        self.polyCoeff = None #[ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02]
            
    def HasPolyFit(self):
        hasPolyFit = (self.polyCoeff != None)
        return(hasPolyFit)
    
    def SelectPixelsUsingPolynomial(self, argPolyCoeff, imgIn, selectionWidth = 100):
        # For readability
        coA, coB, coC = argPolyCoeff[0],argPolyCoeff[1],argPolyCoeff[2],
        
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

        return selectedPixelsX, selectedPixelsY, imgSelection
         
    def CalcPolyFit(self, imgIn):
        imageHeight = imgIn.shape[0]
        selectedCoordsY = np.linspace(0, imageHeight-1, num=imageHeight)

        if (self.HasPolyFit() == True):
            selectedPixelsX, selectedPixelsY, imgSelection = self.SelectPixelsUsingPolynomial(self.polyCoeff, imgIn)
        else:
            # selectedCoordsY, selectedCoordsX = Random_PolyPoints(selectedCoordsY)
            print("No previous polynomial. Searching with sliding window...")
            selectedPixelsX, selectedPixelsY, imgSelection = SelectPixelsUsingSlidingWindow(imgIn, self.name)

        polyCoeffNew = np.polyfit(selectedPixelsY, selectedPixelsX, 2)       
        self.polyCoeff = polyCoeffNew
        
        imgPlot = imgIn.copy()
        imgSelection = PlotSelectedAndPolynomial(polyCoeffNew, selectedPixelsY, selectedPixelsX, imgPlot)

        return polyCoeffNew, imgSelection
    
#%matplotlib notebook
#P2Main(g_imageIter)


# In[39]:


#calFileInNames = glob.glob('camera_cal/cal*.jpg')
#calFileInNames = ['camera_cal/calibration2.jpg']

def GetImageIteratorFromDir():
    dirImagesIn = "ImagesIn/TestImagesIn/POVRaw/"
    dirImagesIn = "ImagesIn/VideosIn/project_video_frames/"
    fileNameBase = "straight_lines1.jpg"

    #imgInFileNames = [dirImagesIn + "straight_lines1.jpg", dirImagesIn + "straight_lines2.jpg"]
    imgInFileNames = [dirImagesIn + "straight_lines1.jpg", dirImagesIn + "straight_lines1.jpg"]
    imgInFileNames = [dirImagesIn + "project_video_f1192.jpg"]
    imageIter = ((mpimg.imread(imgInFileName), imgInFileName) for imgInFileName in imgInFileNames)

    return(imageIter)


# In[63]:


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


# In[70]:


def ImageProc_PreProcPipeline(imgRaw, dictCameraCalVals):
    imageRecsPreProc = []

    imgUndistort = CameraCal_Undistort(imgRaw, dictCameraCalVals)
    imageRecsPreProc.append( (imgUndistort, "imgUndistort") ) 
    
    imgPerspect = DoPerspectiveTransform(imgUndistort, dictCameraCalVals["warpMatrix"])
    imageRecsPreProc.append( (imgPerspect, "imgPerspect") ) 

    # Get sample lineFindSrc image
    dirImagesIn = "ImagesIn/TestImagesIn/PipelineStages/"
    imgInFileNameBase = "warped_example.jpg"
    imgInFileName = dirImagesIn + imgInFileNameBase
    imgLineFindSrc = mpimg.imread(imgInFileName)
    imageRecsPreProc.append( (imgLineFindSrc, "imgLineFindSrc", {"cmap":"gray"}) )       

    return imageRecsPreProc
    


# In[65]:


def GetCameraCalVals(cameraDistortionCalValsFileName, cameraPerspectiveWarpMatrixFileName):
    dictCameraCalVals = pickle.load( open(cameraDistortionCalValsFileName, "rb" ) )
    dictCameraCalVals["warpMatrix"] = pickle.load( open(cameraPerspectiveWarpMatrixFileName, "rb" ) )
    return(dictCameraCalVals)


# In[71]:



def P2Main(rawImgSrcIter, dictCameraCalVals):
    
    imageLines = [CImageLine("LEFT"),CImageLine("RIGHT")]
    for (imgRaw, imgRawName) in rawImgSrcIter:
        print("P2Main.ImgPipeProc: ", imgRawName)
        imageRecsCurFrame = []
        imageRecsCurFrame.append( (imgRaw, imgRawName) ) 
        
        imageRecsPreProc = ImageProc_PreProcPipeline(imgRaw, dictCameraCalVals)
        imageRecsCurFrame.extend(imageRecsPreProc)
        imgLineFindSrc = imageRecsPreProc[-1][0]

        for curImageLine in imageLines :
            print("P2Main.LineProc: ", curImageLine.name)
            polyCoeffNew, imgSelection = curImageLine.CalcPolyFit(imgLineFindSrc)
            imageRecsCurFrame.append( (imgSelection, "imgSelection_" + curImageLine.name ))       
            
        print(" ")

        doDebugImages = True
        if doDebugImages :
            get_ipython().run_line_magic('matplotlib', 'qt4')
            PlotImageRecords(imageRecsCurFrame)
            key = cv2.waitKey(2000)


#============================= Main Invocation =================
g_cameraDistortionCalValsFileName = "CameraDistortionCalVals.pypickle"
g_cameraPerspectiveWarpMatrixFileName = "CameraPerspectiveWarpMatrix.pypickle"

g_dictCameraCalVals = GetCameraCalVals(g_cameraDistortionCalValsFileName, g_cameraPerspectiveWarpMatrixFileName)
g_imageIter = GetImageIteratorFromDir()

#================= P2Main() invocation
P2Main(g_imageIter, g_dictCameraCalVals)


# In[ ]:


pipelineImages = [
    (imgRaw, imgRawName),
    (imgPipeProc, "imgPipeProc", {"cmap":"gray"}),
    (imgSelection, "imgSelection", ),      

    #(imgHthr, "imgHthr", {"cmap":"gray"}),
    #(imgLthr, "imgLthr", {"cmap":"gray"}),
    #(imgSthr, "imgSthr", {"cmap":"gray"}),      

    #(imgIn, imgInFileName),
    #(imgHSL, "imgHSL"),

    #(imgSobelThrGradX, "imgSobelThrGradX", {"cmap":"gray"}),
    #(imgSobelThrGradY, "imgSobelThrGradY", {"cmap":"gray"}),
    #(imgSobelThrMag, "imgSobelThrMag", {"cmap":"gray"}),
    #(imgSobelThrDir, "imgSobelThrDir", {"cmap":"gray"}),

    #(imgComboSobel, "imgComboSobel", {"cmap":"gray"}),
    #(imgComboSatSobelX_Stack, "imgComboSatSobelX_Stack", {"cmap":"gray"}),
    #(imgComboSatSobelX_Or, "imgComboSatSobelX_Or", {"cmap":"gray"}), 
]

