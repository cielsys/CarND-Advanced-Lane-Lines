
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import math
import os

from moviepy.editor import VideoFileClip
from IPython.display import HTML
import datetime


# In[8]:


get_ipython().run_cell_magic('HTML', '', '<style> code {background-color : orange !important;} </style>\nfrom IPython.core.display import display, HTML\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# In[10]:


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


# In[18]:


#calFileInNames = glob.glob('camera_cal/cal*.jpg')
#calFileInNames = ['camera_cal/calibration2.jpg']

def GetImageIteratorFromDir():
    dirImagesIn = "ImagesIn/TestImagesIn/POVRaw/"
    fileNameBase = "straight_lines1.jpg"

    imgInFileNames = [dirImagesIn + "straight_lines1.jpg", dirImagesIn + "straight_lines2.jpg"]
    imageIter = ((mpimg.imread(imgInFileName), imgInFileName) for imgInFileName in imgInFileNames)

    return(imageIter)


# In[22]:


def ImageProc_Pipeline(imgRaw):
    imgPipeProc = imgRaw
    return imgPipeProc
    


# In[24]:



def P2Main(rawImgSrcIter):
    for (imgRaw, imgRawName) in rawImgSrcIter:
        print("P2MainProcessing: ", imgRawName)
        
        imgPipeProc = ImageProc_Pipeline(imgRaw)

        
# MainInvocation
g_imageIter = GetImageIteratorFromDir()
P2Main(g_imageIter)

