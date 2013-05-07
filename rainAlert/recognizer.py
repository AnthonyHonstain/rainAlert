import cv2
import cv2.cv as cv
import numpy as np

class Recognizer(object):
    """recognizer class used to identidy people with umbrellas"""


    def __init__(self,maxCorners = 10,qLevel = 0.02, minDist = 50, surfHthreshold = 300):
       self.maxCorners = maxCorners
       self.qLevel = qLevel
       self.minDist = minDist
       self.features = None
       #
       self.surf = cv2.SURF(surfHthreshold)
       self.kp = None
       self.KN = cv2.KNearest()

    #Extract features from BB
    def extractSurfFeatures(self,image,roi):
       self.kp = self.surf.detect(image,None,userProvidedKeypoints = false)

    def getFeatures(self,image):
       #trainCriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
       self.features = cv2.goodFeaturesToTrack(image,self.maxCorners,self.qLevel,self.minDist)
       self.features = self.features.reshape((-1,2))
       

    def drawFeatures(self,image):
       if len(self.features) > 0:
           for x, y in self.features:
                cv2.circle(image, (x, y), 10, (0, 0, 255))
       else:
            print "run get features first"
            return image
       return image


    def detectObjects(self,image):
        print "ready to detect"
        #do something here match on roi of image features
        


