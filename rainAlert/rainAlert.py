import numpy as np
import cv2
import cv2.cv as cv
from Recognizer import Recognizer


class UmbrellaTracker:
    def __init__(self): 
        self.init = False
        #seperate bg weight ratio's
        self.weightBG1 = 0.01
        self.weightBG2 = 0.3
        #gausion weight ratio
        self.blurKp = 3
        #cutoff threshold for BW image
        self.cutOffThresh=35;
        #what size to limit our bounding boxes too
        self.sizeL = 4500
        self.sizeM = 1500
        #kernal size for erode and dilate
        self.kernalH = 3
        self.kernalW = 3
        #kernal size
        self.rec = Recognizer()
        self.currFrame = None
        self.contours = []
       
    def onMouseClick(self,event, x, y, flags, param ):
        #select which roi to train for features
        if event == cv.CV_EVENT_LBUTTONDOWN:
            if len(self.contours) > 0:
                for cont in self.contours:
                     if self.isWithinBB(x,y,cont):
                        #x,y,w,h = cv2.boundingRect(cont)
                        print "Trainig for umbrella"
                        self.rec.extractSurfFeatures(self.currFrame,cv2.boundingRect(cont))
            else:
                print "wait for contours to get initialized"

    #check weather a point clicked on screen is within a bounding box defined by contours
    def isWithinBB(self,x,y,cont):
        xb,yb,wb,hb = cv2.boundingRect(cont)
        xbC = xb + wb
        ybC = yb + hb
        if x > xb and x <xbC:
            if y > yb and y < ybC:
                return True
        return False


    #get Video stream
    def addGausianBlur(self,f1,level):
        return cv2.GaussianBlur(f1,(level,level),1)

    #get diff 
    def subtractFrames(self,f1,f2):
        dif = cv2.absdiff(f1,f2)
        return dif

    #get image threshold to remove some noise
    def getThreshold(self,f,cutOffThreshVal):
        return cv2.threshold(f,cutOffThreshVal,255,0)

    #find the contours in the image
    def findCountoursBW(self,f):
        conts,hier = cv2.findContours(f,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #check size of countours ignore ones that are too small or large
        return conts

    #draw contors on image
    def drawContours(self,image,conts,sizeL,sizeM):
        for cnt in conts:   
            x,y,w,h = cv2.boundingRect(cnt)
            area = float(w)*float(h)
            if area < sizeL and area > sizeM:
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),2)
        return image

    #this really slows things down!
    def makeBW(self,f):
        bwImg = cv2.cvtColor(f,cv.CV_RGB2GRAY)
        return bwImg


    def track(self,f,vid):


        #create numpy arrays from image frames
        avg1 = np.float32(f)
        avg2 = np.float32(f)
        while 1:
            #get frame from video
            _,f = vid.read()

            cv2.accumulateWeighted(f,avg1,self.weightBG1)
            cv2.accumulateWeighted(f,avg2,self.weightBG2)

            #normalize
            res1 = cv2.convertScaleAbs(avg1)
            res2 = cv2.convertScaleAbs(avg2)
     
            res1 = self.addGausianBlur(res1,self.blurKp)
            res2 = self.addGausianBlur(res2,self.blurKp)
        
            #get diff
            res3 = self.subtractFrames(res1,res2)

            #lets threshold the image
            _,res3 = self.getThreshold(res3,self.cutOffThresh)

            #make BW first
            res3 = self.makeBW(res3)

            #Dilate and erode
            kernel = np.ones((self.kernalH,self.kernalW),'uint8')
            res3 = cv2.dilate(res3,kernel,iterations=3)
            res3 = cv2.erode(res3,kernel,iterations=4)
            
            self.currFrame = res2

            cimage = np.copy(res3)
            #find countours
            self.contours  = self.findCountoursBW(cimage)

            res2 = self.drawContours(res2,self.contours,self.sizeL,self.sizeM)
            #res3 = cv2.dilate(res3,kernel)
            

            self.rec.getFeatures(res3)
            res1 = self.rec.drawFeatures(res1)

            cv2.imshow('bg1',res1)
            cv2.imshow('bg2',res2)
            if not self.init:
                cv2.setMouseCallback('bg2',self.onMouseClick,None);
                init = True

            #break if esc is hit
            k = cv2.waitKey(20) 
            if k == 27:
                break

def mainLoop():
        vid = cv2.VideoCapture("./data/test2.mp4")
        _,f = vid.read()
        tracker = UmbrellaTracker()
        tracker.track(f,vid)

if __name__ == '__main__':
  print "starting application"
  mainLoop()
        