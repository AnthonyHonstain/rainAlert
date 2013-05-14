import numpy as np
import cv2
import cv2.cv as cv
from recognizer import Recognizer
from learningRec import SVM



class UmbrellaTracker:
    def __init__(self): 
        self.init = False

        #Are we paused
        self.pause = False
        #training mode enabled 
        self.train = False
        self.tracking = False
        self.useMog = True
        self.showBGDiff = False
        #seperate bg weight ratio's
        self.weightBG1 = 0.2
        self.weightBG2 = 0.6
        self.tracked = []

        self.trained = {}
        self.counter = 0
        self.trainCouter = 0
        self.erodeIter = 1
        #gausion weight ratio
        self.blurKp = 1
        #cutoff threshold for BW image
        self.cutOffThresh=30;
        #what size to limit our bounding boxes too
        self.sizeL = 4500
        self.sizeM = 1000
        #kernal size for erode and dilate
        self.kernalH = 3
        self.kernalW = 3
        self.kernel = np.ones((self.kernalH,self.kernalW),'uint8')
        #kernal size
        self.rec = Recognizer()
        self.currFrame = None
        self.currFrameOpt = None

        self.contours = []
        #tracking will start after this many frames
        self.start = 10
        #track interval don't do tracking every frame
        self.track_interval = 5

        #Use MOG BG extractor
        self.bgs = cv2.BackgroundSubtractorMOG(24*60, 1
                                               , 0.8, 0.5)
        self.svmTracker = SVM()
        self.svmReady = True


    def onMouseClick(self,event, x, y, flags, param ):
        #select which roi to train for features
        if event == cv.CV_EVENT_LBUTTONDOWN:
            if len(self.contours) > 0:
                for cont in self.contours:
                     if self.isWithinBB(x,y,cont):
                        #x,y,w,h = cv2.boundingRect(cont)
                        box = cv2.boundingRect(cont)
                        if self.train:
                            print "Trainig for umbrella"
                            name = "./train/pos/%s.png"%self.trainCouter
                            self.trained[name] = box
                            cv2.imwrite("./train/pos/%s.png"%self.trainCouter,self.currFrame)
                            #self.rec.createTargetFeatures(self.getROI(self.currFrame,box),box)
                            #self.svmTracker.train(self.getROI(self.currFrame,box),np.float32)
                            #self.svmReady = True
                        else:
                            print "Trainig for None umbrella"
                            name = "./train/neg/%s.png"%self.trainCouter
                            self.trained[name] = box
                            cv2.imwrite("./train/neg/%s.png"%self.trainCouter,self.currFrame)

                        self.trainCouter = self.trainCouter + 1 
                               
            else:
                print "wait for contours to get initialized"

    def isWithinBB(self,x,y,cont):
        #check weather a point clicked on screen is within a bounding box defined by contours
        xb,yb,wb,hb = cv2.boundingRect(cont)
        xbC = xb + wb
        ybC = yb + hb
        if x > xb and x <xbC:
            if y > yb and y < ybC:
                return True
        return False

    def getROI(self,image,box):
        #cut out and return the section from an image extraceted from a contour
        if len(box) == 4:
            #make sure we have for courners seems sometimes we don'e #paranoid
            x,y,w,h = box
            return image[y:y+h,x:x+w]
        return None

    #get Video stream
    def addGausianBlur(self,f1,level):
        return cv2.GaussianBlur(f1,(level,level),1)

    #get diff 
    def subtractFrames(self,f1,f2):
        dif = cv2.absdiff(f1,f2)
        #dif = cv2.bitwise_and(f1,f2)
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
    def drawContours(self,image,conts,sizeL,sizeM,color):
        for cnt in conts:   
            x,y,w,h = cv2.boundingRect(cnt)
            area = float(w)*float(h)
            if area < sizeL and area > sizeM:
                cv2.rectangle(image,(x,y),(x+w,y+h), color,2)
        return image
       
    #draw contors on image
    def drawContours2(self,image,conts,sizeL,sizeM,color):
        for cnt in conts:   
            x,y,w,h = cnt[0],cnt[1],cnt[2],cnt[3]
            #area = float(w)*float(h)
            #if area < sizeL and area > sizeM:
            cv2.rectangle(image,(x,y),(x+w,y+h), color,2)
        return image

    def drawBox(self,image,x,y,size,color):
       cv2.rectangle(image,(x,y),(x+size,y+size),color)
       return image
    #this really slows things down!
    def makeBW(self,f):
        bwImg = cv2.cvtColor(f,cv.CV_RGB2GRAY)
        return bwImg

    def createTrainigOutput(self):
        count = 1
        fpP = open("info.dat","w")
        fpN = open("bg.txt","w")
        for key in self.trained.keys():
                if key.find("neg") > 0:
                    output = "%s \n"%(key)
                    fpN.write(output)
                else:
                    output = "%s  %s  %s %s %s %s \n"%(key,count,self.trained[key][0],self.trained[key][1],self.trained[key][2],self.trained[key][3])
                    fpP.write(output)
        fpP.close()
        fpN.close()
    #Home Brew BG removal
    def bgSmoothing(self,frameBG,arr1, arr2):
 
        frameBG = cv2.equalizeHist(frameBG)
        #blurring helps to normalize and smooth out noise
        arr1 = self.addGausianBlur(arr1,self.blurKp)
        arr2 = self.addGausianBlur(arr2,self.blurKp)
        #add to wighted averages of the background in order to obtained weighterd average to remove noise
        cv2.accumulateWeighted(frameBG,arr1,self.weightBG1)
        cv2.accumulateWeighted(frameBG,arr2,self.weightBG2)
        #normalize
        res1 = cv2.convertScaleAbs(arr1)
        res2 = cv2.convertScaleAbs(arr2)
        
        #res1 = self.makeBW(res1)
        #res2 = self.makeBW(res2)
        return res1,res2

    def bgSmooothing2(self,frame):
        #OpenCV BG removal using MOG
         #frameBW = self.makeBW(frame)
         frame = self.addGausianBlur(frame,self.blurKp)
         frame = cv2.equalizeHist(frame)       
         fgMask = self.bgs.apply(frame)
         return fgMask

    def track(self,f,vid):

        f = self.makeBW(f)
        #create numpy arrays from image frames
        avg1 = np.float32(f)
        avg2 = np.float32(f)
        objs = []
        while 1:
            if not self.pause:
                #get frame from video
                _,f = vid.read()

                f = self.makeBW(f)
                res2 = f.copy()
                #make BW first

                if self.useMog:
                    mask = self.bgSmooothing2(res2)
                else:
                    res1,res2 = self.bgSmoothing(f,avg1,avg2)
                    #get diff
                    mask = self.subtractFrames(res1,res2)
                    #lets threshold the image
                    _,mask = self.getThreshold(mask,self.cutOffThresh)

                #Dilate and erode
                #res3 = cv2.dilate(res3,kernel,iterations=3)
                #res3 = cv2.erode(res3,kernel,iterations=2)
                res3 = cv2.erode(mask,self.kernel,iterations=self.erodeIter)
            
                #set this for later use
                self.currFrame = res2
                self.currFrameOpt = res3

                cimage = np.copy(res3)
                #find countours
                if self.counter > self.start:
                    self.contours  = self.findCountoursBW(cimage)

                #make it color again
                #lets do some interesting stuff
                if len(self.contours) > 0 :
                    for cont in self.contours:
                        if self.counter % self.track_interval == 0:
                            box = self.getROI(res2,cv2.boundingRect(cont))
                            #self.rec.findUmbrellas(box)
                            #dont search every frame
                            if self.tracking and self.counter % self.track_interval == 0:
                                for thing in self.contours:
                                    objs = self.rec.detectUmbrellaCascade(box)
                                    umb = thing


                if len(objs) > 0:
                    self.tracked = objs

                res2 = cv2.cvtColor(res2,cv2.COLOR_GRAY2BGR)
                if len(self.tracked) > 0:
                    res2 = self.drawContours2(res2,umb,self.sizeL,self.sizeM,(255,0,0))

                res2 = self.drawContours(res2,self.contours,self.sizeL,self.sizeM,(255,255,0))
                #res3 = cv2.dilate(res3,kernel)
                self.counter = self.counter + 1
                #cv2.imwrite("./pngs/image-"+str(counter).zfill(5)+".png", mask)
                #self.rec.getFeatures(res2,2)
                #res1 = self.rec.drawFeatures(res2)

                if len(self.rec.toTrack) > 0 and self.svmReady:
                        for person in self.contours:
                            #masks = self.rec.flannMatching(cv2.boundingRect(person))
                            matches = self.rec.trackObjects(res2)
                            #res3 = self.drawBox()
            if self.showBGDiff:
                cv2.imshow('bg1',res3)
            cv2.imshow('bg2',res2)
            if not self.init:
                cv2.setMouseCallback('bg2',self.onMouseClick,None);
                init = True

            #break if esc is hit
            k = cv2.waitKey(20) 
            if k == ord('t'):
                if self.train:
                    print "training for Negative samples"
                    self.train = False
                else:
                    print "training for positive samples"
                    self.train = True
            if k == ord('d'):
                if len(self.trained.keys()):
                    print "creating training output file.."
                    self.createTrainigOutput()
                    print "output training files ready"
                    self.trained = {}
                    self.trainCouter = 0
            if k == ord('r'):
                self.rec.reset()
            if k == ord(' '):
                print "paused"
                self.pause = not self.pause
            if k == ord('g'):
                self.tracking = not self.tracking
                print "Tracking:%s"%self.tracking
            if k == ord('b'):
                self.useMog = not self.useMog
                print "MOG bg extraction %s"%self.useMog
            if k == ord('x'):
                self.showBGDiff = not self.showBGDiff
                print "enable bg diff:%s"%self.showBGDiff
            if k == ord('+'):
                self.erodeIter = self.erodeIter + 1
                print "erode : %s"%self.erodeIter
            if k == ord('-') and self.erodeIter > 0:
                self.erodeIter = self.erodeIter -1
                print "erode : %s"%self.erodeIter
            if k == 27:
                break

def mainLoop():
        vid = cv2.VideoCapture("./data/test3.mp4")
        _,f = vid.read()
        print "Press -t  to begin training , select contours"
        print "Press -d  when done training to output cascade files"
        print "Press -g  to begin tracking based on built cascade tracker"
        print "Press -b  to switch between custom BG extraction vs MOG"
        print "Press -x  to switch show raw BG extraction"
        print "Press -space to pause if you wish to select multiple object"
        print "Press +(plus)  to increase erosion coefficient"
        print "Press -(minus) to decrease erosion coefficient"

        tracker = UmbrellaTracker()
        tracker.track(f,vid)

if __name__ == '__main__':
  print "starting application"
  mainLoop()
        