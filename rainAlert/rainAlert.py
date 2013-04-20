import numpy as np
import cv2
import cv


#get Video stream
def addGausianBlur(f1,level):
    return cv2.GaussianBlur(f1,(level,level),0)

#get diff 
def subtractFrames(f1,f2):
    dif = cv2.absdiff(f1,f2)
    return dif

#get image threshold to remove some noise
def getThreshold(f,cutOffThreshVal):
    return cv2.threshold(f,cutOffThreshVal,255,0)

#find the contours in the image
def findCountoursBW(f):
    conts,hier = cv2.findContours(f,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #check size of countours ignore ones that are too small or large
    for cnt in conts:   
        x,y,w,h = cv2.boundingRect(cnt)
        if float(w)*float(h) < 60:
            cv2.rectangle(f,(x,y),(x+w,y+h),(0,255,0),2)
    return f

#this really slows things down!
def makeBW(f):
    bwImg = cv2.cvtColor(f,cv.CV_RGB2GRAY)
    return bwImg

def mainLoop():

    #if no params are set use default imagery 
    vid = cv2.VideoCapture("./data/test.mp4")
    weightBG1 = 0.01
    weightBG2 = 0.6
    blurKp = 1
    cutOffThresh=70;

    _,f = vid.read()

    #create numpy arrays from image frames
    avg1 = np.float32(f)
    avg2 = np.float32(f)

    while 1:
        #get frame from video
        _,f = vid.read()

        cv2.accumulateWeighted(f,avg1,weightBG1)
        cv2.accumulateWeighted(f,avg2,weightBG2)

        #normalize
        res1 = cv2.convertScaleAbs(avg1)
        res2 = cv2.convertScaleAbs(avg2)
 
        res1 = addGausianBlur(res1,blurKp)
        res2 = addGausianBlur(res2,blurKp)
    
        #get diff
        res3 = subtractFrames(res1,res2)

        #lets threshold the image
        _,res3 = getThreshold(res3,cutOffThresh)

        #make BW first
        res3 = makeBW(res3)
        #find countours
        res3 = findCountoursBW(res3)

        cv2.imshow('bg1',res1)
        cv2.imshow('bg2',res3)

        #break if esc is hit
        k = cv2.waitKey(20)
        if k == 27:
            break


if __name__ == '__main__':
    print "starting application"
    mainLoop()
    