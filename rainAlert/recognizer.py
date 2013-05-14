import cv2
import cv2.cv as cv
import numpy as np


from collections import namedtuple


PlanarTarget = namedtuple('PlaneTarget', 'image, rect, keypoints, descrs, data')

TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, quad')

MIN_MATCH_COUNT = 5
USE_SURF_FT = 1
USE_ORB_FT = 2
USE_SIFT_FT = 3


class Recognizer(object):
    """recognizer class used to identidy people with umbrellas"""

    def __init__(self,maxCorners = 10,qLevel = 0.02, minDist = 50, surfHthreshold = 300):
       self.maxCorners = maxCorners
       self.qLevel = qLevel
       self.minDist = minDist
       self.features = None
       self.toTrack = []

       #SURF Features
       self.surf = None
       self.surfDescrp = None
       self.kp = None
       self.descriptors = None
       self.KN = cv2.KNearest()
       
       #ORB Features
       self.orbF = cv2.ORB( nfeatures = 1000 )

       #Cascade classifier
       self.cascade = cv2.CascadeClassifier("./data/umb.xml")

       #FLANN PARAMS
       self.r_threshold = 0.6
       FLANN_INDEX_KDTREE = 1
       self.FLANN_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
       self.FlannMatcher = cv2.FlannBasedMatcher(self.FLANN_params,{})
       self.FlannMatcher2 = None
       

    def reset(self):
        #wipeout everything #newbeginnig
        self.targets = []
        self.matcher.clear()

    def flannMatching(self,image):
        #Do FLANN based matching on sent in discriptors
        #kps, desc2 = self.surf.detect(image,None)
        idx2, dist = self.FlannMatcher.knnSearch(desc2, 2, params = {})
        mask = dist[:,0] / dist[:,1] < self.r_threshold
        idx1 = np.arange(len(desc2))
        pairs = np.int32( zip(idx1, idx2[:,0]) )
        return pairs[mask]

    def trackObjects(self,frame):
        #Track objects in frame based on targets trained
        frame_points, frame_descrs = self.findFeatures(frame,USE_SURF_FT)
        if len(frame_points) < MIN_MATCH_COUNT:
            return []
        
        #matches = self.FlannMatcher.knnMatch(self.frame_descrs,2)
        matches = self.FlannMatcher2.knnSearch(frame_descrs,2, params={})
        #hard coded distance
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        if len(matches) < MIN_MATCH_COUNT:
            return []
        return matches


    def detectUmbrellaCascade(self,image):
        objects = self.cascade.detectMultiScale(  image,
                                                  scaleFactor=1.3,
                                                  minNeighbors=4,
                                                  minSize=(15, 15),
                                                  flags = cv.CV_HAAR_SCALE_IMAGE)
        return objects


    def createTargetFeatures(self,image,rect):
       #extract features using suf and train kmeans matcher, run this on every frame
       #kps = self.surf.detect(image)
       #self.FLANN = cv2.flann_Index(kps, self.FLANN_params)
       kp,des = self.findFeatures(image,USE_SIFT_FT)
       points, descs = [], []
       descs = np.uint8(descs)
       self.FlannMatcher.add([descs])
       self.FlannMatcher2 = cv2.flann_Index(des,self.FLANN_params)
       track = PlanarTarget(image = image, rect=rect, keypoints = kp, descrs=descs, data=None)
       self.toTrack.append(track)
             
    def findFeatures(self,image,type):
       kp = []
       des = []
       if type == USE_ORB_FT:
           kp,des = self.orbF.detectAndCompute(image,None)
       else:
           self.surf = cv2.FeatureDetector_create("SURF")
           self.surfDescrp = cv2.DescriptorExtractor_create("SURF")
           kp = self.surf.detect(image)
           kp, des = self.surfDescrp.compute(image,kp)

       return kp,des

    def extractMatchSurfFeaturesBG(self,image):
       #extract features using suf and train kmeans matcher, you need to run BG matching prior, run this once on image which we wish to match
       kps,descs = self.surf.detect(image,None)
       #enumerate on descriptors
       for h,des in enumerate(descs):
           #not sure what's going on here proir to running kmeanse similar chack using FLANN
           des = np.array(des,np.float32).reshape((1,128))
           retval, results, neigh_resp, dists = knn.find_nearest(des,1)
           res,dist =  int(results[0][0]),dists[0][0]
           x,y = kp[res].pt
           center = (int(x),int(y))
           cv2.circle(img,center,2,color,-1)
       return image

    def getFeatures(self,image,type):
       self.features = cv2.goodFeaturesToTrack(image,self.maxCorners,self.qLevel,self.minDist)
       self.features = self.features.reshape((-1,2))
       
    def findUmbrellas(self,roi):
        circles = cv2.HoughCircles(roi,cv.CV_HOUGH_GRADIENT,dp=1,minDist =1,)
        if circles:
            for circle in circles:
                print "foundCircle"

    def drawFeatures(self,image):
       #draw features for extracted using gftt
       if len(self.features) > 0:
           for x, y in self.features:
                cv2.circle(image, (x, y), 10, (0, 0, 255))
       else:
            print "run get features first"
            return image
       return image

        #do something here match on roi of image features
        


