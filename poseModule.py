import cv2 
import mediapipe as mp
import math
import statistics

class poseDetector:
    
    def __init__(self, mode=False, model_complexity=1, smooth=True, enable_segmentation=False, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth, self.enable_segmentation, self.smooth_segmentation, self.detectionCon, self.trackCon)
        
    def getPose(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if draw:
            if self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(imgRGB, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
        
    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for i, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([i, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        self.lmList = lmList
        return lmList
    
    def getLengths(self, img):
        
        self.joints = {"shoulder": [11, 12],
                  "hip": [23, 24],
                  "left_humerus": [12, 14],
                  "right_humerus": [11, 13],
                  "left_radius": [13, 15],
                  "right_radius": [14, 16],
                  "left_femur": [24, 26],
                  "right_femur": [23, 25],
                  "left_tibia": [26, 28],
                  "right_tibia": [25, 27],
                 }
        
        lengths = {}
        for key, val in self.joints.items():
            lengths[key] = self.distance(self.lmList[val[0]][1:], self.lmList[val[1]][1:])
            
        # calculate torso using midpoint of shoulder and hip
        midpoint_shoulder = self.midpoint(self.lmList[self.joints["shoulder"][0]][1:], self.lmList[self.joints["shoulder"][1]][1:])
        midpoint_hip = self.midpoint(self.lmList[self.joints["hip"][0]][1:], self.lmList[self.joints["hip"][1]][1:])
        lengths["torso"] = self.distance(midpoint_shoulder, midpoint_hip)
        self.lengths = lengths
        return lengths
            
    def distance(self, p1, p2):
        return math.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    
    def midpoint(self, p1, p2):
        return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
    
    def getHeadBox(self, img, draw=True):
        
        # And both years as width of face
        width = self.distance(self.lmList[7][1:], self.lmList[8][1:])
        # The ratio of width-to-height is approximately two third
        height = width*(3/2)
        
        midpoint = self.midpoint(self.lmList[7][1:], self.lmList[8][1:])
        
        # head
        pt1 = (int(midpoint[0]-(width/2)), int(midpoint[1]-(height/2)))
        pt2 = (int(midpoint[0]+(width/2)), int(midpoint[1]+(height/2)))
        
        self.top_head = pt2

        if draw:
            cv2.rectangle(img, pt1=pt1, pt2=pt2, color=(0,0,255), thickness=10)
        
        return img
    
    def getProportions(self, img, draw=True):
        
        # get bottom most y-position
        bottom_point = statistics.mean([self.lmList[29][2], self.lmList[30][2], self.lmList[31][2], self.lmList[32][2]])
        height = abs(self.top_head[1] - bottom_point)
        
        self.proportions = { key: round(val/height, 3) for key, val in self.lengths.items()}
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        if draw:
            for key, val in self.proportions.items():
                if key in self.joints:
                    i, j = self.joints[key]
                    midpoint = self.midpoint(self.lmList[i][1:], self.lmList[j][1:])
                    cv2.putText(img, str(val), (int(midpoint[0]), int(midpoint[1])), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img
        
    
def main():
    
    detector = poseDetector()
    
#     cap = cv2.VideoCapture("images/lebron.webp") # video capture source camera (Here webcam of laptop) 
#     ret, frame = cap.read() # return a single frame in variable `frame`
    
    file = "images/lebron.webp"
    img = cv2.imread(file)
    while True:
        img = detector.getPose(img)
        lmList = detector.getPosition(img)
        print(lmList)
        lengths = detector.getLengths(img)
        print(lengths)
        img = detector.getHeadBox(img)
        
        img = detector.getProportions(img)
        cv2.imshow('img', img) #display the captured image
        #if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
        cv2.imwrite('images/c1.png', img)
        cv2.destroyAllWindows()
        break
    print(type(img))
    #cap.release()
    
    
if __name__ == "__main__":
    main()