import cv2
import math
import imutils
import numpy as np
import random
from tqdm import tqdm

import autolib

class car():

    def __init__(self):
        bgr = cv2.imread('map\\map6.png')
        #bgr = cv2.resize(bgr, (int(bgr.shape[1]//1.5), int(bgr.shape[0]//1.5))) 
        self.envmap = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        self.spawn = 800, 720
        self.y, self.x = self.spawn

        self.texture = self.texture(bgr)

        self.h, self.w = self.envmap.shape[:2]

        self.resized = cv2.resize(self.texture,(self.w//2, self.h//2))
        self.visu = np.copy(self.resized)

        self.totangle = 0
        self.dist = 0
        self.angle = 0

        self.memory = []
        self.it = 0
        
        self.py, self.px = self.y, self.x
        self.vector = np.array([0,1])

        
    def check(self):
        
        # update total angle+dist to check if agent made a half turn

        degree = self.angle*180/math.pi
        self.dist += self.speed 
        self.totangle += degree

    
        if self.envmap[int(self.y)][int(self.x)]>=200:
                    return True, True

        elif int(self.px) - int(self.x) == 0:

            I1 = [min(int(self.py),int(self.y)), max(int(self.py),int(self.y))]

            for y in range(I1[0]-2, I1[1]+2):

                if self.envmap[y][int(self.x)]>= 200 or self.envmap[y][int(self.x)+1]>= 200 or self.envmap[y][int(self.x)-1]>= 200:
                    return True, True

            return False, True
            
        else:

            I1 = [min(int(self.px),int(self.x)), max(int(self.px),int(self.x))]

            A1 = (self.py-self.y)/(self.px-self.x)
            b1 = self.py - (A1 * self.px)

            for x in range(I1[0]-1, I1[1]+1):
                y = (A1*x)+b1

                if self.envmap[int(y)][x]>= 200 or self.envmap[int(y)-1][x]>= 200 or self.envmap[int(y)+1][x]>= 200:
                    return True, True

            return False, True



        if self.totangle>=180 and self.dist<200 or self.totangle<=-180 and self.dist<200:

            self.totangle = 0
            self.dist = 0
            return True, False
            
        return False, True


    def reward(self, done, half):        

        if half == False:
            return -math.pi*30/2

        elif done == True:
            return -150

        elif self.angle != 0:
            return self.speed/2
    
        else:
            return self.speed


    def get_pos(self, action):

        self.action = action

        if self.action == 0:
            self.angle = -self.turnangle
        elif self.action == 1:
            self.angle = 0
        elif self.action == 2:
            self.angle = self.turnangle

        
        ########### calculation ###########
            

        rot_mat = np.array([[math.cos(self.angle), -math.sin(self.angle)],[math.sin(self.angle), math.cos(self.angle)]])

        self.vector = self.vector/math.sqrt((self.vector[0]**2)+(self.vector[1]**2))
        self.vector = self.vector.dot(rot_mat)
        self.vector = self.vector.dot(self.speed)

        self.px = self.x
        self.py = self.y

        self.x += self.vector[0]
        self.y += self.vector[1]

        self.it += 1

        

    def get_view(self):
        
        if self.vector[0] == 0 :
            angle = math.atan(self.vector[1]/0.001)
        else:
            angle = math.atan(self.vector[1]/self.vector[0])

        degree = angle*180/math.pi

        rotated = imutils.rotate(self.texture, degree, center=(int(self.x), int(self.y)))

        
        try:
            if self.vector[0] >= 0 and self.vector[1] >= 0:
                ori = 0
            elif self.vector[0] >= 0 and self.vector[1] <= 0:
                ori = 0
            else:
                ori = -1

            if ori == -1 :
                self.temp = rotated[int(self.y)-300:int(self.y)+300, int(self.x)-300:int(self.x)]
                self.temp = cv2.flip(self.temp, ori)

            else :
                self.temp = rotated[int(self.y)-300:int(self.y)+300, int(self.x):int(self.x)+300]
            
            self.temp = imutils.rotate_bound(self.temp, -90)
            

        except:
            print('failed to get temp')
            pass

        
        self.map_3D() #create perspective + resize/process image
        
        self.state = np.expand_dims(self.state, axis=0) 

        return self.state

    
    ######### graphics and texture functions #########


    def map_3D(self):
        h,w,ch = self.temp.shape

        angle = 0.13
        coef = math.sin(angle)
        y = h
        xmi = 220
        xmx = 380

        pts1 = np.float32([[0, 0],[0, y],[w, 0],[w, y]])
        pts2 = np.float32([[xmi, 0],[0, coef*y],[xmx, 0],[w, coef*y]])


        M = cv2.getPerspectiveTransform(pts1,pts2)

        self.state = cv2.warpPerspective(self.temp,M,(w,h))
        
        self.state = self.state[1:int(coef*h) , xmi:xmx]

        self.state = cv2.resize(self.state,(self.state.shape[1]//2,self.state.shape[0]))

        self.state = self.state[:, 10:self.state.shape[1]-10]


        
    def texture(self, rgb_img):
        
        img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

        road = cv2.imread('back\\2.jpg')
        road = cv2.resize(road,(img.shape[1],img.shape[0]))

        background = cv2.imread('back\\1.jpg')
        background = cv2.resize(background,(img.shape[1],img.shape[0]))

        inter = cv2.bitwise_and(road, road, mask = img)
        inter2 = cv2.bitwise_and(background, background, mask = img)

        new_img = inter2-inter+road
        
        
        ##########

        for x in range(len(rgb_img)):
            for y in range(len(rgb_img[x])):
                
                if rgb_img[x][y][0]==255: #draw white lines with a bit of random for the color
                    
                    val = random.randint(30,100)
                    new_img[x][y][0]=255-val
                    new_img[x][y][1]=255-val
                    new_img[x][y][2]=255-val
                    

                elif rgb_img[x][y][2]==170: #draw yellow lines with a bit of random for the color

                    val = random.randint(0,30)
                    new_img[x][y][0]=30-val
                    new_img[x][y][1]=210-val
                    new_img[x][y][2]=210-val

        ##########

        brght_img = self.rand_brght(new_img, b=100, mid = 100) # add some random brightness/darkness to map
        spot_img = self.rand_light(brght_img, 100,300, 8, color=40)

        return spot_img


    def rand_brght(self,img, b=50, mid = 25):
        a = 0
        rand = random.randint(a, b)

        if rand < mid:
            sign = False
        else:
            sign = True

        img = self.change_brightness(img, value=rand, sign= sign)

        return img

    def change_brightness(self, img, value=30, sign=True): 

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if sign == True: # increase brightness
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value

        if sign == False: # decrease brightness
            lim = 0 + value
            v[v < lim] = 0 #to avoid pixel to be < 0 
            v[v >= lim] -= value

        hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img

    def rand_light(self,img, a, b, nb, color=100):
        for _ in tqdm(range(nb)):
            radius = random.randint(a,b)

            cx = random.randint(300+radius,img.shape[1]-300-radius) #set max-min values for x
            cy = random.randint(400+radius,img.shape[0]-400-radius) #set max-min values for y

            img = self.spot_light(img,cx,cy, radius = radius , color=color)

        return img

    def spot_light(self, img, cx,cy, radius = 50, color = 200): # color = orange 
        
        minix = cx-radius
        miniy = cy-radius
        sqradius= radius**2
        

        for x in range(minix,minix+2*radius-1):
            for y in range(miniy,miniy+2*radius-1):
                dsqr = (cx-x)**2+(cy-y)**2
                if dsqr<sqradius:
                    dif = sqradius-dsqr
                    val = int(dif*color/sqradius)

                    lim = 255 - val
                    img[x][y][img[x][y] > lim] = 255
                    img[x][y][img[x][y] <= lim] += val

        return img

    ##################################################

    def render(self):
        
        cv2.circle(self.visu, (int(self.x)//2,int(self.y)//2), 2, [0,0,255], thickness = -1)

        cv2.imshow('visu', self.visu)
        cv2.imshow('state', self.state[0])
        cv2.waitKey(1)



    
    def step(self, action):
        self.get_pos(action)
        done, half = self.check()
        reward = self.reward(done,half)
        self.get_view()
        
        return self.state, reward, done
       
    
    def reset(self):
        self.y, self.x = self.spawn
        self.vector = np.array([0, 1])
        self.angle = 0
        self.visu = np.copy(self.resized)
        

        
