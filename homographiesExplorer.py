#! /usr/bin/env python3
#encoding: utf-8


import numpy as np
import cv2 
import sys
from myAxes import Axes

argc = len(sys.argv)

# Abre el flujo en donde vienen 
#if argc < 2:
#   print("Es necesario pasa un parámetro")
#   exit(1)

WIDTH = 1024
HEIGHT = 768

class Square:
   def __init__(self, axes, side=10):
      side /= 2
      self.P = np.array([[-side, side,side,-side],\
                         [-side,-side,side, side],\
                         [   1,    1,   1,    1 ]])
      self.originalP = self.P.copy()
      self.Axes = axes
      self.sReg = np.zeros((4,2),np.int32)
      self.Cx, self.Cy = self.Axes.midX, self.Axes.midY
   def applyHomography(self,H):
      self.P = H @ self.originalP
      

   def drawSquare(self, Img):
      np.Scale=np.array([[self.Axes.ticSize, 0, self.Axes.midX], [0, -self.Axes.ticSize, self.Axes.midY], [0, 0, 1]])
      for i in range(4):
         p = np.Scale @ self.P[:,i]
         p = p / p[2]
         #self.sReg[i,:] = [int(np.round(x)) for x in [nrm * (self.Cx+self.P[0,i]), nrm * (self.Cy-self.P[1,i])]]
         self.sReg[i,:] = [int(np.round(x)) for x in [p[0], p[1]]]
         cv2.circle(Img, (self.sReg[i, 0], self.sReg[i,1]), 3, (0,255,255), 2)
      cv2.polylines(Img, [self.sReg], True, (0, 255, 128))

plane = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

class trackBar:
   def __init__(self, barName, winName, mn, mx, ini, grain = 100):
      self.barName = barName
      self.winName = winName
      self.minValue = mn
      self.maxValue = mx
      self.value = ini
      self.grain=grain
      self.Fact = (self.maxValue - self.minValue) / self.grain
      self.iFact = 1./self.Fact
      self.pos = self.val2Pos(self.value)
      print ("Pos=",self.pos)
      print ("barName:", self.barName)
      print ("winName:", self.winName)
      cv2.createTrackbar(self.barName, self.winName, self.pos, self.grain, self.onTrackBar)

   def pos2Val(self, pos):
      return self.Fact * pos - self.minValue

   def val2Pos(self, val):
      return int(np.round(self.iFact * (val+self.minValue)))

   def onTrackBarExtra(self, trBar): pass 

   def onTrackBar(self, msfl, pos):
      self.pos = pos
      self.value = self.pos2Val(pos)
      self.onTrackBarExtra(msfl, pos)

class homography:
   def __init__(self, w, h):
      self.width  = w
      self.height = h
      self.H = np.eye(3)
      self.change = False
      self.Tx = self.Ty = 0.
      self.Theta = self.Phi = 0.
      self.L1 = self.L2 = 1.
      self.V1 = self.V2 = 0.
      self.Rpp = np.eye(2)
      self.Rmp = np.eye(2)
      self.Rth = np.eye(2)
      self.L = np.eye(2)
      self.winName = "Homografia"
      self.fondo=np.zeros((400,600,3), dtype=np.uint8)
      cv2.namedWindow(self.winName, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED )
      cv2.imshow(self.winName, self.fondo)
      cv2.waitKey(1)
      self.createTrackBars()

   def createTrackBars(self):
      self.txTB = trackBar("Tx",self.winName, -self.width//2, self.width//2,  0, grain=self.width)
      self.tyTB = trackBar("Ty",self.winName,-self.height//2, self.height//2, 0, grain=self.height)
      self.thetaTB = trackBar("Theta",self.winName,-np.pi, np.pi, 0, grain=360)
      self.phiTB = trackBar("Phi",self.winName,-np.pi, np.pi, 0, grain=360)
      self.L1TB = trackBar("L1",self.winName,0, 10, 1, grain=100)
      self.L2TB = trackBar("L2",self.winName,0, 10, 1, grain=100)
      self.V1TB = trackBar("V1",self.winName,-1, 1, 0, grain=200)
      self.V2TB = trackBar("V2",self.winName,-1, 1, 0, grain=200)
      self.txTB.onTrackBarExtra = self.onTx      
      self.tyTB.onTrackBarExtra = self.onTy
      self.V1TB.onTrackBarExtra = self.onV1
      self.V2TB.onTrackBarExtra = self.onV2
      self.thetaTB.onTrackBarExtra = self.onTheta
      self.phiTB.onTrackBarExtra = self.onPhi
      self.L1TB.onTrackBarExtra = self.onL1
      self.L2TB.onTrackBarExtra = self.onL2

   def Rot(self, alpha):
      c=np.cos(alpha)
      s=np.sin(alpha)
      return np.array([[c, -s],[s,c]])

   def A(self):
      return self.Rth @ self.Rmp @ self.L @ self.Rpp 

   def onTx(trBar, val):
      print("Tx = ", val, trBar.value)
      self.H[0,2] = trBar.value
      self.change = True
   
   def onTy(trBar, val):
      print("Ty = ", val, trBar.value)
      self.H[1,2] = trBar.value
      self.change = True

   def onV1(trBar, val):
      print("V1 = ", val, trBar.value)
      self.H[2,0] = trBar.value
      self.change = True

   def onV2(self, trBar, val):
      print("V2 = ", val, trBar.value)
      self.H[2,1] = trBar.value
      self.change = True

   def onTheta(self, trBar, val):
      print("Theta = ", val, trBar.value)
      self.Theta = trBar.value
      self.Rth = self.Rot(self.Theta)
      self.H[:2,:2] = self.A() 
      self.change = True

   def onPhi(self, trBar, val):
      print("Phi = ", val, trBar.value)
      self.Phi = trBar.value
      self.Rpp = self.Rot(self.Phi)
      self.Rmp = self.Rot(-self.Phi)
      self.H[:2,:2] = self.A() 
      self.change = True
   
   def onL1(self, trBar, val):
      print("L1 = ", val, trBar.value)
      self.L1 = trBar.value
      self.L[0,0] = self.L1
      self.H[:2,:2] = self.A() 
      self.change = True

   def onL2(self, trBar, val):
      print("L2 = ", val, trBar.value)
      self.L2 = trBar.value
      self.L[1,1] = self.L2
      self.H[:2,:2] = self.A() 
      self.change = True

cv2.namedWindow('Plano',\
   cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED )

A = Axes(HEIGHT, WIDTH, 10)
Sq = Square(A, 25)

minTx, midTx, maxTx = 0, WIDTH // (A.ticSize * 2), WIDTH // A.ticSize
minTy, midTy, maxTy = 0, HEIGHT // (A.ticSize * 2), HEIGHT // A.ticSize

hG = homography(WIDTH, HEIGHT)

hG.change = False
print("H=\n",hG.H)
while True:

   if hG.change == True:
      plane=np.zeros(plane.shape, dtype=np.uint8)
      A.drawAxes(plane)
      print("H=\n",hG.H)
      Sq.applyHomography(hG.H)
      Sq.drawSquare(plane)
      changeH = False


   cv2.imshow("Plano", plane)

   key = cv2.waitKeyEx(33) & 0x000000FF
   
   if key == 27:
      break
   
cv2.destroyAllWindows()


