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

   def reset(self):
      self.P = self.originalP.copy()
      

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
      self.baseValues = [mn, mx, ini, grain]
      self.Fact = (self.maxValue - self.minValue) / self.grain
      self.iFact = 1./self.Fact
      self.val2Pos(self.value)

      print ("barName:", self.barName)
      print ("winName:", self.winName)
      print ("value", self.value)
      print ("Pos=",self.position)
      print ("Fact=", self.Fact)
      print ("iFact=", self.iFact, "\n\n")

   def reset(self):
      self.minValue = self.baseValues[0]
      self.maxValue = self.baseValues[1]
      self.value = self.baseValues[2]
      self.grain = self.baseValues[3]
      self.Fact = (self.maxValue - self.minValue) / self.grain
      self.iFact = 1./self.Fact
      self.val2Pos(self.value)
      self.onTrackBar(self.position)

   def onTrackBar(self, pos): pass 

   def launch(self):
      cv2.createTrackbar(self.barName, self.winName, self.position, self.grain, self.onTrackBar)
      self.onTrackBar(self.position)

   def pos2Val(self, pos):
      self.value = self.Fact * pos - self.minValue

   def val2Pos(self, val):
      self.value = val
      self.position = int(np.round(self.iFact * (self.value + self.minValue)))
      print("val2pos:",val, "int(np,round(%f * (%f + %f))) = %f" % (self.iFact, self.value, self.minValue, self.position))

class homography:
   def __init__(self, w, h):
      self.width  = w
      self.height = h
      self.change = False
      
      self.tx = self.ty = 0.
      self.theta = self.phi = 0.
      self.l1 = self.l2 = 1.
      self.v1 = self.v2 = 0.
      self.Rpp = np.eye(2)
      self.Rmp = np.eye(2)
      self.Rth = np.eye(2)
      self.L = np.eye(2)
      self.H = np.eye(3)
      self.winName = "Homografia"
      self.fondo=np.zeros((400,600,3), dtype=np.uint8)

      cv2.namedWindow(self.winName, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED )
      cv2.imshow(self.winName, self.fondo)
      cv2.waitKey(1)

      self.txTB = trackBar("Tx", self.winName, -self.width//2, self.width//2,  0, grain=self.width)
      self.tyTB = trackBar("Ty", self.winName, -self.height//2, self.height//2, 0, grain=self.height)
      self.l1TB = trackBar("L1", self.winName,0, 10, 1, grain=100)
      self.l2TB = trackBar("L2", self.winName,0, 10, 1, grain=100)
      self.thetaTB = trackBar("Theta", self.winName, -np.pi, np.pi, 0, grain=360)
      self.phiTB = trackBar("Phi", self.winName, -np.pi, np.pi, 0, grain=360)
      self.v1TB = trackBar("V1", self.winName, -1, 1, 0, grain=200)
      self.v2TB = trackBar("V2", self.winName, -1, 1, 0, grain=200)
      self.assignCallBacks()
      self.launchaTrackbars()
      

   def reset(self):
      self.H = np.eye(3)
      self.tx = self.ty = 1.
      self.v1 = self.v2 = 0.
      self.theta = self.phi = 0.
      self.l1 = self.l2 = 1.
      self.txTB.reset()
      self.tyTB.reset()
      self.l1TB.reset()
      self.l2TB.reset()
      self.thetaTB.reset()
      self.phiTB.reset()
      self.v1TB.reset()
      self.v2TB.reset()
      self.change = True
      

   def Rot(self, alpha):
      c=np.cos(alpha)
      s=np.sin(alpha)
      return np.array([[c, -s],[s,c]])

   def computeA(self):
      return self.Rth @ self.Rmp @ self.L @ self.Rpp 

   def onTx(self, pos):
      print("*"*10, "ON Tx", "*"*10)
      self.txTB.position = pos
      self.txTB.pos2Val(pos)
      self.tx = self.txTB.value
      print("Tx = ", pos, self.tx)
      self.H[0,2] = self.tx
      self.change = True

   def onTy(self, pos):
      print("*"*10, "ON Ty", "*"*10)
      self.tyTB.position = pos
      self.tyTB.pos2Val(pos)
      self.ty = self.tyTB.value
      print("Ty = ", pos, self.ty)
      self.H[1,2] = self.ty
      self.change = True

   def onL1(self, pos):
      print("*"*10, "ON L1", "*"*10)
      self.l1TB.position = pos
      self.l1TB.pos2Val(pos)
      self.l1 = self.l1TB.value
      self.L[0,0] = self.l1
      print("L1 = ", pos, self.l1, self.L)
      self.H[:2,:2] = self.computeA() 
      self.change = True

   def onL2(self, pos):
      print("*"*10, "ON L2", "*"*10)
      self.l2TB.position = pos
      self.l2TB.pos2Val(pos)
      self.l2 = self.l2TB.value
      self.L[1,1] = self.l2
      print("L2 = ", pos, self.l2, self.L)
      self.H[:2,:2] = self.computeA() 
      self.change = True
   
   def onTheta(self, pos):
      print("*"*10, "ON θ", "*"*10)
      self.thetaTB.position = pos
      self.thetaTB.pos2Val(pos)
      self.theta = self.thetaTB.value
      print("Theta = ", pos, self.theta)
      self.Rth = self.Rot(self.theta)
      self.H[:2,:2] = self.computeA() 
      self.change = True

   def onPhi(self, pos):
      print("*"*10, "ON ϕ", "*"*10)
      self.phiTB.position = pos
      self.phiTB.pos2Val(pos)
      self.phi = self.phiTB.value
      print("Phi = ", pos, self.phi)
      self.Rth = self.Rot(self.phi)
      self.H[:2,:2] = self.computeA() 
      self.change = True

   def onV1(self, pos):
      print("*"*10, "ON V1", "*"*10)
      self.v1TB.position = pos
      self.v1TB.pos2Val(pos)
      self.v1 = self.v1TB.value
      print("v1 = ", pos, self.v1)
      self.H[2,0] = self.v1
      self.change = True

   def onV2(self, pos):
      print("*"*10, "ON V2", "*"*10)
      self.v2TB.position = pos
      self.v2TB.pos2Val(pos)
      self.v2 = self.v2TB.value
      print("v2 = ", pos, self.v2)
      self.H[2,1] = self.v2
      self.change = True

   def assignCallBacks(self):
      self.txTB.onTrackBar = self.onTx
      self.tyTB.onTrackBar = self.onTy
      self.l1TB.onTrackBar = self.onL1
      self.l2TB.onTrackBar = self.onL2
      self.thetaTB.onTrackBar = self.onTheta
      self.phiTB.onTrackBar = self.onPhi
      self.v1TB.onTrackBar = self.onV1
      self.v2TB.onTrackBar = self.onV2

   def launchaTrackbars(self):
      self.txTB.launch()
      self.tyTB.launch()
      self.l1TB.launch()
      self.l2TB.launch()
      self.thetaTB.launch()
      self.phiTB.launch()
      self.v1TB.launch()
      self.v2TB.launch()

cv2.namedWindow('Plano',\
   cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED )

A = Axes(HEIGHT, WIDTH, 10)
Sq = Square(A, 25)

minTx, midTx, maxTx = 0, WIDTH // (A.ticSize * 2), WIDTH // A.ticSize
minTy, midTy, maxTy = 0, HEIGHT // (A.ticSize * 2), HEIGHT // A.ticSize

hG = homography(WIDTH, HEIGHT)

hG.change = True
print("H=\n",hG.H)
while True:
   if hG.change == True:
      plane=np.zeros(plane.shape, dtype=np.uint8)
      A.drawAxes(plane)
      print("H=\n",hG.H)
      Sq.applyHomography(hG.H)
      Sq.drawSquare(plane)
      hG.change = False


   cv2.imshow("Plano", plane)

   key = cv2.waitKeyEx(33) & 0x000000FF
   if (key < 255):
      print("key=",key)
   if key == 27:
      break
   elif chr(key) == 'r' or chr(key) == 'R':
      Sq.reset()
      hG.reset()
   
cv2.destroyAllWindows()


