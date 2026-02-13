#! /usr/bin/env python3
#encoding: utf-8

import numpy as np
import cv2 

class trackbar:
   def __init__(self, barName, winName, mn, mx, ini, grain = 100):
      self.barName = barName
      self.winName = winName
      self.minValue = mn
      self.maxValue = mx
      self.value = ini
      self.grain=grain
      self.position = 0
      self.baseValues = [mn, mx, ini, grain]
      self.Fact = (self.maxValue - self.minValue) / self.grain
      self.iFact = 1./self.Fact
      self.val2Pos(self.value)

   def pos2Val(self, pos):
      self.position = pos
      self.value = self.Fact * pos + self.minValue
      
   def val2Pos(self, val):
      self.value = val
      self.position = int(np.round(self.iFact*(val - self.minValue)))
   

   def reset(self):
      self.minValue = self.baseValues[0]
      self.maxValue = self.baseValues[1]
      self.value = self.baseValues[2]
      self.grain = self.baseValues[3]
      self.Fact = (self.maxValue - self.minValue) / self.grain
      self.iFact = 1./self.Fact
      self.val2Pos(self.value)
      self.onTrackbar(self.position)

   def onTrackbar(self, pos): pass 

   def launch(self):
      cv2.createTrackbar(self.barName, self.winName, self.position, self.grain, self.onTrackbar)
      self.onTrackbar(self.position)