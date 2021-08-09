#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly


# In[2]:


image = cv2.imread("C:/Users/LENOVO/Desktop/YMeet LLC/ymeet-images/ymeet-images/Cars_driving_on_an_expressway.jpeg")
box, label, count = cv.detect_common_objects(image)
output = draw_bbox(image, box, label, count)
plt.imshow(output)
plt.show()
print("Number of cars in this image are " +str(label.count('car')))
print("Numbr of trucks in this image are" +str(label.count('truck')))


# In[3]:


image = cv2.imread("C:/Users/LENOVO/Desktop/YMeet LLC/ymeet-images/ymeet-images/balaji-srinivasan-hZNvounjE_0-unsplash.jpg")
box, label, count = cv.detect_common_objects(image)
output = draw_bbox(image, box, label, count)
plt.imshow(output)
plt.show()
print("Number of cars in this image are " +str(label.count('car')))
print("Numbr of trucks in this image are" +str(label.count('truck')))


# In[4]:


image = cv2.imread("C:/Users/LENOVO/Desktop/YMeet LLC/ymeet-images/ymeet-images/michiel-annaert-pFeCiV0lUwI-unsplash.jpg")
box, label, count = cv.detect_common_objects(image)
output = draw_bbox(image, box, label, count)
plt.imshow(output)
plt.show()
print("Number of cars in this image are " +str(label.count('car')))
print("Numbr of trucks in this image are" +str(label.count('truck')))


# In[ ]:




