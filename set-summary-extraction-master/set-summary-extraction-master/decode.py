# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:13:09 2019

@author: salvadiswar.sankari
"""

#import numpy as np
#import cv2

# returns:
#   rects -- an array of [x, y, w, h] that describe rectangles
#   confidences -- an array of floats correstponding to each rectangle in rects
#   baggage -- an array of dictionaries that contain info about each rect including its offset and angle
#
def decode(scores, geometry, confidenceThreshold):
    
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding confidence scores
    (numRows, numCols) = scores.shape[2:4]
    confidences = []    
    rects = [] #(x,y,w,h)
    baggage =[]

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        dTop =          geometry[0, 0, y]
        dRight =        geometry[0, 1, y]
        dBottom =       geometry[0, 2, y]
        dLeft =         geometry[0, 3, y]
        anglesData =    geometry[0, 4, y]
            
        # loop over the number of columns
        for x in range(0, numCols):
        
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < confidenceThreshold:
                continue
    
            confidences.append(float(scoresData[x]))
    
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
                
            # extract the rotation angle for the prediction and then
#            angle = anglesData[x]
                                        
            # offsetX|Y is where the dTop, dRight, dBottom and dLeft are measured from
            # calc the rect corners
            upperRight = (offsetX + dRight[x], offsetY - dTop[x])
            lowerRight = (offsetX + dRight[x], offsetY + dBottom[x])
            upperLeft = (offsetX - dLeft[x], offsetY - dTop[x])
            lowerLeft = (offsetX - dLeft[x], offsetY + dBottom[x])

            rects.append([
                int(upperLeft[0]), # x
                int(upperLeft[1]),  # y
                int(lowerRight[0]-upperLeft[0]), # w
                int(lowerRight[1]-upperLeft[1]) # h
            ])
            
            baggage.append({
                "offset": (offsetX, offsetY),
                "angle": anglesData[x],
                "upperRight": upperRight,
                "lowerRight": lowerRight,
                "upperLeft": upperLeft,
                "lowerLeft": lowerLeft,
                "dTop": dTop[x],
                "dRight": dRight[x],
                "dBottom": dBottom[x],
                "dLeft": dLeft[x]
            })
    
    return (rects, confidences, baggage)
                 
  
    