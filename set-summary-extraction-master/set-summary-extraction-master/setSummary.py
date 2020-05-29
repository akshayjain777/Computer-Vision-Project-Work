# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:19:41 2019

@author: salvadiswar.sankari
"""

#import pytesseract
import numpy as np
from nms import nms
from decode import decode
import configparser
import cv2


class setSummaryExtraction:
    
    def __init__(self,VIDEO_FILE_NAME,series):
        
        
        configParser = configparser.RawConfigParser()   
        configFilePath = r"./data/Input/Config_file.txt"
        configParser.read(configFilePath)
        
        self.VIDEO_FILE_PATH = configParser.get("Input","VIDEO_FILE_PATH")
        self.VIDEO_FILE_NAME = VIDEO_FILE_NAME
        self.output_path = configParser.get("Output","Output_file_path")
        self.frame_len = configParser.get("Scoreboard_Extraction","frame_len")
#        self.tesseract_path = configParser.get("Scoreboard_Extraction","tesseract_path")
        self.model = configParser.get("Scoreboard_Extraction","model_file")
        
        self.series = series
        
        if self.series == "Australian_Open":
            self.flag = 1
        else:
            self.flag = 0
        
    def findFrames(self):
        
        
        net = cv2.dnn.readNet(self.model)
        vcap = cv2.VideoCapture(self.VIDEO_FILE_PATH+self.VIDEO_FILE_NAME)

        num_frames = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frameNum = []
        self.rec_Cnt = []
        
        self.start = []
        self.end   = [] 
        
        cnt = 0
        print("Finding frames with Statscard / Score ")
        while cnt < num_frames:
            
            vcap.set(cv2.CAP_PROP_POS_FRAMES, cnt)
            ret, img = vcap.read()
            
            blob = cv2.dnn.blobFromImage(img, 1.0, (960, 960), (123.68, 116.78, 103.94), True, False)
            
            
            outputLayers = []
            outputLayers.append("feature_fusion/Conv_7/Sigmoid")
            outputLayers.append("feature_fusion/concat_3")
            
            net.setInput(blob)
            output = net.forward(outputLayers)
             
            scores = output[0]
            geometry = output[1]
            confThreshold = 0.5
            nmsThreshold = 0.4
            
            [boxes, confidences,_] = decode(scores, geometry, confThreshold)
            
            try:
                indicies = nms.boxes(boxes, confidences, nms_function=nms.fast.nms, confidence_threshold=confThreshold,
                                         nsm_threshold=nmsThreshold)
            
                indicies = np.array(indicies).reshape(-1)
                drawrects = np.array(boxes)[indicies]
                self.frameNum.append(cnt)
                self.rec_Cnt.append(len(drawrects))
#                print(cnt,len(drawrects))
                cnt += int(self.frame_len)
            
            except:
                cnt += int(self.frame_len)
                continue
        
        setSummary = np.where(np.array(self.rec_Cnt) >= 40)[0]
        if len(setSummary) > 1:
            self.start = (self.frameNum[setSummary[0]]/25)-2
            self.end   = (self.frameNum[setSummary[-1]]/25)+2
            print("The start and end time for set summary is: ",\
                  self.start,self.end)
        else:
            print("Set Summary is not present in the vdieo")    
            
        return(self.start,self.end)
            
#        print("frameNum is ", self.frameNum)
#        print("rec_CNt is ", self.rec_Cnt)
            
            




            