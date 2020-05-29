# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:14:49 2019

@author: salvadiswar.sankari
"""
import time
import configparser
import os
import setSummary as es
import sys
import pandas as pd

if __name__=='__main__':
    
    configParser = configparser.RawConfigParser()   
    configFilePath = r"./data/Input/Config_file.txt"
    configParser.read(configFilePath)
    
    
    input_path = configParser.get("Input","VIDEO_FILE_PATH")
    output_path = configParser.get("Output","Output_file_path")
    
    series = sys.argv[1]
    Video = []
    Start = []
    End   = []
    
    for VIDEO_FILE_NAME in os.listdir(input_path):
        start_time = time.time()
        print("Processing Video - ",VIDEO_FILE_NAME)
        
        ### Scorecard extraction
        
        print("Series - ",series)
        extractScores = es.setSummaryExtraction(VIDEO_FILE_NAME,series)
        start, end = extractScores.findFrames()
        Video.append(VIDEO_FILE_NAME)
        Start.append(start)
        End.append(end)
        print("Extraction Complete for - ",VIDEO_FILE_NAME)
        print("Total Time taken - ",time.time()-start_time)
        
    cols = ["VIDEO_FILE_NAME","StartTime","EndTime"]    
    df = pd.DataFrame(list(zip(Video,Start,End)),columns=cols)
    df.to_csv(output_path+"SetSummaryAO.csv",index=False) 
        
        
        
        