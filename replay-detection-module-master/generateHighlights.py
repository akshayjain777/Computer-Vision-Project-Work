# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:33:49 2019

@author: salvadiswar.sankari
"""
import os
import replayExtraction as re
import pandas as pd
import configparser
import sys
import time
import markShots as ms
#import classification as cl


if __name__=='__main__':
    
    start_time_1 = time.time()
    
    final_video = []
    t1_final = []
    t2_final = []
    t3_final = []
    
    configParser = configparser.RawConfigParser()   
    configFilePath = r"./data/Input/Config_file.txt"
    configParser.read(configFilePath)
    
    
    
    Video_file_path = configParser.get("Input","VIDEO_FILE_PATH")
    
    series = sys.argv[1]
    
    #print(series)
    
    print("Processing for - ",series)

    for video_name in os.listdir(Video_file_path):
        
        
        
        start_time = time.time()
        print("Processing Video - ",video_name)
        
        ### Shot Detection 
        print("Shot Detection")
        shotDetection = ms.markShots(video_name)
        shotDetection.get_shots()
        print("Shot Detection Complete")
        

        
        ### Replay Detection
        
        replayExt = re.logodetection(video_name,series)
        replayExt.logoDetect()
        t1,t2,t3 = replayExt.extractReplay()
        print("Replay Sections of the video are - {},{},{} ".format(t1,t2,t3))
        print("Replay Extraction Complete for - ",video_name)
        
        print("Total Time taken for Replay Extraction - ",time.time()-start_time)
        
        ### Appending Final
        
        final_video.append(video_name)
        t1_final.append(t1)
        t2_final.append(t2)
        t3_final.append(t3)
        
    final_dict = {"video":final_video,"t1":t1_final,"t2":t2_final,"t3":t3_final}
    
    
    final_df = pd.DataFrame(final_dict)
    
    #print(final_df)
    
    #final_df = pd.read_csv("./data/Output/Detected_Logo_601.csv")
    
#    for index,row in final_df.iterrows():
#        try:
#            if (row['t1'] != "" and row['t2'] == "") and (index != len(final_df)):
#    #            print(type(row['t1']))
#    #            print(final_df.loc[index+1,'t1'])
#                if (final_df.loc[index+1,'t1']) == '' or int(final_df.loc[index+1,'t1']) > 2:
#    #                if int(final_df.loc[index+1,'t1']) > 2:
#    #                print("True")
#    #                print(index)
#                    final_df.loc[index,'t1'] = ''
#                else:
#                    index = index+2
#        except:
#            continue
                
            
           
    
    final_df.to_csv("./data/Output/Detected_Logo.csv",index=False)

    print("Total time taken for Replay extraction - ",time.time() - start_time_1)
    