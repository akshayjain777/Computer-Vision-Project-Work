# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:22:00 2019

@author: salvadiswar.sankari
"""

import cv2 
import numpy as np 
import imutils
import multiprocessing as mp
import pandas as pd
#import os
import configparser


class logodetection:	
    
    def __init__(self,VIDEO_FILE_NAME,series):
        
        configParser = configparser.RawConfigParser()   
        configFilePath = r"./data/Input/Config_file.txt"
        configParser.read(configFilePath)
        
        self.VIDEO_FILE_PATH = configParser.get("Input","VIDEO_FILE_PATH")
        self.VIDEO_FILE_NAME = VIDEO_FILE_NAME
        self.replay_path = configParser.get("Output","Output_file_path")
        self.template_path =configParser.get("Logo","Logo_Path")
        self.template_img = configParser.get(series,"logo_file")
        self.gap = configParser.get(series,"gap")
        self.gap = int(self.gap)
        self.threshold = configParser.get(series,"threshold")
        self.threshold = float(self.threshold)
        self.series = series
        self.frame_range = configParser.get(series,"frame_range")
        self.frame_range = int(self.frame_range)
        
    def template_matching(self,frameCount): 
        print("Processsing frame ",frameCount)
        vcap = cv2.VideoCapture(self.VIDEO_FILE_PATH + self.VIDEO_FILE_NAME)
        vcap.set(cv2.CAP_PROP_POS_FRAMES,frameCount)
        ret, img = vcap.read()
        found = None
        overall_found = []
    # Convert to grayscale 
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # Read the template 
        #for template_img in os.listdir(self.template_path):
        template = cv2.imread(self.template_path + self.template_img )
#            cv2.imshow("out",template)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) 
        template = cv2.bitwise_not(template)
        template  = imutils.resize(template, width = int(img_gray.shape[1] * 0.30),height = int(img_gray.shape[0] * 0.30)) 
        w, h = template.shape
        
        for scale in np.linspace(0.5, 1.0, 20)[::-1]: 
      
        # resize the image according to the scale, and keep track 
        # of the ratio of the `resizing 
            resized = imutils.resize(img_gray, width = int(img_gray.shape[1] * scale)) 
            r = img_gray.shape[1] / float(resized.shape[1])  
            if resized.shape[0] < h or resized.shape[1] < w: 
                    break
            res = cv2.matchTemplate(resized,template,cv2.TM_CCOEFF_NORMED) 

            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)

            if (found is None) or (maxVal>found[0]):
                found = (maxVal, maxLoc, r)
        (_, maxLoc, r) = found 
        overall_found.append(found)
            
        return overall_found,frameCount
    
    def logoDetect(self):
    
        out = []
        #p = mp.Pool(mp.cpu_count())
        p = mp.get_context("spawn").Pool(mp.cpu_count())
        input_video = cv2.VideoCapture(self.VIDEO_FILE_PATH+self.VIDEO_FILE_NAME)
        videoLength  = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        #framenum = range(200,videoLength,self.gap)
        self.data_file = pd.read_csv(self.replay_path+"Frame_Classification.csv")
        sub_df = self.data_file[(self.data_file.Difference > 8) | (self.data_file.Difference < -8)].reset_index()
        num = list(sub_df.FrameNum)
        final = []
        for i in num:
            final.extend(range(i-self.frame_range,i+self.frame_range))
        framenum = []    
        for num in final: 
                if num not in framenum: 
                    framenum.append(num)
        framenum = list(filter(lambda x: x >= 0 and x<videoLength , framenum))
        out = []
        frame_num = []
        sim_index = []
        cnt = 0
        #print(framenum)
        for out_found in p.imap(self.template_matching, framenum):
            cnt += self.gap
            found,frame = out_found[0],out_found[1]
            #print("Processing frame {} ".format(frame))
            for i in range(0,len(found)):
                #print(found[i][0])
                #if found[i][0]>self.threshold:
                frame_num.append(frame)
                sim_index.append(found[i][0])
                out.append( "Logo")      
            
        print("Logo Detection Complete")

        new_dict = {"FrameNum":frame_num,"Output":out,"SSIM":sim_index}
        self.df = pd.DataFrame(new_dict)
        self.df.to_csv("./data/Output/Intermediate.csv")
        p.close() 
        p.join()
        
    def extractReplay(self):
#        FrameNum = []
#        replayLabel = []
        print("Extracting Replay Videos")
#        cnt = 1
        actuallogoFrames = []
        self.df_new = self.df[self.df['SSIM'] > self.threshold]
        sub_df = self.df_new[self.df_new['Output'] == 'Logo']
        sub_df = sub_df.reset_index()
        logoFrames = list(sub_df['FrameNum'])
#        print(logoFrames)
        
        if self.series == "Australian_open":        
            i=0
            try:        
                while i < (len(logoFrames)-1):
                    if abs(logoFrames[i+1] - logoFrames[i])< 75:
                        del logoFrames[i]
                        i=0
                    else:
                        i+=1
            except:
                pass
    #        print(logoFrames)
            for j in (logoFrames):
                self.df_frames = self.df.loc[(self.df['FrameNum'] >= j-4) & (self.df['FrameNum'] < j+4) ]
                self.df_frames = self.df_frames.reset_index()
    #            print(self.df_frames)
    #            if len( self.df_frames) > 8 and self.df_frames['SSIM'][0] >= 0.35 and self.df_frames['SSIM'][0] < 0.46 and self.df_frames['SSIM'][len(self.df_frames)-1] >= 0.35 and self.df_frames['SSIM'][len(self.df_frames)-1] <= 0.45:
    #                actuallogoFrames.append(j-5)
                #if (round(self.df_frames['SSIM'].mean(),2) <= 0.46 and round(self.df_frames['SSIM'][0],2) <= 0.46 and round(self.df_frames['SSIM'].mean(),2) >= 0.35 and (self.df_frames['SSIM']<0.4).any()):
                #print(round(self.df_frames['SSIM'].mean(),2))
                #print((self.df_frames['SSIM']<0.4).any())
                #print(self.df_frames[(self.df_frames['SSIM']>0.5)].count()['SSIM'] )
                #print(abs(self.df_frames['SSIM'][0]-self.df_frames['SSIM'][len(self.df_frames)-1]))
                #print(self.df_frames['SSIM'].idxmax())
                if (round(self.df_frames['SSIM'].mean(),2) <= 0.46  and round(self.df_frames['SSIM'].mean(),2) >= 0.35 and (round(self.df_frames['SSIM'],2)<=0.4).any() and abs(self.df_frames['SSIM'][0]-self.df_frames['SSIM'][len(self.df_frames)-1])<0.15 and len(self.df_frames)>6 and (self.df_frames[(self.df_frames['SSIM']>0.5)].count()['SSIM']) <= 2 ):    
                    max_ssim = self.df_frames['SSIM'].idxmax()
    #                print(round(self.df_frames['SSIM'][0:max_ssim],2).is_monotonic)
    #                print(round(self.df_frames['SSIM'][max_ssim:len(self.df_frames)],2).is_monotonic)
    #                print(self.df_frames['SSIM'][0:max_ssim].apply(lambda x:pd.algos.is_monotonic_float64(-x.values)[0]))
    #                print(self.df_frames['SSIM'][max_ssim:len(self.df_frames)].apply(lambda x:pd.algos.is_monotonic_float64(-x.values)[0]))
                        
                    #print("Max index- ",max_ssim)
                    if (self.df_frames['SSIM'][max_ssim] - self.df_frames['SSIM'][max_ssim-1]) < 0.07 and (self.df_frames['SSIM'][max_ssim+1] - self.df_frames['SSIM'][max_ssim]) < 0.07 and self.df_frames['SSIM'][max_ssim] < 0.55:
                        #if (round(self.df_frames['SSIM'][0:max_ssim],2).is_monotonic == True or round(self.df_frames['SSIM'][1:max_ssim],2).is_monotonic == True)  and  round(self.df_frames['SSIM'][max_ssim:len(self.df_frames)],2).is_monotonic == False:
        
                        actuallogoFrames.append(j-5)
                    
            if len(actuallogoFrames) > 2:
                t1 =  int(actuallogoFrames[0]/25)
                t2 = int(actuallogoFrames[1]/25)+1
                t3 = int(actuallogoFrames[2]/25)+1
                
            elif len(actuallogoFrames) == 2:
                t1 = int(actuallogoFrames[0]/25)
                t2 = int(actuallogoFrames[1]/25)+1
                t3 = ""
            elif len(actuallogoFrames) == 1:
                t1 = int(actuallogoFrames[0]/25)
                t2 = ""
                t3 = ""
            else:
                t1 = ""
                t2 = ""
                t3 = ""
#        for i in (logoFrames):
#            if cnt% 2 == 1 and (i-10)>0:
#                actuallogoFrames.append(i-10)
#            if cnt % 2 == 0:
#                actuallogoFrames.append(i+10)
                
#            cnt += 1
        #replaycnt = 1 
#        if len(actuallogoFrames) == 1:
#            actuallogoFrames = []
#            
#        elif len(actuallogoFrames)%2 != 0:
#            del(actuallogoFrames[0])
        
        elif self.series == "French_Open":         
            
            subl = []
            subfin = []
            requi = pd.DataFrame(logoFrames)-pd.DataFrame(logoFrames).shift()
            
            if len(requi) > 0 :
                for i in range(len(requi[0].values.astype(int))):
                    if requi[0].values.astype(int)[i] < 15:
                        subl.extend([logoFrames[i]])
                        if i == len(requi[0].values.astype(int))-1:
                            subfin.append(subl)
                    else:
                        subfin.append(subl)
                        subl = [] 
                        subl.extend([logoFrames[i]])
            
#            for i in subfin:
#                if len(i) <3:
#                    subfin.remove(i)            
            
            sub = 0
            for i in range(len(subfin)):
                i = i + sub
#                print("i",i)
                if len(subfin[i]) <3:
#                    print(subfin[i])
                    del subfin[i]
                    sub = sub-1
                    
                    
                    
            Logoframes = []        
            for fnums in subfin:
                ssim = sub_df[sub_df.FrameNum.isin(fnums)][1:-1]
                if len(ssim) == 0 :
                    continue
                Logoframes.append((ssim["FrameNum"].reset_index(drop=True))[ssim["SSIM"].values.argmax()])
#            print(Logoframes)            
            logoFrames = Logoframes
            
            for j in (logoFrames):
                self.df_frames = self.df.loc[(self.df['FrameNum'] >= j-4) & (self.df['FrameNum'] < j+4) ]
                self.df_frames = self.df_frames.reset_index()
#                print(self.df_frames)
                # change 49 to 47
                # self.df_frames['SSIM'].max()>=0.5 (due to false + this was added)
                if (round(self.df_frames['SSIM'].mean(),2) >= 0.47 and len(self.df_frames)>6 and not(np.any(self.df_frames.SSIM.values<0.3)) and self.df_frames['SSIM'].max()>=0.5):    
                    max_ssim = self.df_frames['SSIM'].idxmax()
                        
                    if (self.df_frames['SSIM'][max_ssim] - self.df_frames['SSIM'][max_ssim-1]) < 0.07 and (self.df_frames['SSIM'][max_ssim+1] - self.df_frames['SSIM'][max_ssim]) < 0.07 :        
                        actuallogoFrames.append(j-5)
                    
            if len(actuallogoFrames) > 2:
                t1 =  int(actuallogoFrames[0]/25)
                t2 = int(actuallogoFrames[1]/25)
                t3 = int(actuallogoFrames[2]/25)
                
            elif len(actuallogoFrames) == 2:
                t1 = int(actuallogoFrames[0]/25)
                t2 = int(actuallogoFrames[1]/25)
                t3 = ""
            elif len(actuallogoFrames) == 1:
                t1 = int(actuallogoFrames[0]/25)
                t2 = ""
                t3 = ""
            else:
                t1 = ""
                t2 = ""
                t3 = ""            
                    
        
        return t1,t2,t3
        
#        vcap = cv2.VideoCapture(self.VIDEO_FILE_PATH+self.VIDEO_FILE_NAME)
#        i = 0
#        while i < len(actuallogoFrames):
#            fourcc = 0x00000021
#            #print(self.replay_path+self.VIDEO_FILE_NAME.split(".")[0]+'_Replay_'+str(replaycnt)+'.mp4')
#            output_video = cv2.VideoWriter(self.replay_path+self.VIDEO_FILE_NAME.split(".")[0]+'_Replay_'+str(replaycnt)+'.mp4', fourcc, 25, (1920, 1080))
#            startFrame = actuallogoFrames[i]
#            endFrame = actuallogoFrames[i+1]
#            print("Extracting Replay - ",replaycnt)
#            while startFrame < endFrame:
#                vcap.set(cv2.CAP_PROP_POS_FRAMES,startFrame)
#                FrameNum.append(startFrame)
#                replayLabel.append("Replay")
#                ret, imgFrame = vcap.read()
#                output_video.write(imgFrame)
#                startFrame += 1
#                
#            i += 2
#            output_video.release()
#            replaycnt+=1
#        #df = {"FrameNum":FrameNum,"replayLabel":replayLabel}
#        #labelsdf = pd.DataFrame.from_dict(df)
#        #datafile = pd.read_csv(self.csv_file)
#        #final_df = pd.merge(datafile,labelsdf,how='left',left_on='FrameNum',right_on='FrameNum')
#        #final_df.to_csv("./data/Output/Frame_Classification.csv",index=False)
#        vcap.release()
        #print("Replay Extraction Complete")
        #return labelsdf
        
#if __name__=='__main__':
#    
#   
#    VIDEO_FILE_PATH = sys.argv[1]    
#    VIDEO_FILE_NAME = sys.argv[2]
#    output_path = sys.argv[3]
#    template = sys.argv[4]
#    
#    replay_ext = logodetection(VIDEO_FILE_PATH, VIDEO_FILE_NAME, output_path,template)
#    replay_ext.logoDetect()
#    replay_ext.extractReplay()
#i=0
#try:        
#    while i < (len(logo)-1):
#        if abs(logo[i+1] - logo[i])< 100:
#            del logo[i+1]
#            i=0
#        else:
#            i+= 1
#except:
#    pass 