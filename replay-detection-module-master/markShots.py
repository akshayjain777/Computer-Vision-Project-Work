# Identifies the Shot boudaries and identifies the frames that belong to a 
# camera shot
import cv2
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
import pandas as pd
import numpy as np
import configparser

# main function to idetifies the shots and the frames belonging to a shot
class markShots:
    
    def __init__(self,VIDEO_FILE_NAME):
        
        configParser = configparser.RawConfigParser()   
        configFilePath = r"./data/Input/Config_file.txt"
        configParser.read(configFilePath)
    
        self.VIDEO_FILE_PATH = configParser.get("Input","VIDEO_FILE_PATH")
        
        self.VIDEO_FILE_NAME = VIDEO_FILE_NAME
        
        self.output_path = configParser.get("Output","Output_file_path")
        
    
    def get_shots(self):
    
        # get the full path for the input vdieo
        input_video = self.VIDEO_FILE_PATH + self.VIDEO_FILE_NAME
    
        # set the differnt components of the scene detector package
        video_manager = VideoManager([input_video])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(ContentDetector(threshold=19.0,min_scene_len=30))
    
        try:
            # Set downscale factor to improve processing speed (no args means default).
            video_manager.set_downscale_factor()
    
            # Start video_manager.
            video_manager.start()
            
            # get total number of frames
            num_frames = video_manager.get(cv2.CAP_PROP_FRAME_COUNT)
            video_fps = video_manager.get_framerate()
    
            # Perform scene detection on video_manager.
            scene_manager.detect_scenes(frame_source=video_manager)
            
            rows_list = []
            # get frame level stats
            for i in range(1, num_frames):
                row_dict = {}
                row_dict['fileName'] = self.VIDEO_FILE_NAME
                row_dict['FrameNum'] = i
                row_dict['timeCode'] = FrameTimecode(i,video_fps).get_timecode()
                frame_metrics = stats_manager.get_metrics(i,['content_val',
                                                             'delta_hue',
                                                             'delta_lum',
                                                             'delta_sat'])
                row_dict['metContentVal'] = frame_metrics[0]
                row_dict['metDeltaHue'] = frame_metrics[1]
                row_dict['metDeltaLum'] = frame_metrics[2]
                row_dict['metDeltaSat'] = frame_metrics[3]
                row_dict['Difference'] = 0
                row_dict['shotId'] = 0
                
                rows_list.append(row_dict)
    
            # create the features dataframe
            df_video_features = pd.DataFrame(rows_list)
            
            df_video_features['Difference'] = df_video_features['metContentVal'] - df_video_features['metContentVal'].shift()
            start_frame = 1
            cnt = 0
            end_frame = 0
            for index,row in df_video_features.iterrows():
                end_frame = row['FrameNum']
                if (row['Difference'] > 8 or row['Difference'] < -8):
                    
                    end_frame = row['FrameNum']
                   
                    cond_lower = (df_video_features['FrameNum'] >= start_frame)
                    cond_upper = (df_video_features['FrameNum'] <= end_frame)
                    
                    if ((end_frame - start_frame) > 1):
                        
                        df_video_features.loc[cond_lower & cond_upper ,'shotId'] = int(cnt + 1)
                        start_frame = end_frame
                        cnt += 1
                        
                elif (index == len(df_video_features)-1):
                     df_video_features['shotId'] = np.where(df_video_features['shotId'] ==0 , cnt+1 , df_video_features['shotId'] )
                    
        finally:
            video_manager.release()
        df_video_features = df_video_features.drop(['metContentVal','metDeltaHue','metDeltaLum','metDeltaSat'],axis=1)
        df_video_features.to_csv(self.output_path+"Frame_Classification.csv",index=False)