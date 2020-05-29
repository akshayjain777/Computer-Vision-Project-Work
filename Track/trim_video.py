# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:28:29 2020

@author: akshay.jain23
"""

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip("cv5_TN3.mp4",13,16, targetname="TN3_in.mp4")