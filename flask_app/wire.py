# -*- coding: utf-8 -*-

import logging
import time
from flask_app import app, api_bp
from .utils import *
import requests
import json
import numpy
import cv2
import os
import sys

from flask import current_app, Blueprint, jsonify, request
import io
import PIL.Image as Image

os.chdir("/opt/flask_app")
sys.path.append("/opt/flask_app")
import HandTrackingModule as htm
detector = htm.handDetector(detectionCon=0.75)

###############################################

logger = logging.getLogger(__name__)


print("load function.")

@api_bp.route('/PredictPose', methods=['GET', 'POST'])
def Pose():
    try:
        img_bytes = request.files['image'].read()
        filename = 'predict.jpg'  
          
        ##待修改
        image = Image.open(io.BytesIO(img_bytes))
        image.save(filename)
        value_result = ''
        tipIds = [4, 8, 12, 16, 20]
        
        img = cv2.imread(filename)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
       # lmList =   [[0, 528, 366], [1, 470, 369], [2, 417, 338], [3, 376, 313], [4, 332, 315], [5, 452, 231], [6, 435, 164], [7, 430, 123], [8, 430, 89], [9, 491, 219], [10, 481, 137], [11, 478, 88], [12, 480, 43], [13, 530, 222], [14, 532, 143], [15, 535, 97], [16, 538, 52], [17, 569, 239], [18, 594, 187], [19, 614, 157], [20, 630, 125]]
        output={ 'status': "100" } 
        res_objects = []
        
        if len(lmList) != 0:
            fingers = []
            classtype = -1
            result = ''
            a,b,c = map(list,zip(*lmList))
                
        ###############################################################################################################    
        
            if lmList[4][2] == min(c):
                classtype = 1
                result = 'thumb_up'
            if lmList[4][2] == max(c):
                classtype = 1
                result = 'thumb_down'
        
            #為了大拇指
            if (lmList[0][1] - lmList[2][1]) / (lmList[0][2] - lmList[2][2])<0:
                       
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
        
                if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        
            # 4 FingersNo documentation a
            for id in range(1, 5):
                if  ((lmList[tipIds[id]][2]-lmList[0][2])**2 + (lmList[tipIds[id]][1]-lmList[0][1])**2) > ((lmList[tipIds[id]-1][2]-lmList[0][2])**2 + (lmList[tipIds[id]-1][1]-lmList[0][1])**2):
                    fingers.append(1)
                else:
                    fingers.append(0)
        
            totalFingers = fingers.count(1)
        
            if classtype != 1:
                value_result = totalFingers
            else:
                value_result = result
                
               
            
            res_object = {}
            res_object['label'] = value_result
            res_object['objectRectangle'] =  {
                "top": min(c),
                "left": min(b),
                "width": max(b) - min(b) ,
                "height": max(c) - min(c)  
                }
            
            res_objects.append(res_object)
    
            output['status'] = 0
            output['predict'] = res_objects

            
        return jsonify(output)
    
    except Exception as err:
        logger.error("Fatal error in %s", err, exc_info=True)
        status = {"Fatal": str(err)}
        return jsonify(status)
    
    
@api_bp.route('/test', methods=['GET', 'POST'])
def test():
    """
    ---
    get:
      description: test endpoint
      responses:
        '200':
          description: call successful
          content:
            application/json:
              schema: OutputSchema
      tags:
          - testing
    """
    output = {"msg": "I'm the test endpoint from blueprint_x."}
    return jsonify(output)

