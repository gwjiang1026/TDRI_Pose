# -*- coding: utf-8 -*-

from flask import jsonify
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

os.chdir("/opt/flask_app")
sys.path.append("/opt/flask_app")
import HandTrackingModule as htm
detector = htm.handDetector(detectionCon=0.75)

###############################################

logger = logging.getLogger(__name__)


print("load function.")

@api_bp.route('/PredictPose', methods=['GET', 'POST'])
def Pose():

    img_bytes = request.files['image'].read()
    filename = 'predict.jpg'  
      
    ##待修改
    image = Image.open(io.BytesIO(img_bytes))
    image.save(filename)

    tipIds = [4, 8, 12, 16, 20]
    output =''
    img = cv2.imread(filename)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
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
            output = totalFingers
        else:
            output = result
            
        res = {"msg": output}
    return jsonify(res)

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

