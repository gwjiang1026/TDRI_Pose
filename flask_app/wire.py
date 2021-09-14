# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:21:57 2021

@author: GWJIANG
"""

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
import numpy as np
import base64
from flask import current_app, Blueprint, jsonify, request
import io
import PIL.Image as Image
from cvzone.HandTrackingModule import HandDetector

###############################################

logger = logging.getLogger(__name__)
detector = HandDetector(detectionCon=0.8, maxHands=2)


os.chdir("/opt/flask_app")
sys.path.append("/opt/flask_app")
import HandTrackingModule as htm
mdeia_detector = htm.handDetector(detectionCon=0.75)



print("load function.")


def findpose(lmList):
    tipIds = [4, 8, 12, 16, 20]
    fingers = []
    classtype = -1
    a,b = map(list,zip(*lmList))
    
    if lmList[4][1] == min(b):
        classtype = 1
        result = 'thumb_up'
    elif lmList[4][1] == max(b):
        classtype = 1
        result = 'thumb_down'

    elif detector.findDistance(lmList[4], lmList[1]) < detector.findDistance(lmList[3], lmList[1]) or detector.findDistance(lmList[3], lmList[11])[0]< 30 or (lmList[4][1] - lmList[13][1])**2 + (lmList[4][0] - lmList[13][0])**2 < (lmList[3][1] - lmList[13][1])**2 + (lmList[3][0] - lmList[13][0])**2*0.8 :
        fingers.append(0)
    else:
        fingers.append(1)

    for id in range(1, 5):
        if  ((lmList[tipIds[id]][1]-lmList[0][1])**2 + (lmList[tipIds[id]][0]-lmList[0][0])**2) > ((lmList[tipIds[id]-1][1]-lmList[0][1])**2 + (lmList[tipIds[id]-1][0]-lmList[0][0])**2):
            fingers.append(1)
        else:
            fingers.append(0)
    if classtype != 1:
        totalFingers = fingers.count(1)
        value_result = totalFingers
    else:
        value_result = result         
            
    return value_result


def Pose_backup(img):

    output={ 'status': "100" } 
    value_result = ''
    tipIds = [4, 8, 12, 16, 20]
    lmList = []
    res_objects = []
    result_array = []
    ratio = ''
    detector_img = detector.findHands(img)
    
    for i in range(1,2):
        detector_img = mdeia_detector.findHands(img) # 一次的時候會漏掉
        lmList = mdeia_detector.findPosition(detector_img, draw=False)
        
        if len(lmList) != 0:
            fingers = []
            classtype = -1
            result = ''
            hand = ''
            a,b,c = map(list,zip(*lmList))
                
        ###############################################################################################################    
        
            if (lmList[0][2] - lmList[2][2]) / (lmList[0][1] - lmList[2][1])<0 :
                
                hand = 'right'
            else:
                hand = 'left'
        
        
            if lmList[4][2] == min(c):
                classtype = 1
                result = 'thumb_up'
            elif lmList[4][2] == max(c):
                classtype = 1
                result = 'thumb_down'
        
            #為了大拇指
            elif (lmList[0][2] - lmList[2][2]) / (lmList[0][1] - lmList[2][1])<0 :
                
                #print('right')
                if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1] or (lmList[4][2] - lmList[5][2])**2 + (lmList[4][1] - lmList[5][1])**2<(lmList[3][2] - lmList[5][2])**2 + (lmList[3][1] - lmList[5][1])**2 * 0.8:
                    fingers.append(0)
                else:
                    fingers.append(1)
            else:
               
                #print('left')
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] or (lmList[4][2] - lmList[5][2])**2 + (lmList[4][1] - lmList[5][1])**2<(lmList[3][2] - lmList[5][2])**2 + (lmList[3][1] - lmList[5][1])**2*0.8:
                    fingers.append(0)
                else:
                    fingers.append(1)
        
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
                
            result_array.append(value_result) 
            
    maxlabel = max(result_array,key = result_array.count)

    res_object = {}
    res_object['label'] = maxlabel
    #res_object['hand'] = hand
    res_object['objectRectangle'] =  {
        "top": min(c),
        "left": min(b),
        "width": max(b) - min(b) ,
        "height": max(c) - min(c)  
        }
    output={ 'status': "100" } 
    res_objects.append(res_object)
    output['status'] = 0
    output['pose'] = res_objects

    return output



@api_bp.route('/PredictPose', methods=['GET', 'POST'])
def Pose():
    try:
        
        output={ 'status': "100" } 

        res_objects = []
        value_result1 = ''
        value_result2 = ''
        res_object1 = {}
        res_object2 = {}
        
        
        img_bytes = request.files['image'].read() 
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
        
        #img = cv2.imread(r"C:\Users\GWJIANG\Desktop\GW\python code\yolov4\DEMO\4.JPG")
        hands, img = detector.findHands(img) 

        if hands:
            # Hand 1
            
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
           # centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right
            
            value_result1 = findpose(lmList1)
            res_object1 = {}

            res_object1['label'] = value_result1
            #res_object1['hand'] = handType1
            res_object1['objectRectangle'] =  {
                "top": bbox1[0],
                "left": bbox1[1],
                "width": bbox1[2],
                "height": bbox1[3]  
                }
        
            if len(hands) == 2:
                # Hand 2
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # List of 21 Landmark points
                bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
               # centerPoint2 = hand2['center']  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type "Left" or "Right"
                value_result2 = findpose(lmList2)
                res_object2['label'] = value_result2
                #res_object2['hand'] = handType2
                res_object2['objectRectangle'] =  {
                    "top": bbox2[0],
                    "left": bbox2[1],
                    "width": bbox2[2],
                    "height": bbox2[3]  
                    }


        if res_object1 == {}:
            output = Pose_backup(img)
            return jsonify(output)
        else:
        
            output={ 'status': "100" } 
            if res_object2 != {}:
                res_objects.append(res_object1)
                res_objects.append(res_object2)
            else:
                res_objects.append(res_object1)
            
            output['status'] = 0
            output['pose'] = res_objects
    
            return jsonify(output)
    
    
    
    except Exception as err:
        logger.error("Fatal error in %s", err, exc_info=True)
        status = {"Fatal": str(err)}
        status['pose']= []
        status['status']= "100" 

        return jsonify(status)
    
    
    
    
    
@api_bp.route('/test', methods=['GET', 'POST'])
def test():
    output = {"msg": "I'm the test endpoint from blueprint_x."}
    return jsonify(output)
