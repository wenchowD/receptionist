#!/usr/bin/env python3
# coding: UTF-8
import rospy
import time
from std_msgs.msg import String # String类型消息,从String.data中可获得信息
from receptionist_Detector import Detector,FaceRecognition #  Detector用于拍照功能 FaceRecognition使用百度API
from receptionist_face_detection import Receptionst
import sys
sys.path.append("..")


NameAndDrink = {
    "host":{
        'name':'HOST', #  主人的名字 要修改
        'drink':'Drink-H' #  主人的饮料 要修改
    },
    "guest1":{
        'name':'GUEST--1', #  客人的名字 要修改
        'drink':'Drink-1' #  客人的饮料 要修改
    },
    "guest2":{
        'name':'GUEST--2', #  客人的名字 要修改
        'drink':'Drink-2' #  客人的饮料 要修改
    },
    "guest3":{
        'name':'GUEST--3', #  客人的名字 要修改
        'drink':'Drink-3' #  客人的饮料 要修改
    }
}

GuestBook = {}


class Rece_vision:
    def __init__(self, path = '/home/dingw/Desktop/test/'):
        self.img_folder_path = path
        self.faceR = FaceRecognition()
        self.takephoto = Detector()
        self.receptionst = Receptionst(self.img_folder_path)
        print('<Done Initializing: R-Vision Ready>')

    def take_pic(self,pic_name):
        print('This is ' + pic_name + '!')
        self.guest1_image_path = self.img_folder_path + pic_name + ".jpg"
        self.takephoto.take_photo(device="cam",image_path=self.guest1_image_path) #  <拍照>: 命名为guest1
        #time.sleep(1)

    def face_rec(self):
        """人脸识别，寻找客人"""
        #time.sleep(2)
        self.guests_path = self.img_folder_path + "X.jpg"
        X_cut = self.receptionst.main(image_put=self.guests_path) #  分割出X中各脸 命名为为X1.jpg, X2.jpg, 并返回脸中点横坐标，横坐标取值0～1，0.5为画面中央
        facenums = X_cut.shape[0] #  脸的个数
        #print(X_cut) #  打印脸坐标 X1 X2
        for i in range(1,1+facenums):
            X_path = self.img_folder_path + "X" + str(i) + ".jpg"
            
            for j in range(1,3):
                guest_path = self.img_folder_path + "guest" + str(j) + ".jpg"

                Rresult = self.faceR.face_run(path1=guest_path,path2=X_path) #  比对 guest_j 和 X_i
                time.sleep(0.3) #  要加延迟，等识别结果

                if Rresult["score"] > 85: #  可信度>85 (满分100)
                    #print('X'+str(i)+' =',NameAndDrink['guest'+str(j)]['name']) #  打印 X_i = guest_j (e.g. X_1 = guest_1)
                    name = NameAndDrink['guest'+str(j)]['name']
                    drink = NameAndDrink['guest'+str(j)]['drink']
                    GuestBook['guest'+str(j)] = {'name': name, 'drink': drink, 'angle':X_cut[i-1], 'x': i}  # 返回啥有待修改
                    break

        #print(GuestBook)
        return GuestBook


if __name__ == '__main__':
    try:
        Rece_vision()  # 实例化Controller,参数为初始化ros节点使用到的名字
        rospy.spin()  # 保持监听订阅者订阅的话题
    except rospy.ROSInterruptException:
        pass
