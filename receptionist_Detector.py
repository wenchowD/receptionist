#!/usr/bin/env python3
# coding: UTF-8

import rospy
import cv2
from aip import AipFace
import base64
#from pyKinectAzure import pyKinectAzure, _k4a

class Detector(object):
    def __init__(self):
        pass

    def take_photo(self, device='', image_path = ''):
        """电脑摄像头拍照保存"""
        if device == 'k4a' or device == 'kinect':
            pass
            '''
            self.modulePath = r'/usr/lib/x86_64-linux-gnu/libk4a.so'
            self.k4a = pyKinectAzure(self.modulePath)
            self.k4a.device_open()
            device_config = self.k4a.config
            device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
            print(device_config)
            self.k4a.device_start_cameras(device_config)
            path = self.photopath + '/photo.jpg'
            while True:
                self.k4a.device_get_capture()
                color_image_handle = self.k4a.capture_get_color_image()
                if color_image_handle:
                    color_image = self.k4a.image_convert_to_numpy(color_image_handle)
                    cv2.imwrite(path, color_image)
                    if 'photo.jpg' in os.listdir(self.photopath):
                        self.k4a.image_release(color_image_handle)
                        self.k4a.capture_release()
                        break
            self.k4a.device_stop_cameras()
            self.k4a.device_close()
            '''
        else:
            cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
            cap.open(0)
            flag, frame = cap.read()
            cv2.imwrite(image_path, frame) #  保存图片
            cv2.namedWindow("Face", cv2.WINDOW_NORMAL) #  建立图像对象
            faceimg = cv2.imread(image_path)
            cv2.imshow("Face",faceimg) #  显示图片
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            cap.release()
        return image_path



class FaceRecognition(Detector):
    """https://ai.baidu.com/ai-doc/FACE/ek37c1qiz"""
    def __init__(self):
        self.app_id = '27913782'
        self.api_key = 'PqNtUCkOdhPb5I5jbGxaT5qz'
        self.secret_key = '8vbTysHrBBoRphukZP9i69CQsMtlv5Nt'
        self.client = AipFace(self.app_id, self.api_key, self.secret_key)
        super(FaceRecognition, self).__init__()

    def face_run(self,path1,path2):
        image1 = path1
        # print("1:",image1)
        image2 = path2
        # print("2:",image2)
        outcome = self.client.match([
            {
                'image': str(base64.b64encode(open(image1, 'rb').read()),'utf-8'),
                'image_type': 'BASE64',
            },
            {
                'image': str(base64.b64encode(open(image2, 'rb').read()),'utf-8'),
                'image_type': 'BASE64',
            }
        ])
        result = {}
        # if outcome['error_msg'] == 'SUCCESS':
        #     score = outcome['result']['score']
        #     print(score)
        # else:
        #     print('错误信息：', result['error_msg'])
        if outcome['error_code'] == 0:
            result = {}
            result['score'] = outcome['result']['score']
            #if result['score'] > 85:
            #    print("The same!")
        return result

    def detect(self, device='camera'):
        """电脑摄像头拍照检测"""
        path = self.take_photo(device)
        print("saving path:",path)
        result = self.face_run(path2=path)
        return result


if __name__ == '__main__':
    try:
        rospy.init_node('name', anonymous=True)
    except rospy.ROSInterruptException:
        pass
