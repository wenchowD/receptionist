# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse
import time
import numpy as np
import cv2 as cv

from receptionist_yunet import YuNet

from PIL import Image
import PIL.Image
if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0  # 解决Pillow报错问题，若所用Pillow版本>9.0, 请删除整个if分支
    PIL.Image.Resampling = PIL.Image



class Receptionst:
    def __init__(self, path):

        self.img_folder_path = path

        backends = [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_CUDA]
        targets = [cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16]
        help_msg_backends = "Choose one of the computation backends: {:d}: OpenCV implementation (default); {:d}: CUDA"
        help_msg_targets = "Chose one of the target computation devices: {:d}: CPU (default); {:d}: CUDA; {:d}: CUDA fp16"
        try:
            backends += [cv.dnn.DNN_BACKEND_TIMVX]
            targets += [cv.dnn.DNN_TARGET_NPU]
            help_msg_backends += "; {:d}: TIMVX"
            help_msg_targets += "; {:d}: NPU"
        except:
            print('This version of OpenCV does not support TIM-VX and NPU. Visit https://gist.github.com/fengyuentau/5a7a5ba36328f2b763aea026c43fa45f for more information.')

        parser = argparse.ArgumentParser(description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
        parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
        parser.add_argument('--model', '-m', type=str, default='face_detection_yunet_2022mar.onnx', help='Path to the model.')
        parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
        parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
        parser.add_argument('--conf_threshold', type=float, default=0.9, help='Filter out faces of confidence < conf_threshold.')
        parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
        parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
        parser.add_argument('--save', '-s', type=str, default=True, help='Set true to save results. This flag is invalid when using camera.')
        parser.add_argument('--vis', '-v', type=self.str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
        self.args = parser.parse_args()


        # Instantiate YuNet
        # 优化地址
        self.model = YuNet(modelPath=self.args.model,
                # inputSize=[320, 320],
                inputSize = [1080,720],
                confThreshold=self.args.conf_threshold,
                nmsThreshold=self.args.nms_threshold,
                topK=self.args.top_k,
                backendId=self.args.backend,
                targetId=self.args.target)
        
    def str2bool(self,v):
        if v.lower() in ['on', 'yes', 'true', 'y', 't']:
            return True
        elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
            return False
        else:
            raise NotImplementedError


    def visualize(self,image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
        output = image.copy()
        landmark_color = [
            (255,   0,   0), # right eye
            (  0,   0, 255), # left eye
            (  0, 255,   0), # nose tip
            (255,   0, 255), # right mouth corner
            (  0, 255, 255)  # left mouth corner
        ]

        if fps is not None:
            cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

        facenum = 1
        for det in (results if results is not None else []):
            bbox = det[0:4].astype(np.int32)
            cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
            #cv.rectangle(output, (0, 0), (100, 200), box_color, 2) #  测试坐标系

            img = Image.open(self.img_folder_path + "X.jpg")  # PIL模块开启 地址要改

            imgCrop = img.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))  # 裁切
            #imgResize = imgCrop.resize((bbox[2], bbox[3]), Image.ANTIALIAS)  # 高质量重制大小
            imgResize = imgCrop.resize((bbox[2], bbox[3]), Image.Resampling.LANCZOS)  # 高质量重制大小

            imgResize.save(self.img_folder_path + "X" + str(facenum) + ".jpg")  # 存储文件 地址要改
            facenum += 1

            conf = det[-1]
            cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

            landmarks = det[4:14].astype(np.int32).reshape((5,2))
            for idx, landmark in enumerate(landmarks):
                cv.circle(output, landmark, 2, landmark_color[idx], 2)

        return output
    
    def main(self,image_put=None):
    # If input is an image
        self.args.input = image_put
        if self.args.input is not None:
            image = cv.imread(self.args.input)
            h, w, _ = image.shape

            # Inference
            self.model.setInputSize([w, h])
            results = self.model.infer(image)

            # Print results
            print('{} faces detected.'.format(results.shape[0]))
            
            '''
            for idx, det in enumerate(results):
                print('{}: {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(
                    idx, *det[:-1])
                )
            '''

            # Draw results on the input image
            image = self.visualize(image, results)
            
            # 标注右下角底色是黄色
            cv.rectangle(image, (image.shape[1] - 250, image.shape[0] - 40), (image.shape[1], image.shape[0]), (0, 255, 255), -1)

            # 标注找到多少人脸
            cv.putText(image, "Finding " + str(results.shape[0]) + " face", (image.shape[1] - 240, image.shape[0] - 5),cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

            # Visualize results in a new window and Save results
            if self.args.vis:
                cv.namedWindow(self.args.input, cv.WINDOW_AUTOSIZE)
                cv.imshow(self.args.input, image)
                cv.imwrite(self.img_folder_path + 'X_result.jpg', image) #  地址要改
                cv.waitKey(3000)
                cv.destroyAllWindows()

            CPx = (2*results[:,0] + results[:,2])/2/w # 中点
            return CPx #  返回 脸中心的x坐标，范围0～1，0.5为画面中央
            

if __name__ == '__main__':
    receptionst = Receptionst()
    receptionst.main()

