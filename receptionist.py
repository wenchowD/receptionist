
from std_msgs.msg import String # String类型消息,从String.data中可获得信息
from receptionist_vision import Rece_vision
import rospy
import time

class Controller:
    def __init__(self):
        rospy.init_node('receptionist', anonymous=True) #  初始化ros节点
        
        self.receptionst = Rece_vision(path = '/home/dingw/Desktop/test1/')

        rospy.Subscriber('/go_receptionist', String, self.go_recep)
        print('<Done Initializing: Receptionist Ready>')

    def go_recep(self,msg):

        #self.receptionst.take_pic(pic_name='guest1') #  给guest1拍照
        #time.sleep(3)
        #self.receptionst.take_pic(pic_name='guest2') #  给guest2拍照
        #time.sleep(3)
        #self.receptionst.take_pic(pic_name='guest3') #  给guest3拍照
        #time.sleep(3)
        #self.receptionst.take_pic(pic_name='X') #  合照，待识别
        #time.sleep(3)

        result = self.receptionst.face_rec() # 检测, 返回字典
        print(result)

if __name__ == '__main__':
    try:
        Controller()  # 实例化Controller,参数为初始化ros节点使用到的名字
        rospy.spin()  # 保持监听订阅者订阅的话题
    except rospy.ROSInterruptException:
        pass
