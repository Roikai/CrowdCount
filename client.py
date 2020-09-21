import socket
from head_detection import detect
import os
import glob as gb
import time
from src.head_detector_vgg16 import Head_Detector_VGG16
from trainer import Head_Detector_Trainer




# client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# client.connect(('10.100.101.34',8081))
#
# count=0
# lab=['video01','video02']
# while True:
#     count+=1
#
#     data=detect(,'./checkpoints/head_detector_final')
#     client.send(data.encode('utf-8'))

if __name__ == '__main__':
    path=r"D:\人群计数\边缘检测端\data\*.png"#关键帧存放地址
    img=gb.glob(path)
    head_detector = Head_Detector_VGG16(ratios=[1], anchor_scales=[2,4])
    trainer = Head_Detector_Trainer(head_detector).cuda()
    trainer.load('./checkpoints/head_detector_final')
    # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client.connect(('192.168.0.104', 8091))
    dic={"01":-1,"02":-1}
    for i in img:
        t1 = time.time()
        temp=(os.path.split(i))[-1]
        num=detect(i,head_detector)
        if dic[temp[5:7]]!=num:
            data=str(temp[5:7])+"-"+str(num)
            print(data)
            dic[temp[5:7]]=num
            # client.send((str(data)).encode())
        else:
            print("数据重复")
        t2= time.time()
        print(t2-t1)

