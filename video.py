#encoding:utf-8
import cv2
import argparse
import os
import datetime


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Process pic')
    parser.add_argument('--input', help='video to process', dest='input',  type=str)
    parser.add_argument('--output', help='pic to store', dest='output',  type=str)
    # default为间隔多少帧截取一张图片
    parser.add_argument('--skip_frame', dest='skip_frame', help='skip number of video', default=30, type=int)
    # input为输入视频的路径 ，output为输出存放图片的路径
    args = parser.parse_args(['--input', r'E:\QQDate\786058509\FileRecv', '--output', r'G:\picture2'])
    print(args.input)
    return args


def process_video(i_video, o_video, num,lab):
    cap = cv2.VideoCapture(i_video)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    expand_name = '.png'
    if not cap.isOpened():
        print("Please check the path.")
    cnt = 0
    count = 0
    while 1:
        ret, frame = cap.read()
        cnt += 1

        if not ret:
            break

        newframe=cv2.resize(frame,(640,480))

        if cnt % num == 0:
            count += 1
            temp=str(count).zfill(3)
            print(os.path.join(o_video,lab+str(count) + expand_name))
            cv2.imwrite(os.path.join(o_video,lab+temp+expand_name), newframe)


if __name__ == '__main__':
    name=['video01','video02','video03']
    starttime = datetime.datetime.now()
    args = parse_args()
    for i in name:
        input = args.input + '\\' + i + ".mp4"
        if not os.path.exists(args.output):
            print(args.output)
            os.makedirs(args.output)
        print(args)
        process_video(input, args.output, args.skip_frame,i)
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
