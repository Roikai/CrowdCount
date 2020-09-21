from __future__ import division

import os
import torch as t
from src.config import opt
from src.head_detector_vgg16 import Head_Detector_VGG16
from trainer import Head_Detector_Trainer
from PIL import Image
import numpy as np
from data.dataset import preprocess
import matplotlib.pyplot as plt 
import src.array_tool as at
from src.vis_tool import visdom_bbox
import argparse
import src.utils as utils
from src.config import opt
import time

SAVE_FLAG = 0
THRESH = 0.01
IM_RESIZE = False

def read_img(path):
    f = Image.open(path)
    if IM_RESIZE:
        f = f.resize((640,480), Image.ANTIALIAS)

    f.convert('RGB')
    img_raw=f.copy()
    #img_raw = np.asarray(f, dtype=np.uint8)
    #img_raw_final = img_raw.copy()
    img = np.asarray(f, dtype=np.float32)
    _, H, W = img.shape
    img = img.transpose((2,0,1))
    img = preprocess(img)
    _, o_H, o_W = img.shape
    scale = o_H / H
    img_raw=img_raw.resize((o_W, o_H), Image.ANTIALIAS)
    return img,img_raw,scale

def detect(img_path, model_path):
    file_id = utils.get_file_id(img_path)
    img,img_raw ,scale = read_img(img_path)
    head_detector = Head_Detector_VGG16(ratios=[1], anchor_scales=[2,4])
    trainer = Head_Detector_Trainer(head_detector).cuda()
    trainer.load(model_path)
    img = at.totensor(img)
    img = img[None, : ,: ,:]
    img = img.cuda().float()
    st = time.time()
    pred_bboxes_, _ = head_detector.predict(img, scale, mode='evaluate', thresh=THRESH)
    et = time.time()
    tt = et - st
    print ("[INFO] 监测完成. 时间为: {:.4f} s".format(tt))


    for i in range(pred_bboxes_.shape[0]):
        ymin, xmin, ymax, xmax = pred_bboxes_[i,:]
        #utils.draw_bounding_box_on_image_array(img_raw,ymin-1.5*(ymax-ymin), xmin-0.2*(xmax), ymax-1.0*(ymax-ymin), xmax-0.2*(xmax))
        utils.draw_bounding_box_on_image(img_raw, ymin, xmin,ymax, xmax)
    plt.axis('off')
    plt.imshow(img_raw)
    if SAVE_FLAG == 1:
        plt.savefig(os.path.join(opt.test_output_path, file_id+'.png'), bbox_inches='tight', pad_inches=0)
    else:
        print("人数为:", pred_bboxes_.shape[0])
        plt.ion()
        plt.pause(1)
        #plt.close()
        plt.show()

    #plt.close()
    return pred_bboxes_.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help="test image path",default='./test/1.png')
    parser.add_argument("--model_path", type=str, default='./checkpoints/head_detector_final')
    #parser.add_argument("--model_path", type=str, default=r'H:\mod\checkpoints\head_detector_')
    args = parser.parse_args()
    num=detect(args.img_path, args.model_path)
    print("人数为:",num)
    # model_path = './checkpoints/sess:2/head_detector08120858_0.682282441835'

    # test_data_list_path = os.path.join(opt.data_root_path, 'brainwash_test.idl')
    # test_data_list = utils.get_phase_data_list(test_data_list_path)
    # data_list = []
    # save_idx = 0
    # with open(test_data_list_path, 'rb') as fp:
    #     for line in fp.readlines():
    #         if ":" not in line:
    #             img_path, _ = line.split(";")
    #         else:
    #             img_path, _ = line.split(":")

    #         src_path = os.path.join(opt.data_root_path, img_path.replace('"',''))
    #         detect(src_path, model_path, save_idx)
    #         save_idx += 1



