#coding=utf-8
import pymysql
import matplotlib
import PIL
import numpy as np


def image_to_array(filenames):
    """
    将图片转化为数组并存为二进制文件
    """
    n = filenames.__len__()  # 获取图片个数
    result = np.array([])  # 创建一个空的一维数组
    print("开始将图片转化为数组")
    for i in range(n):
        image = PIL.Image.open(image_base_path + filenames[i])
        r, g, b = image.split()  # rgb通道分离
        # 注意：下面一定要reshpae(1024)使其变为一维数组，否则拼接的数据会出现错误，导致无法恢复图片
        r_arr = np.array(r).reshape(1024)
        g_arr = np.array(g).reshape(1024)
        b_arr = np.array(b).reshape(1024)
        # 行拼接，类似于接火车；最终结果：共n行，一行3072列，为一张图片的rgb值
        image_arr = np.concatenate((r_arr, g_arr, b_arr))
        result = np.concatenate((result, image_arr))

    result = result.reshape(n, 3072)  # 将一维数组转化为n行3072列的二维数组
    print(result)


# fp = open("./test/1.jpg",'rb')
# img = fp.read()
# fp.close()
# db = pymysql.connect("127.0.0.1","root","root","test")
# cursor = db.cursor()
# sql="INSERT INTO pic(scene,picture) values(%s,%s)"
# cursor.execute(sql,(1,img))
#
# db.commit()
# cursor.close()
# db.close()


def array_to_image(arr):
    arr=np.array(arr)
    rows = arr.shape[0] #rows=5
    arr = arr.reshape(rows,3,32,32)
    print(arr)	#打印数组
    for index in range(rows):
        a = arr[index]
        r = PIL.Image.fromarray(a[0]).convert('L')
        g = PIL.Image.fromarray(a[1]).convert('L')
        b = PIL.Image.fromarray(a[2]).convert('L')
        image = PIL.Image.merge("RGB",(r,g,b))
        #显示图片
        matplotlib.pyplot.imshow(image)
        matplotlib.pyplot.show()
        #image.save(self.image_base_path + "result" + str(index) + ".png",'png')

image_to_array("./test/1.jpg")


# db = pymysql.connect("127.0.0.1","root","root","test")
# cursor = db.cursor()
# sql="select picture from pic where id=%s"
# cursor.execute(sql,3)
# img=cursor.fetchall()
#
# cursor.close()
# db.close()
# array_to_image(img)

