import tensorflow as tf
import matplotlib.image as mping
from cv2 import cv2
import pandas as pd

new_model = tf.keras.models.load_model("./weight5/xcedetect_v1.h5")
HEIGHT = 1080
WIDTH = 1440
SIZE = 299
# Obtain test images and predict drawing. This is the first step in testing.
with open("./50test.txt","r",encoding = 'utf-8') as f:
        lines = f.readlines()
        for i in lines:
            image = mping.imread('./test/{}/p2.jpeg'.format(i.split(",")[0]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            k = image
            cv2.imencode('.jpeg',k)[1].tofile('./weight5/original/{}.jpeg'.format(i.split(",")[0]))
            image = image/255
            image = cv2.resize(image,(SIZE,SIZE))
            image = image.reshape(1,SIZE,SIZE,3)
            # predict
            out_1,out_2,out_3,out_4 = new_model.predict(image)        
            x,y,width,height = out_1[0][0]*WIDTH,out_2[0][0]*HEIGHT,out_3[0][0]*WIDTH,out_4[0][0]*HEIGHT
            #33a02c Tag color
            draw_1 = cv2.rectangle(k, (int(x),int(y)), (int(x+width),int(y+height)),(44,160,51),3)
            cv2.imencode('.jpeg',draw_1)[1].tofile('./weight5/label/{}.jpeg'.format(i.split(",")[0]))
            # tailoring
            if x<0:
                x = 0
            if y<0:
                y = 0
            crop = k[int(y):int(height+y),int(x):int(width+x)]
            path = './weight5/split/{}.jpeg'.format(i.split(",")[0])
            cv2.imencode('.jpeg', crop)[1].tofile(path)