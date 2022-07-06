import tensorflow as tf
import matplotlib.image as mping
from cv2 import cv2
import pandas as pd

new_model = tf.keras.models.load_model("./weight5/resNet101V2detect_v1.h5")
HEIGHT = 1080
WIDTH = 1440
SIZE = 224
# Obtain test images and predict drawing
with open("./50test.txt","r",encoding = 'utf-8') as f:
        lines = f.readlines()
        for i in lines:
            image = mping.imread('./test/{}/p2.jpeg'.format(i.split(",")[0]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            k = mping.imread('./weight5/label/{}.jpeg'.format(i.split(",")[0]))
            k = cv2.cvtColor(k, cv2.COLOR_BGR2RGB)
            image = image/255
            image = cv2.resize(image,(SIZE,SIZE))
            image = image.reshape(1,SIZE,SIZE,3)
            # predict
            out_1,out_2,out_3,out_4 = new_model.predict(image)        
            x,y,width,height = out_1[0][0]*WIDTH,out_2[0][0]*HEIGHT,out_3[0][0]*WIDTH,out_4[0][0]*HEIGHT
            #984ea3 Tag color
            draw_1 = cv2.rectangle(k, (int(x),int(y)), (int(x+width),int(y+height)),(163,78,152),3)#GBR
            cv2.imencode('.jpeg',draw_1)[1].tofile('./weight5/label/{}.jpeg'.format(i.split(",")[0]))

