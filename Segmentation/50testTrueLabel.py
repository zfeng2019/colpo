import matplotlib.image as mping
from cv2 import cv2

# Draw the standard answer rectangle into the image for comparison
with open("./50test.txt","r",encoding = 'utf-8') as f:
        lines = f.readlines()
        for i in lines:
            k = mping.imread('./weight5/label/{}.jpeg'.format(i.split(",")[0]))
            k = cv2.cvtColor(k, cv2.COLOR_BGR2RGB)            
            x,y,width,height = int(i.split(",")[2]),int(i.split(",")[3]),int(i.split(",")[4]),int(i.split(",")[5])
            #386cb0 
            draw_1 = cv2.rectangle(k, (int(x),int(y)), (int(x+width),int(y+height)),(255,255,255),3)
            cv2.imencode('.jpeg',draw_1)[1].tofile('./weight5/50labeltestpic/{}.jpeg'.format(i.split(",")[0]))