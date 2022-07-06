from skimage.feature import greycomatrix, greycoprops
import numpy as np
import cv2
import csv
import codecs

# texture data
def tt(characters):
    row = []
    for i in range(4):
        row.append(np.average(characters[i]))
    return row

def title(name):
    rowname = [name]
    for i in range(3):
        rowname.append('')
    return rowname
train_features = [] # The feature vectors
train_labels   = [] # Training Lable
texture = []

totalrow = []
def col(pa,line,na):
        row = []
        image = cv2.imdecode(np.fromfile((pa),dtype = np.uint8),-1)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(gray,distances = [1,5,10,15],angles = [0,np.pi/4, np.pi/2, 3*np.pi/4], levels = 256)

        contrast = greycoprops(glcm, 'contrast') # row is distance，column is angle 
        contrast = tt(contrast)


        dissimilarity = greycoprops(glcm, 'dissimilarity')
        dissimilarity = tt(dissimilarity)

        homogeneity = greycoprops(glcm, 'homogeneity')
        homogeneity = tt(homogeneity)

        correlation = greycoprops(glcm, 'correlation')
        correlation = tt(correlation)

        ASM = greycoprops(glcm, 'ASM')
        ASM = tt(ASM)

        energy = greycoprops(glcm, 'energy')
        energy = tt(energy)
        row = list(contrast)+list(dissimilarity)+list(homogeneity)+list(correlation)+list(ASM)+list(energy)
        if line.split(",")[3] == '1  0  0':
            row.append(0)
        if line.split(",")[3] == '0  1  0':
            row.append(1)
        if line.split(",")[3] == '0  0  1':
            row.append(2)
        row.append(na)

        totalrow.append(row)

with open('./testtext.txt','r',encoding = 'utf-8') as f:
    lines = f.readlines()
    for fi in lines:
        col('./split_test/VIA/{}.jpeg'.format(fi.split(',')[0]),fi,fi.split(',')[0])
titlename = title('contrast')+title('dissimilarity')+title('homogeneity')+title('correlation')+title('ASM')+title('energy')
f = codecs.open('./split_test/VILItesttexture.csv','w','gbk')
writer = csv.writer(f)
writer.writerow(titlename)
writer.writerows(totalrow)
f.close()
