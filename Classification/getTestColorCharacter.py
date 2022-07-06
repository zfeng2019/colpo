import cv2
import numpy as np
import csv
import codecs

# empty arrays for separating the channels for plotting
B = []
G = []
R = []
H = []
S = []
V = []
Y = []
Cr = []
Cb = []
LL = []
LA = []
LB = []
GRAY =  []
name = []

# Color feature extraction
def col(pa,na):
    # BGR
    im = cv2.imdecode(np.fromfile((pa),dtype = np.uint8),-1)
    im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
    im = cv2.resize(im,(150,150))
    name.append(na)
    b = im[:,:,0]
    b = b.reshape(b.shape[0]*b.shape[1])
    bmid = np.average(b)
    bmed = np.median(b)
    bstd = np.std(b)
    counts = np.bincount(b)
    bmode = np.argmax(counts)
    otsu, th1 = cv2.threshold(b, 0, 255, cv2.THRESH_OTSU)  # method use THRESH_OTSU
    
    B.append([bmid,bmed,bstd,bmode,otsu])

    g = im[:,:,1]
    g = g.reshape(g.shape[0]*g.shape[1])
    gmid = np.average(g)
    gmed = np.median(g)
    gstd = np.std(g)
    counts = np.bincount(g)
    gmode = np.argmax(counts)
    otsu, th1 = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)  # method use THRESH_OTSU
    G.append([gmid,gmed,gstd,gmode,otsu])

    r = im[:,:,2]
    r = r.reshape(r.shape[0]*r.shape[1])
    rmid = np.average(r)
    rmed = np.median(r)
    rstd = np.std(r)
    counts = np.bincount(r)
    rmode = np.argmax(counts)
    otsu, th1 = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)  # method use THRESH_OTSU
    R.append([rmid,rmed,rstd,rmode,otsu])

    # HSV
    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    h = h.reshape(h.shape[0]*h.shape[1])
    hmid = np.average(h)
    hmed = np.median(h)
    hstd = np.std(h)
    counts = np.bincount(h)
    hmode = np.argmax(counts)
    otsu, th1 = cv2.threshold(h, 0, 255, cv2.THRESH_OTSU)  # method use THRESH_OTSU
    H.append([hmid,hmed,hstd,hmode,otsu])

    s = hsv[:,:,1]
    s = s.reshape(s.shape[0]*s.shape[1])
    smid = np.average(s)
    smed = np.median(s)
    sstd = np.std(s)
    counts = np.bincount(s)
    smode = np.argmax(counts)
    otsu, th1 = cv2.threshold(s, 0, 255, cv2.THRESH_OTSU)  # method use THRESH_OTSU
    S.append([smid,smed,sstd,smode,otsu])

    v = hsv[:,:,2]
    v = v.reshape(v.shape[0]*v.shape[1])
    vmid = np.average(v)
    vmed = np.median(v)
    vstd = np.std(v)
    counts = np.bincount(v)
    vmode = np.argmax(counts)
    otsu, th1 = cv2.threshold(v, 0, 255, cv2.THRESH_OTSU)  # method use THRESH_OTSU
    V.append([vmid,vmed,vstd,vmode,otsu])

    # YCrCb
    ycb = cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb)
    y = ycb[:,:,0]
    y = y.reshape(y.shape[0]*y.shape[1])
    ymid = np.average(y)
    ymed = np.median(y)
    ystd = np.std(y)
    counts = np.bincount(y)
    ymode = np.argmax(counts)
    otsu, th1 = cv2.threshold(y, 0, 255, cv2.THRESH_OTSU)  # method use THRESH_OTSU
    Y.append([ymid,ymed,ystd,ymode,otsu])

    cr = ycb[:,:,1]
    cr = cr.reshape(cr.shape[0]*cr.shape[1])
    crmid = np.average(cr)
    crmed = np.median(cr)
    crstd = np.std(cr)
    counts = np.bincount(cr)
    crmode = np.argmax(counts)
    otsu, th1 = cv2.threshold(cr, 0, 255, cv2.THRESH_OTSU)  # method use THRESH_OTSU
    Cr.append([crmid,crmed,crstd,crmode,otsu]) 

    cb = ycb[:,:,2]
    cb = cb.reshape(cb.shape[0]*cb.shape[1])
    cbmid = np.average(cb)
    cbmed = np.median(cb)
    cbstd = np.std(cb)
    counts = np.bincount(cb)
    cbmode = np.argmax(counts)
    otsu, th1 = cv2.threshold(cb, 0, 255, cv2.THRESH_OTSU)  # method use THRESH_OTSU
    Cb.append([cbmid,cbmed,cbstd,cbmode,otsu])     

    # Lab
    lab = cv2.cvtColor(im,cv2.COLOR_BGR2LAB)
    ll = lab[:,:,0]
    ll = ll.reshape(ll.shape[0]*ll.shape[1])
    llmid = np.average(ll)
    llmed = np.median(ll)
    llstd = np.std(ll)
    counts = np.bincount(ll)
    llmode = np.argmax(counts)
    otsu, th1 = cv2.threshold(ll, 0, 255, cv2.THRESH_OTSU)  # method use THRESH_OTSU
    LL.append([llmid,llmed,llstd,llmode,otsu])

    la = lab[:,:,1]
    la = la.reshape(la.shape[0]*la.shape[1])
    lamid = np.average(la)
    lamed = np.median(la)
    lastd = np.std(la)
    counts = np.bincount(la)
    lamode = np.argmax(counts)
    otsu, th1 = cv2.threshold(la, 0, 255, cv2.THRESH_OTSU)  # method use THRESH_OTSU
    LA.append([lamid,lamed,lastd,lamode,otsu])

    lb = lab[:,:,2]
    lb = lb.reshape(lb.shape[0]*lb.shape[1])
    lbmid = np.average(lb)
    lbmed = np.median(lb)
    lbstd = np.std(lb)
    counts = np.bincount(lb)
    lbmode = np.argmax(counts)
    otsu, th1 = cv2.threshold(lb, 0, 255, cv2.THRESH_OTSU)  # method use THRESH_OTSU
    LB.append([lbmid,lbmed,lbstd,lbmode,otsu])

    #GRAY
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = gray.reshape(gray.shape[0]*gray.shape[1])
    graymid = np.average(gray)
    graymed = np.median(gray)
    graystd = np.std(gray)
    counts = np.bincount(gray)
    graymode = np.argmax(counts)
    otsu, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # method use THRESH_OTSU
    GRAY.append([graymid,graymed,graystd,graymode,otsu])    

# Data creation
# append the values from each file to the respective channel
with open('./testtext.txt','r',encoding = 'utf-8') as f:
    lines = f.readlines()
    for fi in lines:
        col('./split_test/VILI/{}.jpeg'.format(fi.split(',')[0]),fi.split(',')[0])


# Prints the color characteristics of the test set
f = codecs.open('./split_test/VILItestyense.csv','w','gbk')
writer = csv.writer(f)
writer.writerow(['','R_mid','R_med','R_std','R_mode','R_otsu','G_mid','G_med','G_std','G_mode','G_otsu','B_mid','B_med','B_std','B_mode','B_otsu','H_mid','H_med','H_std','H_mode','H_otsu','S_mid','S_med','S_std','S_mode','S_otsu','V_mid','V_med','V_std','V_mode','V_otsu','Y_mid','Y_med','Y_std','Y_mode','Y_otsu','Cr_mid','Cr_med','Cr_std','Cr_mode','Cr_otsu','Cb_mid','Cb_med','Cb_std','Cb_mode','Cb_otsu','LL_mid','LL_med','LL_std','LL_mode','LL_otsu','LA_mid','LA_med','LA_std','LA_mode','LA_otsu','LB_mid','LB_med','LB_std','LB_mode','LB_otsu','gray_mid','gray_med','gray_std','gray_mode','gray_otsu'])
for i in range(len(B)):
    row = [] 
    row.append(name[i])
    row.append(R[i][0])
    row.append(R[i][1])
    row.append(R[i][2])
    row.append(R[i][3])
    row.append(R[i][4])

    row.append(G[i][0])
    row.append(G[i][1])
    row.append(G[i][2])
    row.append(G[i][3])
    row.append(G[i][4])

    row.append(B[i][0])
    row.append(B[i][1])
    row.append(B[i][2])
    row.append(B[i][3])
    row.append(B[i][4])

    row.append(H[i][0])
    row.append(H[i][1])
    row.append(H[i][2])
    row.append(H[i][3])
    row.append(H[i][4])
    
    row.append(S[i][0])
    row.append(S[i][1])
    row.append(S[i][2])
    row.append(S[i][3])
    row.append(S[i][4])

    row.append(V[i][0])
    row.append(V[i][1])
    row.append(V[i][2])
    row.append(V[i][3])
    row.append(V[i][4])

    row.append(Y[i][0])
    row.append(Y[i][1])
    row.append(Y[i][2])
    row.append(Y[i][3])
    row.append(Y[i][4])

    row.append(Cr[i][0])
    row.append(Cr[i][1])
    row.append(Cr[i][2])
    row.append(Cr[i][3])
    row.append(Cr[i][4])

    row.append(Cb[i][0])
    row.append(Cb[i][1])
    row.append(Cb[i][2])
    row.append(Cb[i][3])
    row.append(Cb[i][4])

    row.append(LL[i][0])
    row.append(LL[i][1])
    row.append(LL[i][2])
    row.append(LL[i][3])
    row.append(LL[i][4])

    row.append(LA[i][0])
    row.append(LA[i][1])
    row.append(LA[i][2])
    row.append(LA[i][3])
    row.append(LA[i][4])
    
    row.append(LB[i][0])
    row.append(LB[i][1])
    row.append(LB[i][2])
    row.append(LB[i][3])
    row.append(LB[i][4])

    row.append(GRAY[i][0])
    row.append(GRAY[i][1])
    row.append(GRAY[i][2])
    row.append(GRAY[i][3])
    row.append(GRAY[i][4])
    writer.writerow(row)
f.close()
