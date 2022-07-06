import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
ll = os.listdir('./data/image')
ll.sort(key =  lambda x:int(x[:-5]))
images = []

# Gets the name of the image
for i in ll:
    images.append("./data/image/"+i)

HEIGHT = 1080
WIDTH = 1440 # Image resolution
SIZE = 224 # Specifies the size of the input image 
imgs_train = [img for img in images] 
test_count = int(len(imgs_train)*0.2)
train_count = len(imgs_train)-test_count
def to_labels(path):
    sess = []
    with open(path, 'r',encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # Normalize 
            K = [line.split(";")[2],line.split(";")[3],line.split(";")[4],line.split(";")[5]]
            K[0] = float(K[0])/WIDTH
            K[1] = float(K[1])/HEIGHT
            K[2] = float(K[2])/WIDTH
            K[3] = float(K[3])/HEIGHT
            sess.append(K)
    return sess
# Get the output tag    
labels = to_labels('./weight2/150label.txt')

# Output four coordinates, the rectangle's upper-left coordinates x, y, width and height
out_1,out_2,out_3,out_4 = list(zip(*labels))  #把x,y,w,h打包在一起，并且变为列表
out_1 = np.array(out_1)
out_2 = np.array(out_2)
out_3 = np.array(out_3)
out_4 = np.array(out_4)
label_datasets = tf.data.Dataset.from_tensor_slices((out_1,out_2,out_3,out_4))
# Load image
def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,channels = 3)
    image = tf.image.resize(image,(SIZE,SIZE))
    image = image/255
    return image

image_dataset = tf.data.Dataset.from_tensor_slices(imgs_train)
image_dataset = image_dataset.map(load_image)
dataset = tf.data.Dataset.zip((image_dataset,label_datasets))

# Construct training set and optimization set
dataset_train = dataset.skip(test_count)
dataset_validation = dataset.take(test_count)
# The number of steps per loop
BATCH_SIZE = 15
BUFFER_SIZE = 15
STEPS_PER_EPOCH = train_count//BATCH_SIZE
VALIDATION_STEPS = test_count//BATCH_SIZE
dataset_train = dataset_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
dataset_train = dataset_train.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
dataset_validation = dataset_validation.batch(BATCH_SIZE)

#Reconstructing different models
#Xception
#xception = tf.keras.applications.Xception(weights = "imagenet",include_top = False,input_shape = (SIZE,SIZE,3))
#VGG16
vgg16 = tf.keras.applications.VGG16(weights = "imagenet",include_top = False,input_shape = (SIZE,SIZE,3))
#resNet50V2
#resNet50V2 = tf.keras.applications.resnet_v2.ResNet50V2(weights = "imagenet",include_top = False,input_shape = (SIZE,SIZE,3))
#resNet101V2
#resNet101V2 = tf.keras.applications.resnet_v2.ResNet101V2(weights = "imagenet",include_top = False,input_shape = (SIZE,SIZE,3))
#denseNet169
#denseNet169 = tf.keras.applications.densenet.DenseNet169(weights = "imagenet",include_top = False,input_shape = (SIZE,SIZE,3))

inputs = tf.keras.layers.Input(shape = (SIZE,SIZE,3))
x = vgg16(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(2048,activation = "relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256,activation = "relu")(x)
out_1 = tf.keras.layers.Dense(1)(x)  
out_2 = tf.keras.layers.Dense(1)(x)
out_3 = tf.keras.layers.Dense(1)(x)
out_4 = tf.keras.layers.Dense(1)(x)
prediction = [out_1,out_2,out_3,out_4]
model = tf.keras.models.Model(inputs = inputs,outputs = prediction)
model.compile(tf.keras.optimizers.Adam(lr = 0.0001),loss = "mae",metrics = ["mae"])

Epochs = 200
history = model.fit(dataset_train,epochs = Epochs,steps_per_epoch = STEPS_PER_EPOCH,validation_steps = VALIDATION_STEPS,validation_data = dataset_validation)

# Save
#model.save("./weight5/xcedetect_v1.h5")
model.save("./weight5/vgg16detect_v1.h5")
#model.save("./weight5/resNet50V2detect_v1.h5")
#model.save("./weight5/resNet101V2detect_v1.h5")
#model.save("./weight5/denseNet169detect_v1.h5")
loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Save the training loss results
with open('./weight5/vgg16loss.txt','w',encoding = 'utf-8') as f:
    for i in loss:
        f.write("{}\n".format(i))

with open('./weight5/vgg16loss_val.txt','w',encoding = 'utf-8') as f:
    for i in val_loss:
        f.write("{}\n".format(i))
plt.figure()
plt.plot(loss,"r",label = "Training loss")
plt.plot(val_loss,"bo",label = "Validation loss")
plt.title("vgg16 Training and validation Loss")
plt.xlabel("Epoch")
plt.ylim([0,1])
plt.legend()
plt.show()