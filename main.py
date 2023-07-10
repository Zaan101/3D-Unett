from array import array
from tkinter import Image
import numpy as np
from numpy.lib import type_check
import tensorflow as tf
import glob
import pandas
import ants
import matplotlib.pyplot as plt
import test
import keras
from keras.preprocessing.image import ImageDataGenerator
import nibabel as nib
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical,normalize
from keras.models import Model
from keras.layers import BatchNormalization, MaxPooling2D,Conv2D,Dense,Concatenate,Input,Dropout,Maximum,Activation,Dense,Flatten,UpSampling2D,Conv2DTranspose,Add,Multiply,Lambda, concatenate
import os
import keras.callbacks as callbacks
import keras.initializers as initializers
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras import regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import segmentation_models as sm
kernel_initializer =  'he_uniform'

import glob
import random
def sorter(e):
    new = e.split("_")
    if "t2w" in new[4]:
        return (new[1],new[2],1)
    if "flair"in new[4]:
        return (new[1],new[2],2)
    else:
 
        return (new[1],new[2],3,new[3])

def mask_sorter(m):
    new = m.split("_")
    return (new[2],new[3])

def display_image(img):
  plt.imshow(img)
  plt.axis("off")
  plt.show()

 


image_array = []
temp_arr = []
files = glob.glob("/home/afilippov/Downloads/image_data/*.gz")
for file in sorted(files,key=sorter):
  temp = ants.utils.convert_nibabel.nifti_to_ants(nib.load(file))

  if temp.shape != (238,256,16):
    temp = ants.resample_image(temp,(238,256,16),use_voxels=True,interp_type=0) 
  temp = ants.crop_indices( temp, (7,16,0), (231,240,16))
  temp_arr.append(temp)
  if len(temp_arr) == 4:


    image_array.append(temp_arr)
    temp_arr = []


mask_array=[]
files = sorted(glob.glob("/home/afilippov/Downloads/seg_data/*.gz"),key=mask_sorter)
for file in files:
  print(file)
  temp = ants.utils.convert_nibabel.nifti_to_ants(nib.load(file))
  temp = ants.crop_indices( temp, (7,16,0), (231,240,16))

  mask_array.append(temp)





for x in range(len(image_array)):
  fi = image_array[x][0]
  for i in range(1,4): 
    t =ants.registration(fixed=fi,moving=image_array[x][i], type_of_transform= 'Affine')
  
    image_array[x][i] = ants.n3_bias_field_correction(t.get("warpedmovout"))

  

for x in range(len(mask_array)):

  mask_array[x]= mask_array[x].numpy()  

for x in range(len(image_array)):
   
  for i in range(0,4):
    image_array[x][i] = image_array[x][i].numpy()
    image_array[x][i] = (image_array[x][i] - np.min(image_array[x][i])) / (np.max(image_array[x][i]) - np.min(image_array[x][i]))

image_array = np.asarray(image_array)
mask_array = np.asarray(mask_array)

slice = random.randint(0,21)
chan = random.randint(0,15)
i =6
plt.imshow(image_array[6,0,:,:,i], cmap='gray')
plt.imshow(mask_array[6,:,:,i], cmap='jet', alpha=0.5)
plt.show()
print(mask_array.shape)
print(image_array.shape)



mask_array = mask_array.reshape(-1,224,224,1)

image_array= image_array.reshape(-1,224,224,4)
mask_array[mask_array==7.]=6.

mask_array = to_categorical(mask_array)
print(np.shape(mask_array))
from sklearn.utils import class_weight


x1,x_test,y1,y_test =train_test_split(image_array,mask_array,test_size = 0.2)

x_train,x_val,y_train,y_val = train_test_split(x1,y1,test_size =0.1)

num_classes = 7
seed = 1
batch_size = 8

img_data_gen_args = dict(
                      data_format = "channels_last",
                      rotation_range=180,
                      width_shift_range=0.7,
                      height_shift_range=0.7,
                      shear_range=0.5,
                      zoom_range=0.8,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='constant',
                      cval =0,
                      )



image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_generator = image_data_generator.flow(x_train,y_train,seed=seed,batch_size=batch_size)



val_data_generator = image_data_generator.flow(x_val,y_val,seed=seed,batch_size=batch_size)





from keras import backend as K
def dice_coef(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection) / (K.sum(K.square(y_true),axis=-1) + K.sum(K.square(y_pred),axis=-1) + epsilon)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

inputs = Input(shape=(224,224,4),name='input')  

block0_conv1 = Conv2D(32,3,padding='same',activation='relu',name='block0_conv1')(inputs)
block0_conv2 = Conv2D(32,3,padding='same',activation='relu',name='block0_conv2')(block0_conv1)
block0_norm = BatchNormalization(name='block0_batch_norm')(block0_conv2)
block0_pool = MaxPooling2D(name='block0_pool')(block0_norm)

block1_conv1 = Conv2D(64,3,padding='same',activation='relu',name='block1_conv1')(block0_pool)
block1_conv2 = Conv2D(64,3,padding='same',activation='relu',name='block1_conv2')(block1_conv1)
block1_norm = BatchNormalization(name='block1_batch_norm')(block1_conv2)
block1_pool = MaxPooling2D(name='block1_pool')(block1_norm)

block2_conv1 = Conv2D(128,3,padding='same',activation='relu',name='block2_conv1')(block1_pool)
block2_conv2 = Conv2D(128,3,padding='same',activation='relu',name='block2_conv2')(block2_conv1)
block2_norm = BatchNormalization(name='block2_batch_norm')(block2_conv2)
block2_pool = MaxPooling2D(name='block2_pool')(block2_norm)

encoder_dropout_1 = Dropout(0.2,name='encoder_dropout_1')(block2_pool)

block3_conv1 = Conv2D(256,3,padding='same',activation='relu',name='block3_conv1')(encoder_dropout_1)
block3_conv2 = Conv2D(256,3,padding='same',activation='relu',name='block3_conv2')(block3_conv1)
block3_norm = BatchNormalization(name='block3_batch_norm')(block3_conv2)
block3_pool = MaxPooling2D(name='block3_pool')(block3_norm)

block4_conv1 = Conv2D(512,3,padding='same',activation='relu',name='block4_conv1')(block3_pool)
block4_conv2 = Conv2D(512,3,padding='same',activation='relu',name='block4_conv2')(block4_conv1)
block4_norm = BatchNormalization(name='block4_batch_norm')(block4_conv2)
block4_pool = MaxPooling2D(name='block4_pool')(block4_norm)

block5_conv1 = Conv2D(1024,3,padding='same',activation='relu',name='block5_conv1')(block4_pool)

#decoder
up_pool1 = Conv2DTranspose(512,3,strides = (2, 2),padding='same',activation='relu',name='up_pool1')(block5_conv1)
merged_block1 = Add()([block4_norm,up_pool1])
decod_block1_conv1 = Conv2D(512,3, padding = 'same', activation='relu',name='decod_block1_conv1')(merged_block1)

up_pool2 = Conv2DTranspose(256,3,strides = (2, 2),padding='same',activation='relu',name='up_pool2')(decod_block1_conv1)
merged_block2 = Add()([block3_norm,up_pool2])
decod_block2_conv1 = Conv2D(256,3,padding = 'same',activation='relu',name='decod_block2_conv1')(merged_block2)

decoder_dropout_1 = Dropout(0.3,name='decoder_dropout_1')(decod_block2_conv1)

up_pool3 = Conv2DTranspose(128,3,strides = (2, 2),padding='same',activation='relu',name='up_pool3')(decoder_dropout_1)
merged_block3 = Add()([block2_norm,up_pool3])
decod_block3_conv1 = Conv2D(128,3,padding = 'same',activation='relu',name='decod_block3_conv1')(merged_block3)

up_pool4 = Conv2DTranspose(64,3,strides = (2, 2),padding='same',activation='relu',name='up_pool4')(decod_block3_conv1)
merged_block4 = Add()([block1_norm,up_pool4])
decod_block4_conv1 = Conv2D(64,3,padding = 'same',activation='relu',name='decod_block4_conv1')(merged_block4)

up_pool5 = Conv2DTranspose(32,3,strides = (2, 2),padding='same',activation='relu',name='up_pool5')(decod_block4_conv1)
merged_block5 = Add()([block0_norm,up_pool5])
decod_block5_conv1 = Conv2D(32,3,padding = 'same',activation='relu',name='decod_block5_conv1')(merged_block5)

pre_output = Conv2D(64,1,padding = 'same',activation='relu',name='pre_output')(decod_block5_conv1)

output = Conv2D(7,1,padding='same',activation='softmax',name='output')(pre_output)

model = Model(inputs = inputs, outputs = output)
model.summary()


    

focal_loss = sm.losses.CategoricalFocalLoss()
model.save('first_run.hdf5')

checkpointer = callbacks.ModelCheckpoint(filepath = 'firstrun.hdf5',save_best_only=True)
training_log = callbacks.TensorBoard(log_dir='./Model_logs')

model.compile(optimizer=Adam(lr=1e-5),loss=focal_loss,metrics=['accuracy'])




steps = len(x_train) //batch_size
val_steps = len(x_val) //batch_size

history = model.fit(image_generator,
                    epochs=100, 
                    validation_data= val_data_generator, 
                    steps_per_epoch = steps,
                    validation_steps = val_steps,
                    verbose = 1,
                    shuffle=True,
                    callbacks = [training_log,checkpointer]
                    )
                    




#plot the training and validation accuracy and loss at each epoch
 
_, acc = model.evaluate(x_test, y_test)
print("Accuracy is = ", (acc * 100.0), "%")


predict =[]
temp_arr = []
files = glob.glob("/home/afilippov/Downloads/predict_image/*.gz")
for file in sorted(files,key=sorter):
  temp = ants.utils.convert_nibabel.nifti_to_ants(nib.load(file))
  if temp.shape != (238,256,16):
    temp = ants.resample_image(temp,(238,256,16),use_voxels=True,interp_type=0) 
  temp = ants.crop_indices( temp, (7,16,0), (231,240,16))
  temp_arr.append(temp)
predict.append(temp_arr)


for x in range(len(predict)):
  fi = predict[x][0]
  for i in range(1,4): 
    t =ants.registration(fixed=fi,moving=predict[x][i], type_of_transform= 'Affine')
  
    predict[x][i] = ants.n3_bias_field_correction(t.get("warpedmovout"))

for x in range(len(predict)):
  for i in range(0,4):
    predict[x][i] = predict[x][i].numpy()
    predict[x][i] = (predict[x][i] - np.min(predict[x][i])) / (np.max(predict[x][i]) - np.min(predict[x][i]))

predict = np.asarray(predict)
np.shape(predict)
plt.imshow(predict[0,0,:,:,5], cmap='jet')
plt.show()
predict = predict.reshape(-1,224,224,4)
predict.shape


y_pred = model.predict(predict)
y_img = np.argmax(y_pred,axis=-1)
np.unique(y_img)
y_img.shape



plt.imshow(y_img[5,:,:], cmap='jet')
plt.show()