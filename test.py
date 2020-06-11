#Import necessary packages

import numpy as np
import cv2
import os
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
K.set_image_dim_ordering('tf')

#path to directories to save classified file

path2 = 'pan'
path1 = 'license'


# load model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

img_rows=128
img_cols=128
num_channel=1
num_epoch=10
filename1='other_images/img1.jpg'
test_image = cv2.imread(filename1,1)

test_image1= cv2.resize(test_image, (800, 500))
cv2.imshow('test_image',test_image1)
cv2.waitKey(0)
a,b= filename1.split('/')
filename=b
img=test_image

test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
   
if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
# Predicting the test image
print((loaded_model.predict(test_image)))
print((loaded_model.predict_classes(test_image)))
result=loaded_model.predict_classes(test_image)

if result[0] == 1:
	print 'pan'
        cv2.imwrite(os.path.join(path2 , filename), img)
        cv2.waitKey(0)
else:
	print 'license'
        cv2.imwrite(os.path.join(path1 , filename), img)
        cv2.waitKey(0)
cv2.destroyAllWindows()

