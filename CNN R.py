
sz=224
batch_size=64
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
'''train_datagen = ImageDataGenerator(
    rotation_range=30,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen=ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory("C:/Users/Sara Durrani/Videos/TPL/Major Assignment/CAT DOG R/data/train",
    target_size=(sz, sz),
    batch_size=batch_size, class_mode='binary')

test_set = test_datagen.flow_from_directory("C:/Users/Sara Durrani/Videos/TPL/Major Assignment/CAT DOG R/data/validation/",
                                            target_size=(sz, sz),
    batch_size=batch_size, class_mode='binary')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(sz,sz,3), activation = 'relu' ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), input_shape=(sz,sz,3), activation = 'relu' ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())


model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



model.summary()

model.fit_generator(train_generator, train_generator.n // batch_size, epochs=8, workers=4,
                   validation_data=test_set, validation_steps=test_set.n // batch_size)


model.save_weights('cat_dog_1.h5')'''

img = image.load_img('C:/Users/Sara Durrani/Videos/TPL/Major Assignment/CAT DOG R/data/evaluation/dog/105.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print (classes)

eval_set = test_datagen.flow_from_directory("C:/Users/Sara Durrani/Videos/TPL/Major Assignment/CAT DOG R/data/evaluation",
                                            target_size=(sz, sz),
    batch_size=batch_size, class_mode='binary')

model.predict_classes(eval_set)

model.load_weights('cat_dog_1.h5')

img = image.load_img('C:/Users/Sara Durrani/Videos/TPL/Major Assignment/CAT DOG R/data/evaluation/cat/150.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print (classes)







































