from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# init CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3), activation='relu'))

# Pooling 
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metics = ['accuracy'])


# data augmentation (flip, rotate existing images )
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set')