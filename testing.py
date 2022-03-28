import cv2
import os
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


labels =['Close','Open']

# load the model, that we have created
model = load_model('models/custmodel.h5')



# setting the path to our eye dataset: 
Directory = 'dataset_new/test'
# specify two categories on which we want to train our data:
CATEGORIES = ['Closed' , 'Open']


#setting image size:
img_size = 24
data = []

#iterating over each image and get the image in array form,
for category in CATEGORIES:
    folder = os.path.join(Directory,category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        img_arr = cv2.resize(img_arr,(img_size, img_size),1)
        data.append([img_arr , label])

# see the length of data:
# print(len(data))




# dividing features and label for training the model: 
X = []
Y = []

for features,label in data:
    X.append(features)
    Y.append(label)


#covert them into array:
X = np.array(X)
Y = np.array(Y)

Y = Y.reshape(-1,1)



# save model and architecture to single file
model.save("models/after testing.h5")

test_input = X

test_target = Y
# test_target = np.random.random((24,1,1))
reconstructed_model = keras.models.load_model("models/after testing.h5")

np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)




reconstructed_model.fit(test_input, test_target)


reconstructed_model.summary()

history = reconstructed_model.fit(x=test_input, y=test_target, epochs = 5 , validation_split = 0.1 , batch_size = 32)

# reconstructed_model.Accuracy()

print(history)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""
"""

results = reconstructed_model.evaluate(x=X, y=Y)

print("test loss, test acc:", results)

predictions = reconstructed_model.predict(X)

print("predictions shape:", predictions.shape)

model.save('models/final.h5')