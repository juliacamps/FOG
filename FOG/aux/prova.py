# MLP for Pima Indians Dataset serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model_fog.h5")
# print("Loaded model from disk")
#
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
#                      metrics=['accuracy'])
#
# result_ = loaded_model.evaluate_generator(validation_generator,
#                                           val_samples=_N_VAL_SAMPLE)
# from os import
import numpy as np


dim = 3
    
H = np.eye(dim)
D = np.ones((dim,))
for n in range(1, dim):
    x = np.random.normal(size=(dim-n+1,))
    D[n-1] = np.sign(x[0])
    x[0] -= D[n-1]*np.sqrt((x*x).sum())
    # Householder transformation

    Hx = np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum()
    mat = np.eye(dim)
    mat[n-1:,n-1:] = Hx
    H = np.dot(H, mat)
# Fix the last sign such that the determinant is 1
D[-1] = -D.prod()
H = (D*H.T).T
 
print(H)
print(np.array(H).shape)
