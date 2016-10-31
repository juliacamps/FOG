"""Exercise: the Messi Neymar Iniesta challenge, using CNNs"""

import numpy as np
import random as rd

from os import mkdir
from os import listdir
from os.path import isfile
from os.path import join
from shutil import rmtree
from PIL import Image

DATA_PATH = './data_fcbdata/'
RAW_DATA_PATH = \
    '/home/juli/Desktop/Deep_learning_developer__exercise/fcbdata/'
DATA_EXT = '.jpg'
DATA_VAL = DATA_PATH + 'validation/'
DATA_TRAIN = DATA_PATH + 'train/'
DATA_TEST = DATA_PATH + 'test/'
IMG_CHANNELS = 3
IMG_HEIGHT = 60
IMG_WIDTH = 60
N_CLASS = 3
N_EPOCH = 100
N_FOLD = 1


def save_transformed_img(orig_path, target_path):
    """Save reshaped image
    
    Parameters
    ----------
    orig_path : str
        Original file path.
    target_path : str
        Target file path.
    """
    im = Image.open(orig_path)
    c = np.asarray(im.convert())
    im.frombytes(c.reshape((IMG_CHANNELS, IMG_WIDTH, IMG_HEIGHT)))
    im.save(target_path)
    im.close()


def create_dataset(train_frac=0.6, val_frac=0.3, test_frac=0.1):
    """Prepare data set
    
    Parameters
    ----------
    train_frac : float, optional, default=0.6
    val_frac : float, optional, default=0.3
    test_frac : float, optional, default=0.1
    
    Return
    ------
    : array-like
        Number of items for each partition.
    """

    try:
        rmtree(DATA_PATH)
    # This would be "except OSError, e:" before Python 2.6
    except Exception as e:
        print(e)

    mkdir(DATA_PATH)
    mkdir(DATA_TRAIN)
    mkdir(DATA_VAL)
    mkdir(DATA_TEST)
    class_name = ['iniesta', 'messi', 'neymar']

    count_val = 0
    count_train = 0
    count_total = 0

    for class_ in class_name:
        new_train_dir = DATA_TRAIN + class_
        new_val_dir = DATA_VAL + class_
        new_test_dir = DATA_TEST + class_
        mkdir(new_train_dir)
        mkdir(new_val_dir)
        mkdir(new_test_dir)
    
        dataset_path = np.array([[join(RAW_DATA_PATH, f), f] for f in
                                 listdir(RAW_DATA_PATH) if
                                 isfile(join(RAW_DATA_PATH,
                                             f)) and f.endswith(
                                     DATA_EXT) and class_.upper()
                                 in f.upper()])
        n_sample = len(dataset_path)
        n_train_ = int(n_sample * train_frac)
        n_val_ = int(n_sample * val_frac)
        rd.shuffle(dataset_path)
        count_train += n_train_
        count_val += n_val_
        count_total += n_sample

        train_data = dataset_path[:n_train_]
        val_data = dataset_path[n_train_:(n_val_ + n_train_)]
        test_data = dataset_path[(n_val_ + n_train_):]

        for train_path, train_name in train_data:
            save_transformed_img(train_path, (new_train_dir + '/'
                                              + train_name))
    
        for val_path, val_name in val_data:
            save_transformed_img(val_path, (new_val_dir + '/'
                                            + val_name))
            
        if test_frac > 0:
            for test_path, test_name in test_data:
                save_transformed_img(test_path, (new_test_dir + '/'
                                                 + test_name))

    return [count_train, count_val, (count_total - count_train
                                     - count_val)]


def build_model():
    """Build the model"""
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            input_shape=(IMG_WIDTH, IMG_HEIGHT,
                                         IMG_CHANNELS),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(N_CLASS, activation='softmax'))
    
    return model


def load_trained_model(weights_path):
    """Load previously trained model
    
    Parameters
    ----------
    weights_path : str
    
    Return
    ------
    model : keras.models.Sequential
    """
    model = build_model()
    model.load_weights(weights_path)
    return model


if __name__ == '__main__':
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.layers.convolutional import Convolution2D
    from keras.layers.convolutional import MaxPooling2D
    from keras.preprocessing.image import ImageDataGenerator
    
    # Settings
    data_dir = 'data_fcbdata/'
    train_dir = data_dir + 'train/'
    validation_dir = data_dir + 'validation/'
    test_dir = data_dir + 'test/'
    
    # fix random seed for reproducibility
    seed = 170
    seed = np.random.seed(seed)
    
    # Iterate for further averaging the results obtained
    run = 0
    best_acc = 0
    best_model = None
    acc = np.zeros(N_FOLD)
    for run in range(N_FOLD):
        # Reset data partitions
        seed = np.random.seed(seed)
        [n_train, n_val, n_test] = create_dataset(train_frac=0.8,
                                                  val_frac=0.1,
                                                  test_frac=0.1)
        
        # build the model
        model = build_model()
        
        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        
        # Create data sets and apply data augmentation
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                # shear_range=0.2,
                # zoom_range=0.2,
                # rotation_range=15,
                # width_shift_range=0.2,
                # height_shift_range=0.2,
                horizontal_flip=True)
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=64, seed=seed)
        
        validation_generator = validation_datagen.flow_from_directory(
                validation_dir,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=64, seed=seed)
        
        # Train
        model.fit_generator(
                train_generator,
                nb_epoch=N_EPOCH,
                validation_data=validation_generator,
                nb_val_samples=n_val,
                samples_per_epoch=n_train)
        
        # Test
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=64, seed=seed)
    
        # Final evaluation of the model
        result = model.evaluate_generator(test_generator,
                                          val_samples=n_test)
        accuracy = result[1]
        acc[run] = accuracy
        if accuracy > best_acc:
            best_model = model
            best_acc = accuracy
        # model.save('challenge_model' + str(run) + '.h5')
            
    # Save model
    # best_model.save('Best_challenge_model.h5')
    # Print restuls
    print('Best ACC: ' + str(best_acc))
    print('Average ACC: ' + str(np.mean(acc)))
    print(acc)

# EOF
