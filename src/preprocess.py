from keras.utils import image_dataset_from_directory
from config import batch_size, image_size, validation_split
import tensorflow as tf
from config import train_directory, test_directory, image_size, batch_size, validation_split
import os


def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    return train_dataset, validation_dataset, test_dataset

def get_transfer_datasets():
    base_path = './kaggle/leukemia'
    
    # Define paths for train and test directories
    train_dir = os.path.join(base_path, 'train')
    test_dir = os.path.join(base_path, 'test')
    
    # Create training and validation datasets from the train directory
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='training',
        seed=47
    )
    
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='validation',
        seed=47
    )
    
    # Create test dataset from the test directory
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )
    
    # Configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, validation_dataset, test_dataset