from keras.utils import image_dataset_from_directory
from config import batch_size, image_size, validation_split
import tensorflow as tf
from config import train_directory, test_directory, image_size, batch_size, validation_split
from sklearn.model_selection import train_test_split
import os
import tempfile
import shutil

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
    
    def create_split_datasets():
        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp()
        splits = ['train', 'validation', 'test']
        categories = ['all', 'hem']
        
        # Create directory structure
        for split in splits:
            for category in categories:
                os.makedirs(os.path.join(temp_dir, split, category), exist_ok=True)

        # Split and copy files for each category
        for category in categories:
            # Get all files in category
            category_path = os.path.join(base_path, category)
            files = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # First split: 80% train+val, 20% test
            train_val_files, test_files = train_test_split(files, test_size=0.2, random_state=47)
            
            # Second split: 80% train, 20% validation (from train_val)
            train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=47)
            
            # Copy files to respective directories
            for file in train_files:
                shutil.copy2(
                    os.path.join(category_path, file),
                    os.path.join(temp_dir, 'train', category, file)
                )
            
            for file in val_files:
                shutil.copy2(
                    os.path.join(category_path, file),
                    os.path.join(temp_dir, 'validation', category, file)
                )
            
            for file in test_files:
                shutil.copy2(
                    os.path.join(category_path, file),
                    os.path.join(temp_dir, 'test', category, file)
                )
        
        return temp_dir
    
    # Create split datasets in temporary directory
    temp_dir = create_split_datasets()
    
    # Create datasets using the split data
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(temp_dir, 'train'),
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        seed=47,
        shuffle=True
    )
    
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(temp_dir, 'validation'),
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        seed=47,
        shuffle=False
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(temp_dir, 'test'),
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        seed=47,
        shuffle=False
    )
    
    # Configure for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)

    return train_dataset, validation_dataset, test_dataset