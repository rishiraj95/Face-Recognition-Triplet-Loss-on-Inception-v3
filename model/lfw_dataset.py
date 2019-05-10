"""tf.data.Dataset interface to split the LFW data into train and test sets"""


from __future__ import print_function
import tensorflow as tf
import pandas as pd
import os




def load_and_decode_image(image_path):
    #Load image from path and normalize from [0.0,255.0] to [0.0,1.0]
    img_raw=tf.read_file(image_path)
    image=tf.image.decode_image(img_raw)
    image=tf.cast(image,tf.float32)
    image=image/255.0   
    return image



def get_dataset(data_dir, mode):
    """Get the train or test dataset.
    Args:
        data_dir: (string) path to the data directory
        mode: (string) either `train` or `test` to choose which dataset to return.
    """

    #Get dataset and metadata path.
    dataset_path=os.path.join(data_dir,'lfw-deepfunneled')
    metadata_path=os.path.join(data_dir, 'lfw-metadata')

    #'lfw_allnames.csv' contains number of images for each person
    lfw_allnames=pd.read_csv(os.path.join(metadata_path,'lfw_allnames.csv'))
    lfw_allnames_dict=dict(zip(lfw_allnames['name'],lfw_allnames['images']))
    
    images_train_path=[]
    images_test_path=[]
    labels_train=[]
    labels_test=[]
    label=0
    test_images_cntr=0

    #Get inside the directory, and choose first 100 people with more than 8 images
    #to form the test set.
    classes=sorted(os.walk(dataset_path).__next__()[1])

    for c in classes:
   
        if c in lfw_allnames_dict and lfw_allnames_dict[c]>8 and test_images_cntr<100:
            c_dir=os.path.join(dataset_path,c)
            c_files=os.walk(c_dir).__next__()
            sample_cntr=0
            test_images_cntr=test_images_cntr+1
            for sample in c_files[2]:
                if sample.endswith('jpg') or sample.endswith('jpeg'):
                    sample_cntr=sample_cntr+1;
                    if sample_cntr<=4:
                        images_test_path.append(os.path.join(c_dir,sample))
                        labels_test.append(label)
                    else:
                        images_train_path.append(os.path.join(c_dir,sample))
                        labels_train.append(label)
            label=label+1

        elif c in lfw_allnames_dict and lfw_allnames_dict[c]>1:
                c_dir=os.path.join(dataset_path,c)
                c_files=os.walk(c_dir).__next__()
                for sample in c_files[2]:
                    if sample.endswith('jpg') or sample.endswith('jpeg'):
                        images_train_path.append(os.path.join(c_dir,sample))
                        labels_train.append(label)
                label=label+1

        
        
    #Make tf Dataset for the train and test images and labels
    train_images_ds=tf.data.Dataset.from_tensor_slices(images_train_path)
    train_images_ds=train_images_ds.map(load_and_decode_image)
    train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels_train, tf.int32))


    test_images_ds=tf.data.Dataset.from_tensor_slices(images_test_path)
    test_images_ds=test_images_ds.map(load_and_decode_image)
    test_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels_test, tf.int32))


    
    if mode=='train':
        return tf.data.Dataset.zip((train_images_ds,train_label_ds))

    elif mode=='test':
        return tf.data.Dataset.zip((test_images_ds,test_label_ds))

    else:
        raise ValueError("mode must be either 'train' or 'test'")

        







            

