Face Recognition with Triplet Loss trained on Inception v3 architecture.


Dataset: This repository is for the Labelled Faces in the Wild (LFW) available on Kaggle. This dataset needs to be downloaded and the images folder put in ‘lfw-dataset/lfw-deepfunneled’. It also uses the metadata files which need to be put in ‘lfw-dataset/lfw-metadata’.

Description of Scripts: This project uses the tf.estimator API in tensorflow.
1.	train.py – Uses tf.estimator model functions {model_fn, inception_v3_model_fn}, input functions {train_input_fn, test_input_fn } and {Params} from model.utils to set up the training and testing interface.
2.	model/inception_v3_model_fn.py – Downloads the inception v3 architecture from tensorflow hub. Adjusts the images to required input shape and outputs the required dimension of the embeddings as in params. Imports triplet loss and sets up the training, prediciting and testing routines.
3.	model/lfw_dataset.py – Reads the image paths label wise from the directory, splits it into train and test sets and creates a tensorflow dataset interface to be accesses by input_fn.py.
4.	model/input_fn.py – Gets the dataset from lfw_dataset.py, prepares to be fed into the model for training and testing purposes.
5.	model/triplet_loss.py – returns the triplet loss from samples by using an online (while training batch-wise) triplet mining procedure as described in https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf . Performs both batch-all and batch-hard techniques and returns the fraction of positive triplets in batch-all case. Refer to code for more details.
6.	model/train-embeddings.py – Uses the embeddings and labels obtained from training the inception v3 on triplet loss to train a MLP Siamese neural network to verify two faces and saves it as verify_face.h5
7.	 model/face_mathcing_mlp.py – Uses verify_face.h5 to build a face verification application in python.


