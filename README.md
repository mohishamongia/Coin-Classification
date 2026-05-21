# Coin Classification using Deep Learning

An image classification project built with TensorFlow and Keras to accurately identify and categorize global coins into 211 distinct classes.


 The project includes training a Convolutional Neural Network (CNN) to classify images of coins. It utilizes transfer learning (InceptionV3) and extensive data augmentation to achieve high accuracy on a diverse dataset.

Technologies Used
* **Python 3**
* **TensorFlow / Keras** (Deep Learning framework)
* **NumPy & Pandas** (Data manipulation)
* **Matplotlib & PIL** (Image visualization and processing)

Dataset
The model is trained on the [Kaggle Coin Images Dataset](https://www.kaggle.com/datasets). 
* **Training Images:** 6,413
* **Validation Images:** 844
* **Testing Images:** 844
* **Total Classes:** 211 unique coin types

The dataset relies on a `cat_to_name.json` mapping file to translate numeric directory labels into human-readable coin names (e.g., "1 Kurus, Turkish Lira, Turkey").

Model Architecture & Training
1. **Data Preprocessing:** Images are resized to `224x224`. We apply data augmentation techniques using `ImageDataGenerator`, including random rotations, shifts, flips, and sample-wise standard normalization to improve model generalization.
2. **Transfer Learning:** The model utilizes pre-trained weights to speed up training and improve feature extraction. 
3. **Callbacks:** * `ModelCheckpoint`: Saves the best model weights during training.
   * `ReduceLROnPlateau`: Dynamically reduces the learning rate if validation loss plateaus.
   * `EarlyStopping`: Halts training if the model stops improving to prevent overfitting.
