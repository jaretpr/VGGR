# VGGR
<img src='https://raw.githubusercontent.com/m4cit/Deep-Learning-Quran-Recognition/main/gallery/icon.png' align="left" height="200">

VGGR (Video-Game Genre Recognition) is a Deep-Learning Image Classification project. The training, validation, and test datasets consist of gameplay images, as well as different augmentations.


## Requirements
1. Install Python **3.10** or newer.
2. Install [PyTorch](https://pytorch.org/get-started/locally/) (I used 2.2.2 + cu121)
3. Install the required packages by running `pip install -r requirements.txt` in your shell of choice. Make sure you are in the project directory.
4. Unzip the test set located in *data/test/*.


## Performance
There are currently three Convolutional Neural Network (CNN) models available:

1. *cnn_v1* | F-score of **70.83 %**
2. *cnn_v2* | F-score of **58.33 %**
3. *cnn_v3* | F-score of **64.58 %**

<img src='https://raw.githubusercontent.com/m4cit/VGGR/main/gallery/perf_v1_1.png' align="left" width="900">
<img src='https://raw.githubusercontent.com/m4cit/VGGR/main/gallery/perf_v1_2.png' align="right" width="900">


## Usage
**Demo with Test Set:**
>```
>python VGGR.py --demo --model cnn_v1 --device cuda
>```
or
>```
>python VGGR.py --demo -m cnn_v1 -d cuda
>```
\
\
**Predict with Custom Input:**
>```
>python VGGR.py --predict -m cnn_v1 -d cuda -i path/to/img.png
>```
\
\
**Training:**
>```
>python VGGR.py --train -m cnn_v1 -d cuda
>```

- The -m (--model) command defaults to the *cnn_v1** model.
- The -d (--device) command defaults to *cpu*.
- You can predict with the included pre-trained models, and re-train if needed. Delete the existing model to train from scratch (both options require training data).


## Model Results (--demo)
The predictions are available as html files in the *results* folder, and also include the corresponding images.


## Data
Most of the images are from my own gameplay footage.
The PES 2012 and FIFA 10 images are from [No Commentary Gameplays](https://www.youtube.com/@NCGameplays) videos, and the FIFA 95 images are from a [10min Gameplay](https://www.youtube.com/@10minGameplay1) video (YouTube).

The training dataset also contains augmentations.


## Augmentation
To augment the training data with *jittering*, *inversion*, and *5 part cropping*, copy-paste the metadata of the images into the *augment.csv* file located in *data/train/metadata/*.
Then run `python VGGR.py --augment`


## Preprocessing
All images are originally 2560x1440p, and get resized to 1280x720p before training, validation, and inference. 


## Libraries
* [PyTorch](https://pytorch.org/) and its dependencies
* [tqdm](https://tqdm.github.io/)
* [pandas](https://pandas.pydata.org/)

