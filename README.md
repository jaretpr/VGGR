# VGGR
<img src='https://raw.githubusercontent.com/m4cit/VGGR/main/gallery/icon.png' align="left" height="250">

VGGR (Video Game Genre Recognition) is a Deep-Learning Image Classification project. The training, validation, and test datasets consist of gameplay images, as well as different augmentations.<br clear="left"/>


## Requirements
1. Install Python **3.10** or newer.
2. Install [PyTorch](https://pytorch.org/get-started/locally/) with
>`pip3 install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
3. Install the required packages by running `pip install -r requirements.txt` in your shell of choice.
4. Download the latest source code in [*releases*](https://github.com/m4cit/VGGR/releases), and unzip the train, test, and validation datasets inside the *data* folder.


**Note:** The provided training dataset does not contain augmentations.


## Performance
There are currently three Convolutional Neural Network (CNN) models available:

1. *cnn_v1* | F-score of **70.83 %**
2. *cnn_v2* | F-score of **58.33 %**
3. *cnn_v3* | F-score of **64.58 %**


### cnn_v1 --demo examples:
<img src='https://raw.githubusercontent.com/m4cit/VGGR/main/gallery/perf_v1_1.png' align="left" width="500">
<img src='https://raw.githubusercontent.com/m4cit/VGGR/main/gallery/perf_v1_2.png' align="right" width="470">
<br clear="left"/>
<br clear="right"/>


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

- The -m (--model) command defaults to the *cnn_v1* model.
- The --demo mode creates html files with the predictions and corresponding images inside the *results* folder.
- The -d (--device) command defaults to *cpu*.
- You can predict with the included pre-trained models, and re-train if needed. Delete the existing model to train from scratch (both options require training data).


## Data
Most of the images are from my own gameplay footage.
The PES 2012 and FIFA 10 images are from videos by [No Commentary Gameplays](https://www.youtube.com/@NCGameplays), and the FIFA 95 images are from a video by [10min Gameplay](https://www.youtube.com/@10minGameplay1) (YouTube).

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

