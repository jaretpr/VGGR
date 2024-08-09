# VGGR
<img src='https://raw.githubusercontent.com/m4cit/Deep-Learning-Quran-Recognition/main/gallery/icon.png' align="left" height="200">
VGGR (Video-Game Genre Recognition) is a Deep-Learning Image Classification project. The training, validation, and test datasets consist of gameplay images, as well as different augmentations.

## Requirements
1. Install Python **3.10** or newer.
2. Install [PyTorch](https://pytorch.org/get-started/locally/) (I used 2.2.2 + cu121)
3. Install the required packages by running `pip install -r requirements.txt` in your shell of choice. Make sure you are in the project directory.
4. Unzip the test set located in *data/test/*.


## Usage
**Example 1:**
>```
>python DLQR.py --demo
>```
\
\
**Example 2:**
>```
>python DLQR.py --predict -t reciter -i .\path\to_some\file.mp3 -dev cpu
>```
or
>```
>python DLQR.py --predict --target reciter --input .\path\to_some\file.mp3 --device cuda
>```
\
\
**Example 3:**
>```
>python DLQR.py --train -m cnn_reciter -dev cuda
>```
or
>```
>python DLQR.py --train --model cnn_reciter --device cpu
>```


The input audio file should be 15 seconds long. If it's longer, it will be trimmed before predicting.

You can predict with the included pre-trained models (currently one model), and re-train if needed.

Delete the existing model to train from scratch (both options require training data).


## Performance
<img src='https://raw.githubusercontent.com/m4cit/Deep-Learning-Quran-Recognition/main/gallery/demo_test_set.png' width="900">

The train test data ratio isn't high enough but nevertheless, there are some observations worth mentioning. The image above suggests that 50% (5/10) of the unseen data is being recognized / predicted correctly, and that the accuracy between different reciters is not consistent.

As for the seen data, over 90% (10/11) is being predicted correctly. Adding low and high intensity noise (somewhat simulating re-recordings via microphone) to one of the samples made no difference (15, 16). As mentioned, most files contain a portion in the beginning which seems to affect results (11, 12). These portions were removed before training.


## Data



## Preprocessing



## Libraries
* [PyTorch](https://pytorch.org/) and its dependencies
* [tqdm](https://tqdm.github.io/)
* [pandas](https://pandas.pydata.org/)

