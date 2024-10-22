# VGGR
<img src='https://raw.githubusercontent.com/m4cit/VGGR/main/gallery/icon.png' align="left" height="180">
Have you ever seen gameplay footage and wondered what kind of video game it is from? No? Well, don't wonder no more.
<br /><br />
VGGR (Video Game Genre Recognition) is a Deep-Learning Image Classification project, answering questions nobody is asking.
<br clear="left"/>


## Requirements
1. Install Python **3.10** or newer.
2. Install [PyTorch](https://pytorch.org/get-started/locally/) with
>`pip3 install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
3. Install the required packages by running `pip install -r requirements.txt` in your shell / terminal.
4. Download the latest source code and the train, test, and validation img zip-files in [*releases*](https://github.com/m4cit/VGGR/releases).
5. Unzip the train, test, and validation img files inside their respective folders located in _**./data/**_.

**Note:** The provided training dataset does not contain augmentations.


## Genres
The available genres are:
- Football / Soccer
- First Person Shooter (FPS)
- 2D Platformer
- Racing

## Games
### Train Set
- FIFA 06
- Call of Duty Black Ops
- Call of Duty Modern Warfare 3
- DuckTales Remastered
- Project CARS

### Test Set
- PES 2012
- FIFA 10
- Counter Strike 1.6
- Counter Strike 2
- Ori and the Blind Forest
- Dirt 3

### Validation Set
- Left 4 Dead 2
- Oddworld Abe's Oddysee
- FlatOut 2

## Usage
### Demo with Test Set
>```
>python VGGR.py --demo --model cnn_v1 --device cpu
>```
or
>```
>python VGGR.py --demo -m cnn_v1 -d cpu
>```
### Predict with Custom Input
>```
>python VGGR.py --predict -m cnn_v1 -d cpu -i path/to/img.png
>```
### Training
>```
>python VGGR.py --train -m cnn_v1 -d cpu
>```

<br />

- The -m (--model) command defaults to the best performing model.
- The -d (--device) command defaults to *cpu*.
- You can predict with the included pre-trained models, and re-train if needed. Delete the existing model to train from scratch.

## Results
The --demo mode creates html files with the predictions and corresponding images inside the _**results**_ folder.

## Performance
There are three Convolutional Neural Network (CNN) models available:

1. *cnn_v1* | F-score of **70.83 %**
2. *cnn_v2* | F-score of **58.33 %**
3. *cnn_v3* | F-score of **64.58 %**


### cnn_v1 --demo examples
<img src='https://raw.githubusercontent.com/m4cit/VGGR/main/gallery/perf_v1_1.png' width="500">
<img src='https://raw.githubusercontent.com/m4cit/VGGR/main/gallery/perf_v1_2.png' width="500">


## Data
Most of the images are from my own gameplay footage.
The PES 2012 and FIFA 10 images are from videos by [No Commentary Gameplays](https://www.youtube.com/@NCGameplays), and the FIFA 95 images are from a video by [10min Gameplay](https://www.youtube.com/@10minGameplay1) (YouTube).

The training dataset also contained augmentations (not in the provided zip-file.

### Augmentation
To augment the training data with *jittering*, *inversion*, and *5 part cropping*, copy-paste the metadata of the images into the *augment.csv* file located in _**./data/train/metadata/**_.

Then run `python VGGR.py --augment`.

The metadata of the resulting images are subsequently added to the _**metadata.csv**_ file.


### Preprocessing
All images are originally 2560x1440p, and get resized to 1280x720p before training, validation, and inference. 4:3 images are stretched to 16:9 to avoid black bars.


## Libraries
* [PyTorch](https://pytorch.org/) and its dependencies
* [tqdm](https://tqdm.github.io/)
* [pandas](https://pandas.pydata.org/)

