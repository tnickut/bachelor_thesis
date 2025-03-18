This repository contains the implementation for the software part of the bachelor thesis "Development and Integration of a Side-Assist System for OpenPilot Using Monocular Vision"

Prerequisites:

1. Option manual installation
   1. Install Python (version 3.7.12, website: [Python 3.7.12](https://www.python.org/downloads/release/python-3712/))
   2. Install dependencies via `pip install -r requirements.txt`

2. Option: Use Anaconda. Execute the following command:
    ‘conda create --name myenv python=3.7.12 --file requirements.txt’

Sciebo link for big files: https://uni-muenster.sciebo.de/s/6YxYy9WgCzRjrsQ

The repo offers the following functionalities
- Training the data set:
    The training can be started by ‘python train.py’. The images and labels from train_images and train_labels are used. In the folder is currently only a subset, the whole part of the dataset can be downloaded from sciebo. The training on the whole training subset takes about 23 hours, the resulting model is saved as ‘orientation_model.h5’. It can also be downloaded from sciebo.

- Visualisation of the MultiBin pipeline using KITTI images:
    In the script, set the variable ‘image’ to the corresponding image number, e.g. ‘003744’, ‘003745’, etc. . If necessary, adjust the path to ‘train_images’ or ‘test_images’. In the current folder in only a subset of the train and test images for visualisation, the rest can also be downloaded from sciebo. Then start the script via ‘python demo_kitti.py’.

- Visualisation of the MultiBin pipeline using GTA5 scenes:
    Change the variable ‘image’ in the script to a desired image from the ‘gta_images’ folder. Then start the script via ‘python demo_gta5.py’.

- Create predictions in KTTI format using the KITTI test data set
    Execute the script ‘create_kitti_predictions.py’. The results are stored in the kitti_predictions folder. Not all test images are stored in the repository. If necessary, the subset can be downloaded from sciebo. The images from 0 to 3740 are for training, the rest for testing.

- Evaluation of orientation_score, distance and average precision
    Using the predictions in the ‘kitti_predictions’ folder and the eval_*.py scripts, the metrics orientation score, error in the distance to the center of the bounding box and AP can be determined. The predictions must be created beforehand using ‘create_kitti_predictions.py’, but the folder already contains all the predictions that were created based on the model. The evaluation of the respective metric can be done either via ‘python eval_distance.py’, ‘python eval_map.py’ or ‘python eval_orientation_score.py’.

- Starting and testing the Side Assist system
    1. start the Side Assist via ‘python side_assist.py’ and wait for the server to boot up
    2. set the video path in the file ‘side_assist_feeder.py’ to an example video in ‘traffic_videos_gta5’. Also set recording position to left or right. Then start via ‘python side_assist_feeder.py’ in another terminal. Whether a warning is issued can then be recognised via port 8080, but the status is also displayed in the console after each recorded frame. A warning for a particular side is represented by the fact that either ‘occupied_left’ or ‘occupied_right’ is marked as true.
