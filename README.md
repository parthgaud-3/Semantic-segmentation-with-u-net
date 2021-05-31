Semantic_Segmentation_With_U-Net
Table of Contents:
Overview of Project

Data Description

Libraries used

Steps followed

Conclusion

How to replicate on your device

Overview of Project:
We are given Helen Dataset which contains images of faces of different persons. Our target is to classify each pixel as

bg (background)

face

lb (left brow)

rb (right brow)

le (left eye)

re (right eye)

nose

ulip

imouth

llip

hair

For this task we will be using the famous U-Net Architecture. U-Net Paper.

Data Description:
For this project , the Helen Dataset used can be downloaded from Helen Dataset.
For each image. It has 3 types of files. One is image.jpg which has the file which will be loaded to the model. Second is the label.png file which has all pixel by pixel classications of the image. The viz.jpg file is just for demonstation purpose and is not of any use to the model.
Following is the directory structure

.
└── helenstar_release
     ├── train
     │   ├── image.jpg
     │   ├── label.png
     ├   ├── viz.jpg
     │   └── ... (1999 sets of 3 images i.e 5997 images total)           
     └── test
         ├── image.jpg
         ├── label.png
         ├── viz.jpg
         └── ... (100 sets of 3 images i.e 300 images total)
Total number of images in dataset : 2099
Number of images in train set : 1999
Number of images in test set : 100

For convenience I will be performing some shiftings to put all image.jpg files in one folders , label.png in other. I will be doing this using the shutil module of python

The final directory strucutre will be as follows

.
└── splitted_Data
        ├── train
        │   ├── images
        │   │   ├── image1.jpg
        │   │   ├── image2.jpg
        │   │   └── ... (1999 files)
        │   └── labels
        │       ├── label1.jpg
        │       ├── label2.jpg
        │       └── ... (1999 files)       
        │           
        └── test
            ├── images
            │   ├── image1.jpg
            │   ├── image2.jpg
            │   └── ... (100 files)
            └── labels
                ├── label1.jpg
                ├── label2.jpg
                └── ... (100 files)
Sample images and their corresponding labels
 

 

 

Libraries used:
Numpy

Matplotlib

torch

torchvision

PIL

os module of python

tqdm

shutil

Steps Followed
1. Importing Necessary Libraries
Getting all the required python libraries required for the implementation of the project

2. Looking at directory structure and making desired shiftings
As shown above, the directory strucutre in the link is changed so that it is easy to execute in the later part

3. Data Preprocessing
Now all the images have different dimensions. But to feed them into the model, all the images need to be of the same size. I resized all the images to 256x256. Also when we load images , they are usually loaded in the form of numpy array with dtype = uint8 . They need to be converted to tensors with dtype = torch.float32

4. Defining Train and Test Dataloaders
Now the train dataset has 1999 images which can not be fed in one go. I use Mini Batch Gradient Descent. with a batch size of 10.

5. Defining model
The model architecture is shown in the picture below 

It has a encoding path and a decoding path. The architecture is difficult to code in one class UNet(nn.Module. So I define some classes before hand which can help to make our code concise and simple to read

The input to the model is of shape 10x3x256x256 and output is 10x11x256x256

6. Defining Dice Loss and optimizer
For this problem we will be defining the DiceLoss. As there is not pre defined Diceloss in pytorch, We will be defining it on our own. The code is inspired from An overview of semantic image segmentation.

7. Performing Forward Propagation
I perform forward propagation for 30 epochs and print losses. Based on the trend of losses, I have occationally interrupted execution and reduced learning rate

8. Visualizing train loss


9. Visualizing predictions
 

 

 


These are predictions on train set, We'll see predictions on test sets in conclusion part

Conclusion
The f1 scores calculated are {'bg': ([0], [0]), 'face': ([1], [1]), 'lb': ([2], [2]), 'rb': ([3], [3]), 'le': ([4], [4]), 're': ([5], [5]), 'nose': ([6], [6]), 'ulip': ([7], [7]), 'imouth': ([8], [8]), 'llip': ([9], [9]), 'hair': ([10], [10]), 'eyes': ([4, 5], [4, 5]), 'brows': ([2, 3], [2, 3]), 'mouth': ([7, 8, 9], [7, 8, 9]), 'overall': ([4, 5, 2, 3, 6, 7, 8, 9], [4, 5, 2, 3, 6, 7, 8, 9])}
f1_bg=0.9240500079001422
f1_face=0.8405945882784983
f1_lb=0.7005460539872244
f1_rb=0.7301984325496081
f1_le=0.8028062022431744
f1_re=0.8269042316258352
f1_nose=0.8508512952562765
f1_ulip=0.682443588332416
f1_imouth=0.6923525599481529
f1_llip=0.7343509096515572
f1_hair=0.64483539303235
f1_eyes=0.814874300118227
f1_brows=0.7150763307756063
f1_mouth=0.8081917367197281
f1_overall=0.8115612496993813

Let's see some predictions
 

 

 

How to replicate on your device
Just store the data as per the directory structures shown in the code. Run the code You can get the pre trained weights for the model at this link Pre-Trained weights.
