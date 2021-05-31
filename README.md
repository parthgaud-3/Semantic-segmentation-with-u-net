# Semantic_Segmentation_With_U-Net

## Table of Contents: 
* Overview of Project

* Data Description 
* Libraries used

* Steps followed

* Conclusion
* How to replicate on your device



## Overview of Project:

We are given Helen Dataset which contains images of faces of different persons. Our target is to classify each pixel as 
* bg   (background)

* face 
* lb  (left brow)

* rb (right brow)
* le  (left eye)
  
* re (right eye)

* nose
* ulip
* imouth
* llip
* hair 

For this task we will be using the famous U-Net Architecture. <a href='https://arxiv.org/abs/1505.04597'>U-Net Paper</a>.

## Data Description:   
For this project , the Helen Dataset used can be downloaded from <a href='https://drive.google.com/file/d/1jweX1u0vltv-tYZhYp6mlyDZDy0aDyrw/view?usp=sharing'>Helen Dataset</a>.<br>
For each image. It has 3 types of files. One is <tt>image.jpg</tt> which has the file which will be loaded to the model. Second is the <tt>label.png</tt> file which has all pixel by pixel classications of the image. The <tt>viz.jpg</tt> file is just for demonstation purpose and is not of any use to the model.<br>
Following is the directory structure 


<pre>.
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
         └── ... (100 sets of 3 images i.e 300 images total)</pre>

Total number of images in dataset : 2099<br>
Number of images in train set : 1999<br>
Number of images in test set : 100


For convenience I will be performing some shiftings to put all image.jpg files in one folders , label.png in other. I will be doing this using the <tt>shutil</tt> module of python

The final directory strucutre will be as follows
<pre>.
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
                └── ... (100 files)</pre>
                
Sample images and their corresponding labels <br>
![](images/image1.png)
![](images/label1.png)

![](images/image2.png)
![](images/label2.png)

![](images/image3.png)
![](images/label3.png)


## Libraries used:
* Numpy

* Matplotlib

* torch
* torchvision<br>
  
* PIL

* os module of python
* tqdm
* shutil

## Steps Followed

* ### 1. Importing Necessary Libraries
Getting all the required python libraries required for the implementation of the project

 
* ### 2. Looking at directory structure and making desired shiftings
As shown above, the directory strucutre in the link is changed so that it is easy to execute in the later part


* ### 3. Data Preprocessing
Now all the images have different dimensions. But to feed them into the model, all the images need to be of the same size. I resized all the images to 256x256. Also when we load images , they are usually loaded in the form of numpy array with <tt>dtype = uint8</tt> . They need to be converted to tensors with <tt>dtype = torch.float32</tt>

* ### 4. Defining Train and Test Dataloaders
Now the train dataset has 1999 images which can not be fed in one go. I use <a href='https://www.youtube.com/watch?v=4qJaSmvhxi8'>Mini Batch Gradient Descent</a>. with a batch size of 10.


* ### 5. Defining model
The model architecture is shown in the picture below
![](Model_arch.png)

It has a encoding path and a decoding path. The architecture is difficult to code in one <tt>class UNet(nn.Module</tt>. So I define some classes before hand which can help to make our code concise and simple to read

The input to the model is of shape 10x3x256x256 and output is 10x11x256x256


* ### 6. Defining Dice Loss and optimizer
For this problem we will be defining the <tt>DiceLoss</tt>. As there is not pre defined Diceloss in pytorch, We will be defining it on our own. The code is inspired from  <a href='https://www.jeremyjordan.me/semantic-segmentation/'>An overview of semantic image segmentation</a>.

* ### 7. Performing Forward Propagation
I perform forward propagation for 30 epochs and print losses. Based on the trend of losses, I have occationally interrupted execution and reduced learning rate



* ### 8. Visualizing train loss
![](train_losses.png)


* ### 9. Visualizing predictions



![](images/label1.png)
![](images/pred1.png)


![](images/label2.png)
![](images/pred2.png)

![](images/label3.png)
![](images/pred3.png)
 
 <br>

These are predictions on train set, We'll see predictions on test sets in conclusion part

## Conclusion
The f1 scores calculated are
{'bg': ([0], [0]), 'face': ([1], [1]), 'lb': ([2], [2]), 'rb': ([3], [3]), 'le': ([4], [4]), 're': ([5], [5]), 'nose': ([6], [6]), 'ulip': ([7], [7]), 'imouth': ([8], [8]), 'llip': ([9], [9]), 'hair': ([10], [10]), 'eyes': ([4, 5], [4, 5]), 'brows': ([2, 3], [2, 3]), 'mouth': ([7, 8, 9], [7, 8, 9]), 'overall': ([4, 5, 2, 3, 6, 7, 8, 9], [4, 5, 2, 3, 6, 7, 8, 9])} <br>
f1_bg=0.9240500079001422 <br>
f1_face=0.8405945882784983 <br>
f1_lb=0.7005460539872244 <br>
f1_rb=0.7301984325496081 <br>
f1_le=0.8028062022431744 <br>
f1_re=0.8269042316258352 <br>
f1_nose=0.8508512952562765 <br>
f1_ulip=0.682443588332416 <br>
f1_imouth=0.6923525599481529 <br>
f1_llip=0.7343509096515572 <br>
f1_hair=0.64483539303235 <br>
f1_eyes=0.814874300118227 <br>
f1_brows=0.7150763307756063 <br>
f1_mouth=0.8081917367197281 <br>
f1_overall=0.8115612496993813 <br>


Let's see some predictions<br>
![](images/test_label1.png)
![](images/test_pred1.png)


![](images/test_label2.png)
![](images/test_pred2.png)

![](images/test_label3.png)
![](images/test_pred3.png)

## How to replicate on your device

Just store the data as per the directory structures shown in the code. Run the code
You can get the pre trained weights for the model at this link  <a href='https://drive.google.com/file/d/1-2PIMuSKIFD4cqLi-Tve7v4jZ_VgnCoO/view?usp=sharing'>Pre-Trained weights</a>. 


