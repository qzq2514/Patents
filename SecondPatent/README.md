# Plate recognition using landmark detection and CTC loss

## introduction:
This program can detect 4 coordinates of 4 plate corners, after projection transformation,  
the plate can be recognized by CNN net trained by CTC loss without segmentation.
On plate pictures of CA states of USA,the program can achiece accuarcy of 96.65%.

## CNN models
model net:  
1.Landmark net :get coordinates of 4 plate corners(landmark.png for details)    
2.Plate Recognition net: recognize plate using CTC loss (CTC_rec.png for details)  

pipline:  
Plate recognition (pipline.png for details)

## training data:  
|   Landmark net  | Plate Recognition net|
|:------------:|:-------------------:|
| ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/FirstPaten/char_table.jpg)    |       ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/FirstPaten/not_table.jpg)        |


## visible result:  
|   Landmark net  | Plate Recognition net|
|:------------:|:-------------------:|

