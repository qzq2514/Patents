# Plate recognition using landmark detection and CTC loss

## introduction:
This program can detect 4 coordinates of 4 plate corners, after projection transformation,  
the plate can be recognized by CNN net trained by CTC loss without segmentation.
On plate pictures of CA states of USA,the program can achiece accuarcy of 96.65%.

## CNN models
model net:  
1.Landmark net :get coordinates of 4 plate corners(landmark.png and [code](https://github.com/qzq2514/DNNCode/tree/master/textRecognition/DWCNN_CTCLoss_plateRec) for details)    
2.Plate Recognition net:   
recognize plate using CTC loss (CTC_rec.png and [code](https://github.com/qzq2514/Patents/tree/master/SecondPatent/train_landmark) for details)  

pipline:  
Plate recognition (pipline.png for details)

## training data:  
|   Landmark net  | Plate Recognition net|
|:------------:|:-------------------:|
| ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/SecondPatent/landmark-org1.png)    |       ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/SecondPatent/CTCRec-org1.png)        |
| ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/SecondPatent/landmark-org2.png)    |       ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/SecondPatent/CTCRec-org2.png)        |
| ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/SecondPatent/landmark-org3.png)    |       ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/SecondPatent/CTCRec-org3.png)        |


## visible result:  
|   Landmark net  | Plate Recognition net|
|:------------:|:-------------------:|
| ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/SecondPatent/landmark-res1.png)    |       ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/SecondPatent/CTC_rec_res1.png)        |
| ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/SecondPatent/landmark-res2.png)    |       ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/SecondPatent/CTC_rec_res2.png)        |
