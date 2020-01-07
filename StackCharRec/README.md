# Stacked character recognition with CTC
model net: recognizing stacked character of plate(net.png for details)  
pipline:  Sliding window to segment characters(pipline.png for details)

## training data:  
|   char   |
|:------------:|
| ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/ThirtdPatent/stack_data.jpg)    |


## result:  
The model can achieve 98.2% accuarcy on stacked characters.
Using [Plate corners detection](https://github.com/qzq2514/Patents/tree/master/PlateLandmark_CTCRec) to detect four corners of plate and [MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) to detect characters(26 letters、10 numbers and a class for stacked characters),The plate with stacked characters can be recognized as following:    
## result:  


|   result   | result | result|
|:------------:|:-------------------:|:-------------------:|
| ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/ThirtdPatent/res1.PNG) |![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/ThirtdPatent/res2.PNG) |![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/ThirtdPatent/res3.PNG) |
| ![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/ThirtdPatent/res4.PNG) |![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/ThirtdPatent/res5.PNG) |![](https://github.com/qzq2514/ImageForGithubMakdown/blob/master/Patents/ThirtdPatent/res6.PNG) |

