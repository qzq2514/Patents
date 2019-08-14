import os
import cv2

save_img_dir="D:/forTensorflow/plateLandmarkDetTrain2/CA/images"
save_label_dir="D:/forTensorflow/plateLandmarkDetTrain2/CA/labels"

for image_file_name in os.listdir(save_img_dir):
    image_file_path=os.path.join(save_img_dir,image_file_name)
    image = cv2.imread(image_file_path)

    label_file_path=image_file_path.replace("images","labels").replace("jpg","json")
    with open(label_file_path,"r") as fr:
        postions=eval(fr.read())
        for posId in range(0, 8, 2):
            cv2.circle(image, (postions[posId], postions[posId + 1]), 2, (0, 0, 255), 2)

    cv2.imshow("image",image)
    cv2.waitKey(0)