import cv2
import numpy as np
import glob
import os
# cv2.imshow("image", im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

people = glob.glob("images/*")
count = 0
for person in people:
    
    images = glob.glob(person + "/*.jpg")
    for img_path in images:
        im = cv2.imread(img_path)
        resized_image = cv2.resize(im, (150, 150))
        cv2.imwrite(
                img_path,
                resized_image)   
        count += 1
        print("Done : ", count)
