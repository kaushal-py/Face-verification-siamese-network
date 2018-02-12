import numpy as np
import cv2
import os

eye_cascade = cv2.CascadeClassifier('haarcascades_eye.xml')

imgs_path = '.eycdata/pre/'

img_list = os.listdir(imgs_path)

patch = cv2.imread('patch.jpg')


for img_path in img_list:

    img_url = imgs_path + img_path + '/' + os.listdir(imgs_path + img_path)[0]

    img = cv2.imread(img_url)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray)

    if len(eyes) > 0:

        eyes= sorted(eyes,key=lambda x:x[2], reverse=True)

        correct_eye = ()
        for eye in eyes:
            (ex,ey,ew,eh) = eye
            if (ex+ew+20 < 225 and ex-20 > 0 and ey+eh+20 < 225 and ey-20 > 0):
                correct_eye = eye
                break
        
        if len(correct_eye) == 0:
            print("No eye found")
            continue

        (ex,ey,ew,eh) = correct_eye

        roi = img[ey-10:ey+eh+10, ex-10:ex+ew+10]

        # cv2.imshow('img',roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Now create a mask of patch and create its inverse mask also
        patch_resized = cv2.resize(patch, (ew+20, eh+20), interpolation = cv2.INTER_AREA)

        if (ex+ew) > 112:
            patch_resized = cv2.flip( patch_resized, 1 )

        patch2gray = cv2.cvtColor(patch_resized,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(patch2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        patch_fg = cv2.bitwise_and(patch_resized,patch_resized,mask = mask)

        dst = cv2.add(img_bg,patch_fg)

        # cv2.imshow('img',dst)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        img[ey-10:ey+eh+10, ex-10:ex+ew+10] = dst

        # cv2.rectangle(img,(ex-20,ey),(ex+ew,ey+eh),(0,255,0),2)
        print(".eycdata/patch/" + img_path)
        os.mkdir(".eycdata/patch/" + img_path)
        
        cv2.imwrite(".eycdata/patch/" + img_path + "/" + os.listdir(imgs_path + img_path)[0], img)
