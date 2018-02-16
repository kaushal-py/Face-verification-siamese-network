import numpy as np
import cv2
import os

class Preprocess:

    def __init__(self, haarcascade, patch):

        self.eye_cascade = cv2.CascadeClassifier(haarcascade)
        self.patch = cv2.imread(patch)

    def add_eyeptach(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        eyes = self.eye_cascade.detectMultiScale(gray)

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
                return None

            (ex,ey,ew,eh) = correct_eye

            roi = img[ey-15:ey+eh+15, ex-15:ex+ew+15]

            # Now create a mask of patch and create its inverse mask also
            patch_resized = cv2.resize(self.patch, (ew+30, eh+30), interpolation = cv2.INTER_AREA)

            if (ex + (ex+ew))/2 > 112:
                patch_resized = cv2.flip( patch_resized, 1 )

            patch2gray = cv2.cvtColor(patch_resized,cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(patch2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            img_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
            patch_fg = cv2.bitwise_and(patch_resized,patch_resized,mask = mask)

            dst = cv2.add(img_bg,patch_fg)

            img[ey-15:ey+eh+15, ex-15:ex+ew+15] = dst
            
            return img

        else:
            return None

    def subtract_backgroud(self, img_path):
        
        img = cv2.imread(img_path)

        img =    cv2.resize(img, (225, 225))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((3,3),np.uint8)

        edges = cv2.Canny(img,0,150)
        edges = cv2.dilate(edges,kernel,iterations = 1)

        # Threshold.
        # Set values equal to or above 220 to 0.
        # Set values below 220 to 255.
                
        # Copy the thresholded image.
        im_floodfill = edges.copy()
        
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = edges.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (10,0), 255);
        cv2.floodFill(im_floodfill, mask, (w-10,0), 255);
        
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        
        # Combine the two images to get the foreground.
        im_out = edges | im_floodfill_inv

        img_bg = cv2.bitwise_and(img,img,mask = im_out)

        return img_bg

if __name__ == "__main__":
    p = Preprocess("haarcascades_eye.xml", "patch.jpg")
    out = p.subtract_backgroud("/mnt/Data/Code/deep-blue/EYC3PDBS3/main/.eycdata/pre/Person-F240445/PRE_F240445_23_21_R.jpg")

    out_sub = p.add_eyeptach(out)

    cv2.imshow('img',out_sub)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()