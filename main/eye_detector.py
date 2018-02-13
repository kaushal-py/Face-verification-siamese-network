import numpy as np
import cv2
import os

def main():
    eye_cascade = cv2.CascadeClassifier('haarcascades_eye.xml')

    imgs_path = '.eycdata/pre/'
    imgs_post_path = '.eycdata/post/'

    img_list = os.listdir(imgs_path)

    patch = cv2.imread('patch.jpg')

    try:
        os.mkdir('.eycdata/patch')
        os.mkdir('.eycdata/postbg')
    except:
        return

    for img_path in img_list:

        img_url = imgs_path + img_path + '/' + os.listdir(imgs_path + img_path)[0]
        img_post_url = imgs_post_path + img_path + '_/' + os.listdir(imgs_post_path + img_path+'_')[0]

        img = cv2.imread(img_url)
        img_post = cv2.imread(img_post_url)

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

            roi = img[ey-15:ey+eh+15, ex-15:ex+ew+15]

            # cv2.imshow('img',roi)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Now create a mask of patch and create its inverse mask also
            patch_resized = cv2.resize(patch, (ew+30, eh+30), interpolation = cv2.INTER_AREA)

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

            img[ey-15:ey+eh+15, ex-15:ex+ew+15] = dst
            hsv_post = cv2.cvtColor(img_post, cv2.COLOR_BGR2HSV)

            gray_post = cv2.cvtColor(img_post, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            kernel = np.ones((3,3),np.uint8)

            edges = cv2.Canny(img,0,150)
            edges_post = cv2.Canny(img_post,0,150)

            edges = cv2.dilate(edges,kernel,iterations = 1)
            edges_post = cv2.dilate(edges_post,kernel,iterations = 1)

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

            ####################################################
            im_floodfill = edges_post.copy()
            
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
            im_out_post = edges_post | im_floodfill_inv

            #######################################################
            img_bg = cv2.bitwise_and(img,img,mask = im_out)
            img_bg_post = cv2.bitwise_and(img_post,img_post,mask = im_out_post)

            gray_post = cv2.cvtColor(img_bg_post, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)


            img_bg_post = cv2.equalizeHist(gray_post)
            img_bg = cv2.equalizeHist(gray)

            # edges = cv2.fillPoly(edges, pts =contours, color=(255,255,255))
            # edges_post = cv2.fillPoly(edges_post, pts =contours_post, color=(255,255,255))
            # laplacian = cv2.Sobel(img_bg,cv2.CV_64F,1,0,ksize=3)
            # laplacian_post = cv2.Sobel(img_bg_post,cv2.CV_64F,1,0,ksize=3)

            # cv2.imshow('img',laplacian)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # cv2.imshow('img',laplacian_post)
            # key = cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # cv2.imshow('img',img_bg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # cv2.imshow('img',img_bg_post)
            # key = cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # if key == 27:
            #     break


            # cv2.rectangle(img,(ex-20,ey),(ex+ew,ey+eh),(0,255,0),2)
            print(".eycdata/patch/" + img_path)
            os.mkdir(".eycdata/patch/" + img_path)
            os.mkdir(".eycdata/postbg/" + img_path)
            
            cv2.imwrite(".eycdata/patch/" + img_path + "/" + os.listdir(imgs_path + img_path)[0], img_bg)
            # print(os.listdir(imgs_post_path + img_path))
            cv2.imwrite(".eycdata/postbg/" + img_path + '/' + os.listdir(imgs_post_path + img_path+'_')[0], img_bg_post)
        
main()