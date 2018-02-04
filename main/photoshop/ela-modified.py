from PIL import Image, ImageChops
import numpy as np
import cv2
import os
import glob

cwd = os.getcwd()

original_dir = cwd+"/Images/source"
destination = cwd+"/Images/ela"

SCALE = 15

for i in glob.glob(original_dir+"/*.jpg"):
    print(os.path.basename(i))
    ORIG = i
    TEMP = i+"-temp.jpg"
    original = Image.open(ORIG)
    original.save(TEMP, quality=90)
    temporary = Image.open(TEMP)
    print("test")
    diff = ImageChops.difference(original, temporary)
    d = diff.load()
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])
    diff.show()
    diff.save(destination+"/"+os.path.basename(i))