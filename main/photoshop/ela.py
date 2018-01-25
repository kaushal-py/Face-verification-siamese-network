from PIL import Image, ImageChops
import numpy as np
import cv2

ORIG = 'post.jpg'
#ORIG = 'post-original.jpg'

TEMP = 'demo_new_ela.jpg'
SCALE = 15


def ELA():
    original = Image.open(ORIG)
    original.save(TEMP, quality=90)
    temporary = Image.open(TEMP)
    
    diff = ImageChops.difference(original, temporary)
    d = diff.load()
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])

    diff.show()
    # np.savetxt("diff",diff,newline=' ')
    # print(np.asarray(diff))

if __name__ == '__main__':
    ELA()
