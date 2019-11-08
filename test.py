import os
import numpy as np
import cv2
from tqdm import tqdm

EPIDURAL = './data/epidural/'

if __name__ == '__main__':
    for i in os.listdir(EPIDURAL):
        img = cv2.imread(EPIDURAL + i)
        print(np.amin(img))
        print(np.amax(img))
        print()
