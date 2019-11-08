import os
from tqdm import tqdm
import im2rec
from shutil import copyfile, rmtree
import random

CLASSES = [ 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural' ]
NONE = 'none'
DATA = './data/'

if __name__ == '__main__':
    no_hemorrhage = os.listdir(DATA + NONE)

    for c in CLASSES:
        if not os.path.exists('./rec'):
            os.mkdir('./rec')

        if not os.path.exists('./rec/' + NONE):
            os.mkdir('./rec/' + NONE)

        if not os.path.exists('./rec/' + c):
            os.mkdir('./rec/' + c)
        
        total_class_images = len(os.listdir(DATA + c))
        none_image_set = set()

        for img in tqdm(os.listdir(DATA + c)):
            copyfile(DATA + c + '/' + img, './rec/' + c + '/' + img)

            while True:
                no_hemorrhage_img = random.choice(no_hemorrhage)

                if no_hemorrhage_img not in none_image_set:
                    none_image_set.add(no_hemorrhage_img)

                    copyfile(DATA + NONE + '/' + no_hemorrhage_img, './rec/' + NONE + '/' + no_hemorrhage_img)

                    break

        os.system('python im2rec.py ./rec/' + c + '_rec ./rec/ --recursive --list --num-thread 8')
        os.system('python im2rec.py ./rec/' + c + '_rec ./rec/ --recursive --pass-through --pack-label --num-thread 8')

        try:
            rmtree(os.path.join('./rec', c))
            rmtree(os.path.join('./rec', NONE))
        except OSError as err:
            print(err)

