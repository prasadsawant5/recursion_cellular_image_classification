import numpy as np
import cv2
import os
from tqdm import tqdm
import modin.pandas as pd
import pydicom

os.environ["MODIN_ENGINE"] = "dask"

TRAIN = './stage_1_train_images/'
TRAIN_CSV = './stage_1_train.csv'
DATA = './data/'
NONE = 'none/'
TOTAL_LENGTH = 674258

CLASSES = ['subdural', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid' ]

def convert_to_jpeg(dcm_path, jpeg_path):
    try:
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array

        if img.all() != None and not os.path.exists(jpeg_path):
            cv2.imwrite(jpeg_path, img)
    except Exception as e:
        print(e)
        print(TRAIN + f_name)

if __name__ == '__main__':
    labels_freq = dict()
    none_dict = dict()

    csv = pd.read_csv(TRAIN_CSV)

    if not os.path.exists(DATA):
        os.mkdir(DATA)
        os.mkdir(DATA + NONE)

    for c in CLASSES:
        if not os.path.exists(DATA + c):
            os.mkdir(DATA + c)

    with tqdm(total=len(csv)) as pbar:
        for index, row in csv.iterrows():
            f_name = row['ID'].split('_')[0] + '_' + row['ID'].split('_')[1]
            label = row['ID'].split('_')[-1]
            probability = float(row['Label'])

            if label == 'any':
                continue

            is_any_class = False

            for c in CLASSES:
                if c == label and probability > 0.0:
                    is_any_class = True
                    convert_to_jpeg(TRAIN + f_name + '.dcm', DATA + label + '/' + f_name + '.jpg')

            if not is_any_class:
                convert_to_jpeg(TRAIN + f_name + '.dcm', DATA +NONE + f_name + '.jpg')
                

            pbar.update(1)