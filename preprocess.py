import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd
import pydicom

# os.environ["MODIN_ENGINE"] = "dask"

TRAIN = './stage_1_train_images/'
TRAIN_CSV = './stage_1_train.csv'
DATA = './data/'
NONE = 'none/'
TOTAL_LENGTH = 674258

CLASSES = [ 'subdural', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid' ]

subdural = set()
epidural = set()
intraparenchymal = set()
intraventricular = set()
subarachnoid = set()
none = set()

def convert_to_jpeg(dcm_path, jpeg_path):
    try:
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array

        if img.all() != None and not os.path.exists(jpeg_path):
            cv2.imwrite(jpeg_path, img)
    except Exception as e:
        print(e)
        print(TRAIN + f_name)
        print()
        print()

        print('subdural intersection none: {}'.format(str(none.intersection(subdural))))
        print('epidural intersection none: {}'.format(str(none.intersection(epidural))))
        print('intraparenchymal intersection none: {}'.format(str(none.intersection(intraparenchymal))))
        print('intraventricular intersection none: {}'.format(str(none.intersection(intraventricular))))
        print('subarachnoid intersection none: {}'.format(str(none.intersection(subarachnoid))))

if __name__ == '__main__':
    labels_freq = dict()
    none_dict = dict()

    csv = pd.read_csv(TRAIN_CSV)
    csv.set_index('ID', inplace=True)


    if not os.path.exists(DATA):
        os.mkdir(DATA)
        os.mkdir(DATA + NONE)

    for c in CLASSES:
        if not os.path.exists(DATA + c):
            os.mkdir(DATA + c)

    for f in tqdm(os.listdir(TRAIN)):
        f_name = f.split('.')[0]

        any_row = csv.loc[f_name + '_any']

        if float(any_row.values[0]) > 0.0:
            subdural_row = csv.loc[f_name + '_subdural']
            if float(subdural_row.values[0]) > 0.0:
                convert_to_jpeg(TRAIN + f_name + '.dcm', DATA + CLASSES[0] + '/' + f_name + '.jpg')
                subdural.add(f_name + '.dcm')

            epidural_row = csv.loc[f_name + '_epidural']
            if float(epidural_row.values[0]) > 0.0:
                convert_to_jpeg(TRAIN + f_name + '.dcm', DATA + CLASSES[1] + '/' + f_name + '.jpg')
                epidural.add(f_name + '.dcm')

            intraparenchymal_row = csv.loc[f_name + '_intraparenchymal']
            if float(intraparenchymal_row.values[0]) > 0.0:
                convert_to_jpeg(TRAIN + f_name + '.dcm', DATA + CLASSES[2] + '/' + f_name + '.jpg')
                intraparenchymal.add(f_name + '.dcm')

            intraventricular_row = csv.loc[f_name + '_intraventricular']
            if float(intraventricular_row.values[0]) > 0.0:
                convert_to_jpeg(TRAIN + f_name + '.dcm', DATA + CLASSES[3] + '/' + f_name + '.jpg')
                intraventricular.add(f_name + '.dcm')

            subarachnoid_row = csv.loc[f_name + '_subarachnoid']
            if float(subarachnoid_row.values[0]) > 0.0:
                convert_to_jpeg(TRAIN + f_name + '.dcm', DATA + CLASSES[4] + '/' + f_name + '.jpg')
                subarachnoid.add(f_name + '.dcm')

        else:
            convert_to_jpeg(TRAIN + f_name + '.dcm', DATA +NONE + f_name + '.jpg')
            none.add(f_name + '.dcm')

    print()

    print('subdural intersection none: {}'.format(str(none.intersection(subdural))))
    print('epidural intersection none: {}'.format(str(none.intersection(epidural))))
    print('intraparenchymal intersection none: {}'.format(str(none.intersection(intraparenchymal))))
    print('intraventricular intersection none: {}'.format(str(none.intersection(intraventricular))))
    print('subarachnoid intersection none: {}'.format(str(none.intersection(subarachnoid))))

