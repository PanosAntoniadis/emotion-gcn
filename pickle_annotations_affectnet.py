import os
import pickle
import argparse
import numpy as np
import pandas as pd

from dataloading import AffectNet_annotation


def get_eyes(landmarks):
    landmarks = landmarks.split(';')
    left = landmarks[36*2:42*2]
    right = landmarks[42*2:48*2]
    left_eye_pts = np.array([[float(left[i]), float(left[i+1])]
                             for i in range(0, len(left), 2)])
    right_eye_pts = np.array([[float(right[i]), float(right[i+1])]
                              for i in range(0, len(right), 2)])
    # compute the center of mass for each eye
    left_eye = left_eye_pts.mean(axis=0).astype("int")
    right_eye = right_eye_pts.mean(axis=0).astype("int")
    return left_eye, right_eye


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle images annotations')

    parser.add_argument('--csv_dir',
                        help='path to annotations of the dataset', required=True)

    args = parser.parse_args()

    df_train = pd.read_csv(os.path.join(args.csv_dir, 'training.csv'))

    df_train = df_train[df_train['expression'] < 7]

    collumns = ["subDirectory_filePath", "face_x", "face_y", "face_width",
                "face_height", "facial_landmarks", "expression", "valence", "arousal"]
    df_val = pd.read_csv(os.path.join(
        args.csv_dir, 'validation.csv'))
    df_val = df_val[df_val['expression'] < 7]

    data_train = []
    data_val = []

    for index, row in df_train.iterrows():
        left_eye, right_eye = get_eyes(row['facial_landmarks'])
        sample = AffectNet_annotation(
            row['subDirectory_filePath'], row['face_x'], row['face_y'], row['face_width'], row['face_height'], row['expression'], row['valence'], row['arousal'], left_eye, right_eye)
        data_train.append(sample)

    for index, row in df_val.iterrows():
        left_eye, right_eye = get_eyes(row['facial_landmarks'])
        sample = AffectNet_annotation(
            row['subDirectory_filePath'], row['face_x'], row['face_y'], row['face_width'], row['face_height'], row['expression'], row['valence'], row['arousal'], left_eye, right_eye)
        data_val.append(sample)

    data = {'train': data_train, 'val': data_val}

    with open('data.pkl', "wb") as w:
        pickle.dump(data, w)
