import os
import glob
import math
import time
import pickle
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2

from dataloading_affwild2 import Affwild2_annotation

def frames_to_label_cat(expression, frames):
    frames_ids = [int(frame.split('/')[-1].split('.')[0]) - 1 for frame in frames]
    drop_ids = [i for i in range(len(expression)) if expression[i]==-1]
    frames_ids = [i for i in frames_ids if i not in drop_ids]
    indexes = [True if i in frames_ids else False for i in range(len(expression))]
    expression = expression[indexes]
    assert len(expression) == len(frames_ids)
    prefix = '/'.join(frames[0].split('/')[:-1])
    return_frames = [prefix+'/{0:05d}.jpg'.format(id+1) for id in frames_ids]
    return expression, return_frames, frames_ids


def frames_to_label_cont(va, frames):
    frames_ids = [int(frame.split('/')[-1].split('.')[0]) - 1 for frame in frames]
    drop_ids = [i for i in range(len(va)) if va[i][0]<-1 or va[i][0]>1 or va[i][1]<-1 or va[i][1]>1]
    frames_ids = [i for i in frames_ids if i not in drop_ids]
    indexes = [True if i in frames_ids else False for i in range(len(va))]
    va = va[indexes]
    assert len(va) == len(frames_ids)
    prefix = '/'.join(frames[0].split('/')[:-1])
    return_frames = [prefix+'/{0:05d}.jpg'.format(id+1) for id in frames_ids]
    return va[:, 0], va[:, 1],  return_frames, frames_ids

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Process annotations of Aff-wild2 database")
    parser.add_argument('--annotations_dir', type=str)
    parser.add_argument('--videos_dir', type=str)

    args = parser.parse_args()
    annotations_dir = args.annotations_dir
    videos_dir = args.videos_dir

    train_cat = os.path.join(annotations_dir, 'EXPR_Set/Train_Set')
    val_cat = os.path.join(annotations_dir, 'EXPR_Set/Validation_Set')
    train_cont = os.path.join(annotations_dir, 'VA_Set/Train_Set')
    val_cont = os.path.join(annotations_dir, 'VA_Set/Validation_Set')

    train_cat_names = []
    val_cat_names = []
    train_cont_names = []
    val_cont_names = []

    for filename in os.listdir(train_cat):
        train_cat_names.append(filename[:-4])
    for filename in os.listdir(val_cat):
        val_cat_names.append(filename[:-4])
    
    for filename in os.listdir(train_cont):
        train_cont_names.append(filename[:-4])
    for filename in os.listdir(val_cont):
        val_cont_names.append(filename[:-4])
    

    train_cat_values = []
    val_cat_values = []
    for filename in train_cat_names:
        with open(os.path.join(train_cat, filename) + '.txt') as f:
            next(f)
            train_cat_values.append(np.array([int(line.strip('\n')) for line in f]))
    for filename in val_cat_names:
        with open(os.path.join(val_cat, filename) + '.txt') as f:
            next(f)
            val_cat_values.append(np.array([int(line.strip('\n')) for line in f]))

    train_cont_values = []
    val_cont_values = []
    for filename in train_cont_names:
        with open(os.path.join(train_cont, filename) + '.txt') as f:
            next(f)
            x = []
            for line in f:
                x.append(list(map(float, line.strip('\n').split(','))))
            train_cont_values.append(np.array(x))
    for filename in val_cont_names:
        with open(os.path.join(val_cont, filename) + '.txt') as f:
            next(f)
            x = []
            for line in f:
                x.append(list(map(float, line.strip('\n').split(','))))
            val_cont_values.append(np.array(x)) 

    data = {}
    data_train_cat = {}
    data_val_cat = {}

    for i, filename in enumerate(train_cat_names):
        frames_paths = sorted(glob.glob(os.path.join(videos_dir, filename, '*.jpg')))

        expression_array, frames_paths, frames_ids = frames_to_label_cat(train_cat_values[i], frames_paths)
        frames = []
        if len(frames_paths) == 0:
            continue
        for j in range(len(frames_ids)):
            sample = Affwild2_annotation(frame_path = frames_paths[j], expression = expression_array[j], valence=None, arousal=None)
            frames.append(sample)

        data_train_cat[os.path.join(videos_dir, filename)] = frames
    
    for i, filename in enumerate(val_cat_names):
        frames_paths = sorted(glob.glob(os.path.join(videos_dir, filename, '*.jpg')))
        expression_array, frames_paths, frames_ids = frames_to_label_cat(val_cat_values[i], frames_paths)
        frames = []
        if len(frames_paths) == 0:
            continue
        for j in range(len(frames_ids)):
            sample = Affwild2_annotation(frame_path = frames_paths[j], expression = expression_array[j], valence=None, arousal=None)
            frames.append(sample)
        
        data_val_cat[os.path.join(videos_dir, filename)] = frames


    data_cont = {}

    for i, filename in enumerate(train_cont_names):
        frames_paths = sorted(glob.glob(os.path.join(videos_dir, filename, '*.jpg')))
        valence_array, arousal_array, frames_paths, frames_ids = frames_to_label_cont(train_cont_values[i], frames_paths)        
        frames = []
        if len(frames_paths) == 0:
            continue
        for j in range(len(frames_ids)):
            sample = Affwild2_annotation(frame_path = frames_paths[j], valence=valence_array[j], arousal=arousal_array[j], expression=None)
            frames.append(sample)

        data_cont[os.path.join(videos_dir, filename)] = frames

    for i, filename in enumerate(val_cont_names):
        frames_paths = sorted(glob.glob(os.path.join(videos_dir, filename, '*.jpg')))
        valence_array, arousal_array, frames_paths, frames_ids = frames_to_label_cont(val_cont_values[i], frames_paths)
        frames = []
        if len(frames_paths) == 0:
            continue
        for j in range(len(frames_ids)):
            sample = Affwild2_annotation(frame_path = frames_paths[j], valence=valence_array[j], arousal=arousal_array[j], expression=None)
            frames.append(sample)
        
        data_cont[os.path.join(videos_dir, filename)] = frames

    train_mtl = []
    val_mtl = []

    for vid1 in data_train_cat.keys():
        for vid2 in data_cont.keys():
            if vid1 == vid2:
                for frame_cat in data_train_cat[vid1]:
                    for frame_cont in data_cont[vid2]:
                        if frame_cat.frame_path == frame_cont.frame_path:
                            train_mtl.append(Affwild2_annotation(frame_path=frame_cat.frame_path, expression=frame_cat.expression,
                                                            valence=frame_cont.valence, arousal=frame_cont.arousal))
                            break
                break


    for vid1 in data_val_cat.keys():
        for vid2 in data_cont.keys():
            if vid1 == vid2:
                for frame_cat in data_val_cat[vid1]:
                    for frame_cont in data_cont[vid2]:
                        if frame_cat.frame_path == frame_cont.frame_path:
                            val_mtl.append(Affwild2_annotation(frame_path=frame_cat.frame_path, expression=frame_cat.expression,
                                                                    valence=frame_cont.valence, arousal=frame_cont.arousal))
    
                            break
                break

    data = {'train': train_mtl, 'val':val_mtl}

    with open('data_affwild2.pkl', "wb") as w:
            pickle.dump(data, w)


