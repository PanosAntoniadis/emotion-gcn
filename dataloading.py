import os
import pickle
import cv2
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

class AffectNet_annotation(object):
    """A class that represents a sample from AffectNet."""

    def __init__(self, image_path, face_x, face_y, face_width, face_height, expression, valence, arousal, left_eye, right_eye):
        super(AffectNet_annotation, self).__init__()
        self.image_path = image_path
        self.face_x = face_x
        self.face_y = face_y
        self.face_width = face_width
        self.face_height = face_height
        self.expression = expression
        self.valence = valence
        self.arousal = arousal
        self.left_eye = left_eye
        self.right_eye = right_eye

class AffectNet_dataset(Dataset):
    """AffectNet: Facial expression recognition dataset."""

    def __init__(self, root_dir, data_pkl, emb_pkl, aligner, train=True, transform=None, crop_face=True):
        self.root_dir = root_dir
        self.aligner = aligner
        self.train = train
        self.transform = transform
        self.crop_face = crop_face

        data_pickle = pickle.load(open(data_pkl, 'rb'))
        if train:
            self.data = data_pickle['train']
        else:
            self.data = data_pickle['val']

        self.inp = torch.load(emb_pkl)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get selected sample
        sample = self.data[idx]
        # Read image
        img_name = os.path.join(self.root_dir, sample.image_path)
        image = cv2.imread(img_name)[..., ::-1]

        image, M = self.aligner.align(image, sample.left_eye, sample.right_eye)
        if self.crop_face:
            # Keep only the face region of the image
            face_x = int(sample.face_x)
            face_y = int(sample.face_y)
            face_width = int(sample.face_width)
            face_height = int(sample.face_height)

            point = (face_x, face_y)
            rotated_point = M.dot(np.array(point + (1,)))
            image = image.crop(
                (rotated_point[0], rotated_point[1], rotated_point[0]+face_width, rotated_point[1]+face_height))

        # Read the expression of the image
        expression = sample.expression
        # Apply the transformation
        if self.transform:
            image = self.transform(image)

        cont = np.array([sample.valence, sample.arousal])

        return image, expression, cont, self.inp


class Affwild2_annotation(object):
    """A class that represents a sample from Aff-Wild2."""

    def __init__(self, frame_path, expression, valence, arousal):
        super(Affwild2_annotation, self).__init__()
        self.frame_path = frame_path
        self.expression = expression
        self.valence = valence
        self.arousal = arousal

class Affwild2_dataset(Dataset):
    """Aff-Wild2"""

    def __init__(self, data_pkl, emb_pkl, train=True, transform=None):

        self.train = train
        self.transform = transform

        data_pickle = pickle.load(open(data_pkl, 'rb'))
        if train:
            self.data = data_pickle['train']
        else:
            self.data = data_pickle['val']

        self.inp = torch.load(emb_pkl)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        img_name = sample.frame_path
        image = Image.open(img_name).convert("RGB")

        expression = sample.expression
        if self.transform:
            image = self.transform(image)

        cont  = np.array([sample.valence, sample.arousal])
        
        return image, expression, cont, self.inp
                
