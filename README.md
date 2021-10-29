[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exploiting-emotional-dependencies-with-graph/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=exploiting-emotional-dependencies-with-graph)

# Emotion-GCN: Exploiting Emotional Dependencies with Graph Convolutional Networks for Facial Expression Recognition

<img src="/model.png" alt="model" width="1100"/>

This repository hosts the official PyTorch implementation of our paper "Exploiting Emotional Dependencies with Graph Convolutional Networks for Facial Expression Recognition" accepted at IEEE FG 2021.

Paper: [https://arxiv.org/abs/2106.03487](https://arxiv.org/abs/2106.03487)

Authored by: Panagiotis Antoniadis, Panagiotis Paraskevas Filntisis, Petros Maragos

## Abstract
> Over the past few years, deep learning methods have shown remarkable results in many face-related tasks including automatic facial expression recognition (FER) in-the-wild. Meanwhile, numerous models describing the human emotional states have been proposed by the psychology community. However, we have no clear evidence as to which representation is more appropriate and the majority of FER systems use either the categorical or the dimensional model of affect. Inspired by recent work in multi-label classification, this paper proposes a novel multi-task learning (MTL) framework that exploits the dependencies between these two models using a Graph Convolutional Network (GCN) to recognize facial expressions in-the-wild. Specifically, a shared feature representation is learned for both discrete and continuous recognition in a MTL setting. Moreover, the facial expression classifiers and the valence-arousal regressors are learned through a GCN that explicitly captures the dependencies between them. To evaluate the performance of our method under real-world conditions we train our models on AffectNet dataset. The results of our experiments show that our method outperforms the current state-of-the-art methods on discrete FER.


## Preparation

- Download the dataset. [[AffectNet]](http://mohammadmahoor.com/affectnet/) [[Aff-Wild2]](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)
- Download the 300-dimensional GloVe vectors trained on the Wikipedia dataset from [here](https://drive.google.com/file/d/1d4A5LwOXTvtNBpzCTMNoUIc3TmlWs6R-/view?usp=sharing).
- Run `pickle_annotations_affectnet.py` and `pickle_annotations_affwild2.py` to prepare each dataset.

## Training

- Train Emotion-GCN on a FER dataset:

```
python main.py --help
usage: main.py [-h] [--image_dir IMAGE_DIR] [--data DATA] [--dataset {affectnet,affwild2}] [--network {densenet,bregnext}] [--adj ADJ]
               [--emb EMB] [--workers WORKERS] [--batch_size BATCH_SIZE] [--model {single_task,multi_task,emotion_gcn}] [--epochs EPOCHS]
               [--lambda_multi LAMBDA_MULTI] [--lr LR] [--momentum MOMENTUM] --gpu GPU --saved_model SAVED_MODEL

Train Facial Expression Recognition model using Emotion-GCN

optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        path to images of the dataset
  --data DATA           path to the pickle file that holds all the information for each sample
  --dataset {affectnet,affwild2}
                        Dataset to use (default: affectnet)
  --network {densenet,bregnext}
                        Network to use (default: densenet)
  --adj ADJ             path to the pickle file that holds the adjacency matrix
  --emb EMB             path to the pickle file that holds the word embeddings
  --workers WORKERS     number of data loading workers (default: 4)
  --batch_size BATCH_SIZE
                        size of each batch (default: 35)
  --model {single_task,multi_task,emotion_gcn}
                        Model to use (default: emotion_gcn)
  --epochs EPOCHS       number of total epochs to train the network (default: 10)
  --lambda_multi LAMBDA_MULTI
                        lambda parameter of loss function
  --lr LR               learning rate (default: 0.001)
  --momentum MOMENTUM   momentum parameter of SGD (default: 0.9)
  --gpu GPU             id of gpu device to use
  --saved_model SAVED_MODEL
                        name of the saved model

```

## Pre-trained Models

We also provide weights for our Emotion-GCN models on AffectNet and Aff-Wild2. Our models achieves 66.46% accuracy on the categorical model of AffectNet outperforming the performance of the current state-of-the-art methods for discrete FER. You can download the pre-trained models [here](https://drive.google.com/drive/folders/1BUUOKelxNtkIETrb93nb6VIP-J4bT7Os?usp=sharing).

## Citation
Available soon.

## Contact
For questions feel free to open an issue.
