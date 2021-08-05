import os
import argparse
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from ccc import CCC_loss
from models import Emotion_GCN
from aligner import FaceAligner
from dataloading import AffectNet_dataset
from training import train_model, eval_model

########################################################
# Configuration
########################################################

# Define argument parser
parser = argparse.ArgumentParser(
    description='Facial expression recognition using Emotion-GCN')

# Data loading
parser.add_argument('--image_dir',
                    default='./affectnet',
                    help='path to images of the dataset')
parser.add_argument('--data',
                    default='./data.pkl',
                    help='path to the pickle file that holds all the information for each sample')
parser.add_argument('--adj',
                    default='./adj.pkl',
                    help='path to the pickle file that holds the adjacency matrix')
parser.add_argument('--emb',
                    default='./emb.pkl',
                    help='path to the pickle file that holds the word embeddings')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=35, type=int,
                    help='size of each batch (default: 35)')
# Training
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to train the network (default: 10)')
parser.add_argument('--lambda_multi', default=1, type=float,
                    help='lambda parameter of loss function')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum parameter of SGD (default: 0.9)')
parser.add_argument('--gpu', type=int,
                    help='id of gpu device to use', required=True)
parser.add_argument('--saved_model', type=str,
                    help='name of the saved model', required=True)

# Get arguments from the parser in a dictionary,
args = parser.parse_args()

# Check inputs.
if not os.path.isdir(args.image_dir):
    raise FileNotFoundError("Image directory not exists")
if not os.path.exists(args.data):
    raise FileNotFoundError("Pickle file not exists")
if args.workers <= 0:
    raise ValueError("Invalid number of workers")
if args.batch_size <= 0:
    raise ValueError("Invalid batch size")
if args.epochs <= 0:
    raise ValueError("Invalid number of epochs")
if args.lr <= 0:
    raise ValueError("Invalid learning rate")
if args.momentum < 0:
    raise ValueError("Invalid momentum value")

# Set cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

########################################################
# Save useful parameters of the model
########################################################
output_file = os.path.join('./outputs', args.saved_model)

print("Image directory: {}".format(args.image_dir))
print("Data file: {}".format(args.data))
print("Adjacency file: {}".format(args.adj))
print("Embeddings file: {}".format(args.emb))
print("Number of workers: {}".format(args.workers))
print("Batch size: {}".format(args.batch_size))
print("Number of epochs: {}".format(args.epochs))
print("Lambda {}".format(args.lambda_multi))
print("Learning rate: {}".format(args.lr))
print("Momentum: {}".format(args.momentum))
print("Gpu used: {}".format(args.gpu))

def main():
    ########################################################
    # Define datasets and dataloaders
    ########################################################

    resized_size = 227
    rotation = 30

    train_transforms = transforms.Compose([
        transforms.Resize(resized_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomRotation(rotation),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(resized_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    aligner = FaceAligner()

    train_dataset = AffectNet_dataset(root_dir=args.image_dir, data_pkl=args.data, emb_pkl=args.emb, aligner=aligner, train=True, transform=train_transforms,
                                      crop_face=True)
    val_dataset = AffectNet_dataset(root_dir=args.image_dir, data_pkl=args.data, emb_pkl=args.emb, aligner=aligner, train=False, transform=val_transforms,
                                    crop_face=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.workers, pin_memory=True)

    #############################################################################
    # Model Definition (Model, Loss Function, Optimizer)
    #############################################################################

    model = Emotion_GCN(adj_file=args.adj)
    # Move the mode weight to cpu or gpu
    model.cuda()
    print(model)

    # Define loss function
    weights = torch.FloatTensor(
        [3803/74874, 3803/134415, 3803/25459, 3803/14090, 3803/6378, 1, 3803/24882]).cuda()
    criterion_cat = torch.nn.CrossEntropyLoss(weight=weights)
    criterion_cont = CCC_loss()

    # We optimize only those parameters that are trainable
    params = list(model.parameters())
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=0)

    #############################################################################
    # Training Pipeline
    #############################################################################

    # Define lists for train and val loss over each epoch
    train_losses = []
    val_losses = []
    train_cat_losses = []
    val_cat_losses = []
    train_cont_losses = []
    val_cont_losses = []

    for epoch in range(args.epochs):
        train_loss, train_loss_cat, train_loss_cont, (y_train_true, y_train_pred) = train_model(
            train_dataloader, model, criterion_cat, criterion_cont, optimizer)

        val_loss, val_loss_cat, val_loss_cont, (y_val_true, y_val_pred) = eval_model(
            val_dataloader, model, criterion_cat, criterion_cont)

        # Save losses to the corresponding lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_cat_losses.append(train_loss_cat)
        val_cat_losses.append(val_loss_cat)
        train_cont_losses.append(train_loss_cont)
        val_cont_losses.append(val_loss_cont)

        # Convert preds and golds in a list.
        y_train_true = np.concatenate(y_train_true, axis=0)
        y_val_true = np.concatenate(y_val_true, axis=0)
        y_train_pred = np.concatenate(y_train_pred, axis=0)
        y_val_pred = np.concatenate(y_val_pred, axis=0)

        # Print metrics for current epoch
        print('Epoch: {}'.format(epoch))
        print("My train loss is : {}".format(train_loss))
        print("My val loss is : {}".format(val_loss))
        print("My train categorical loss is : {}".format(train_loss_cat))
        print("My val categorical loss is : {}".format(val_loss_cat))
        print("My train continuous loss is : {}".format(train_loss_cont))
        print("My val continuous loss is : {}".format(val_loss_cont))
        print("Accuracy for train: {}".format(accuracy_score(
            y_train_true, y_train_pred)))
        print("Accuracy for val: {}".format(
            accuracy_score(y_val_true, y_val_pred)))

    # Save trained model
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(state, output_file)
    print("Model saved succesfully to {}".format(output_file))


if __name__ == '__main__':
    main()
