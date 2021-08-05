import os
import torch
import pickle
import argparse
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from models import Emotion_GCN
from aligner import FaceAligner
from dataloading import AffectNet_dataset

from utils import save_cm_plot, CCC_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a model trained in AffectNet')

    parser.add_argument('--model_path',
                        help='path to the saved model', required=True)
    parser.add_argument('--image_dir',
                        help='path to images of the dataset', required=True)
    parser.add_argument('--data',
                        help='path to the pickle file that holds all the information for each sample', required=True)
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
    parser.add_argument('--gpu', type=int,
                        help='id of gpu device to use', required=True)
    parser.add_argument('--saved_model', type=str,
                        help='name of the saved model', required=True)

    # Get arguments from the parser in a dictionary,
    args = parser.parse_args()
    # Set cuda device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    resized_size = 227
    model_transforms = transforms.Compose([
        transforms.Resize(resized_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    aligner = FaceAligner()

    dataset = AffectNet_dataset(root_dir=args.image_dir, data_pkl=args.data, emb_pkl=args.emb, aligner=aligner, train=False, transform=model_transforms,
                                crop_face=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers, pin_memory=True)

    model = Emotion_GCN(adj_file=args.adj)
    # Move the mode weight to cpu or gpu
    model.cuda()

    print('Model Information:')
    print(model)

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print('Epoch: {}'.format(checkpoint['epoch']))

    y_pred = []
    y_true = []
    cont_true = []
    cont_pred = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            # Get the inputs (batch)
            inputs, labels, labels_cont, inp = batch

            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_cont = labels_cont.to(device)
            inp = inp.to(device)

            outputs_cat, outputs_cont = model(inputs, inp)

            _, preds = torch.max(outputs_cat, 1)

            y_pred.append(preds.cpu().numpy())
            y_true.append(labels.cpu().numpy())
            cont_pred.append(outputs_cont.cpu())
            cont_true.append(labels_cont.cpu())

    # Convert preds and golds in a list.
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    print("Accuracy: {}".format(
        accuracy_score(y_true, y_pred)))

    ccc = CCC_metric(torch.cat(cont_pred), torch.cat(cont_true))
    print("CCC - Valence: {}".format(ccc[0]))
    print("CCC - Arousal: {}".format(ccc[1]))

    labels = [0, 1, 2, 3, 4, 5, 6]
    labels_name = ['Neutral', 'Happy', 'Sad',
                   'Surprise', 'Fear', 'Disgust', 'Anger']

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    save_cm_plot(cm, labels_name, 'confusion_matrix.png')
