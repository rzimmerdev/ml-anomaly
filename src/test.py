import time
import os
import timeit

import lightning
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import MultiClassAnomaly, MultiClassEnsemble
from dataset import SeriesDataset

from plotly import express as px


def metrics(model):
    confusion_matrix = model.confusion_matrix

    figure = "garbage_clean.pdf"
    fig = px.scatter(x=[0], y=[0])
    fig.write_image(figure, format="pdf")
    time.sleep(2)

    # Make x labels vertical
    heatmap = px.imshow(
        confusion_matrix,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        text_auto=True,
        x=['Normal', 'Left Up', 'Left Un.', 'Right Up', 'Right Un.'],
        y=['Normal', 'Left Up', 'Left Un.', 'Right Up', 'Right Un.'],
    )
    heatmap.update_xaxes(tickangle=-90)
    heatmap.update_layout(xaxis_nticks=5, yaxis_nticks=5,
                          font=dict(size=32, family='Courier New, monospace'),
                          width=800, height=800)
    # Save as PDF
    heatmap.write_image("cm.pdf")

    # roc_curve is a tuple with 3 elements: (fpr, tpr, thresholds)
    # Each element is a list with num_classes elements
    roc_curve = model.roc_curve.compute()
    fpr = roc_curve[0]
    tpr = roc_curve[1]

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)

    for i in range(len(fpr)):
        mean_tpr += np.interp(mean_fpr, fpr[i], tpr[i])

    mean_tpr /= len(fpr)

    fig = px.line(x=mean_fpr, y=mean_tpr, labels=dict(x="False Positive Rate", y="True Positive Rate"))

    fig.update_layout(
        xaxis_nticks=5, yaxis_nticks=5,
        font=dict(size=32, family='Courier New, monospace'),
        width=800, height=800)
    fig.write_image("roc.pdf")

    auc = np.trapz(mean_tpr, mean_fpr)
    print(f'AUC: {auc}')


def test(args):
    checkpoint = torch.load(args.checkpoint_path)
    hyperparams = checkpoint['hyperparameters']

    model_args = {
        'input_size': hyperparams['input_size'],
        'hidden_size': hyperparams['hidden_size'],
        'num_heads': hyperparams['num_heads'],
        'num_layers': hyperparams['num_layers'],
        'num_classes': hyperparams['num_classes'],
    }

    model = MultiClassAnomaly(**model_args)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataset = SeriesDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    trainer = lightning.Trainer(default_root_dir=hyperparams['checkpoint_dir'], max_epochs=hyperparams['max_epochs'])

    trainer.test(model=model, dataloaders=dataloader)


def ensemble_test(args):
    ensemble_dir = "checkpoints/ensemble"

    ensemble = []

    for fold in os.listdir(ensemble_dir):
        checkpoint = torch.load(f'{ensemble_dir}/{fold}')
        hyperparams = checkpoint['hyperparameters']

        model_args = {
            'input_size': hyperparams['input_size'],
            'hidden_size': hyperparams['hidden_size'],
            'num_heads': hyperparams['num_heads'],
            'num_layers': hyperparams['num_layers'],
            'num_classes': hyperparams['num_classes'],
        }

        model = MultiClassAnomaly(**model_args)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval().to('cuda')

        ensemble.append(model)

    model = MultiClassEnsemble(5, ensemble)

    dataset = SeriesDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    trainer = lightning.Trainer()

    time = timeit.timeit(lambda: trainer.test(model=model, dataloaders=dataloader), number=1)
    metrics(model)
    print("Time per sample: ", time / len(dataset))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test trained model on the entire dataset')
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/model_checkpoint.pth",
                        help='Path to the model checkpoint file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
    parser.add_argument('--data_dir', type=str, default="data/test", help='Dataset directory for loading series')
    parser.add_argument('--ensemble', type=bool, default=False, help='Test ensemble model')

    pars_args = parser.parse_args()

    if pars_args.ensemble:
        ensemble_test(pars_args)
    else:
        test(pars_args)
