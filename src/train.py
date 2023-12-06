import argparse
import os
import timeit

import torch
from torch.utils.data import DataLoader, Subset
import lightning

from model import MultiClassAnomaly
from dataset import SeriesDataset


def train(args):
    model = MultiClassAnomaly(args.input_size, args.hidden_size, args.num_heads, args.num_layers, args.num_classes)

    dataset = SeriesDataset(args.data_dir)

    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    trainer = lightning.Trainer(default_root_dir=args.checkpoint_dir, max_epochs=args.max_epochs,
                                check_val_every_n_epoch=10)

    # Measure training time
    time = timeit.timeit(lambda: trainer.fit(model=model,
                                             train_dataloaders=train_dataloader,
                                             val_dataloaders=val_dataloader),
                         number=1)

    print(f'Training time: {time} seconds')

    checkpoint = {
        'hyperparameters': vars(args),
        'state_dict': model.state_dict(),
    }
    torch.save(checkpoint, f'{args.checkpoint_dir}/model_checkpoint.pth')


# Train oneclass
def train_oneclass(args):
    dataset = SeriesDataset(args.data_dir)
    os.makedirs(f'{args.checkpoint_dir}/one_out', exist_ok=True)

    validation_acc = []

    for i in range(len(dataset)):
        print(f'Fold {i + 1}/{len(dataset)}')
        val_dataset = Subset(dataset, [i])
        train_indices = list(range(0, i)) + list(range(i + 1, len(dataset)))
        train_dataset = Subset(dataset, train_indices)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        model = MultiClassAnomaly(args.input_size, args.hidden_size, args.num_heads, args.num_layers, args.num_classes)
        trainer = lightning.Trainer(default_root_dir=args.checkpoint_dir, max_epochs=args.max_epochs,
                                    check_val_every_n_epoch=10)

        time = timeit.timeit(lambda: trainer.fit(model=model,
                                                 train_dataloaders=train_dataloader,
                                                 val_dataloaders=val_dataloader),
                             number=1)

        print(f'Training time: {time} seconds')
        validation_acc.append(trainer.callback_metrics['val_accuracy'].item())

    print(f'Validation accuracy: {sum(validation_acc) / len(validation_acc)}')


def main():
    parser = argparse.ArgumentParser(description='CNN-Transformer Time Series Classification')
    parser.add_argument('--input_size', type=int, default=7, help='Number of input features')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for the model')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes for classification')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--data_dir', type=str, default="data/train", help='Dataset directory for loading series')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints/", help='Directory to save checkpoints')
    parser.add_argument('--resume_training', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--max_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--one_out', type=bool, default=False, help='Train oneclass')

    args = parser.parse_args()

    if args.one_out:
        train_oneclass(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
