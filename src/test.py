import lightning
import torch
from torch.utils.data import DataLoader
from src.model import MultiClassAnomaly
from src.dataset import SeriesDataset


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

    trainer = lightning.Trainer(default_root_dir=args['checkpoint_dir'], max_epochs=args['epochs'])
    trainer.test(model=model, dataloaders=dataloader)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test trained model on the entire dataset')
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/model_checkpoint.pth",
                        help='Path to the model checkpoint file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
    parser.add_argument('--data_dir', type=str, default="Sara_dataset/test", help='Dataset directory for loading series')

    args = parser.parse_args()
    test(args)
