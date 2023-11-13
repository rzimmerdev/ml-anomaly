

def train_model(args):
    model = CNNTransformerModel(args.input_size, args.hidden_size, args.num_heads, args.num_layers, args.num_classes)

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='best_model',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )

    train_dataset = TimeSeriesDataset(args.data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = TimeSeriesDataset(args.val_data_dir)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    trainer = CNNTransformerTrainer(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_checkpoint=model_checkpoint
    )

    if args.resume_training:
        trainer.load_checkpoint(args.resume_checkpoint)

    trainer.fit()

    path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    trainer.save_checkpoint(path)


def main():
    parser = argparse.ArgumentParser(description='CNN-Transformer Time Series Classification')
    parser.add_argument('--input_size', type=int, default=6, help='Number of input features')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for the model')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes for classification')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--data_dir', type=str, default, help='Dataset directory from which to load series')

    train(parser.args)

