import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from SourceCode import AudioFolderDataset, AudioCNN, make_transform, evaluate


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = make_transform(args.sample_rate, args.n_mels)

    test_dir = args.test_dir or os.path.join(args.data_root, "testing")
    test_ds = AudioFolderDataset(test_dir, sample_rate=args.sample_rate, transform=transform)
    n_classes = len(test_ds.class_to_idx)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = AudioCNN(n_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Test set | loss {test_loss:.4f}, acc {test_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained ESC-50 model")

    parser.add_argument(
        "--data_root",
        type=str,
        default=r"C:\Users\bkarimov\OneDrive - University of Tasmania\Desktop\ESC-50-master\renamed",
        help="Root folder that contains 'testing' directory"
    )
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Optional explicit test folder (else data_root/testing)")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--model_path", type=str, default="esc50_cnn_best.pt")

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args(args=[])

    main(args)
