import argparse
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


LABEL_MAP = {0: 5, 1: 7, 2: 10, 3: 12, 4: 13, 5: 14, 6: 15, 7: 16}


class EmbryoPredictDataset(Dataset):
    def __init__(self, images_path):
        self.images_path = images_path
        self.data = glob.glob(images_path + '/*')
        self.tfs = transforms.Compose((transforms.CenterCrop((256, 512)),
                                       transforms.ToTensor()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx])
        image = self.tfs(image)
        return image


def predict(model, loader):
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            preds = model(imgs).cpu().numpy().argmax(1)
            all_predictions.extend(preds)
    remapped_predictions = np.array([LABEL_MAP[i] for i in all_predictions])
    return remapped_predictions


def move_to_folders(preds, dset):
    for stage in np.unique(preds):
        os.mkdir(os.path.join(dset.images_path, str(stage)))
    for img, p in zip(dset.data, preds):
        tif_name = os.path.split(img)[1]
        new_name = os.path.join(dset.images_path, str(p), tif_name)
        os.rename(img, new_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict embyo stage for all images in a given folder')
    parser.add_argument('images_folder', type=str, help='folder with images')
    parser.add_argument('--device', type=str, default='cpu', help='GPU to use',
                        choices=[str(n) for n in range(8)] + ['cpu', ])
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size for prediction')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='number of dataloader workers')
    parser.add_argument('--model', type=str, default='stage_classifier.pth',
                        help='model to use')
    args = parser.parse_args()

    if args.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device('cpu')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.device('cuda')
    print("Using device {}".format(args.device))

    print("Loading the model")
    model = models.resnet18(num_classes=8)
    model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.load_state_dict(torch.load(args.model, map_location=torch.device(device)))
    model = model.to(device)

    print("Loading the data")
    dataset = EmbryoPredictDataset(args.images_folder)
    predict_loader = DataLoader(dataset, batch_size=10, num_workers=16)

    print("Predicting")
    predictions = predict(model, predict_loader)

    print("Sorting files")
    move_to_folders(predictions, dataset)
