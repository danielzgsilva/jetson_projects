from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import os
import json
from PIL import Image as pil


class FlirDataset(Dataset):
    def __init__(self, data_root='/groups/mshah/data/FLIR/pre_dat', validation=False, transforms=None):
        self.data_root = data_root
        self.img_files = []
        self.annot_files = []

        self.transforms = transforms
        if self.transforms is None:
            self.transforms = ToTensor()

        # Get all image file names
        if validation:
            val_files = os.listdir(os.path.join(self.data_root, 'valid/PreviewData/'))
            val_files.sort()
            self.img_files.extend(os.path.join(self.data_root, 'valid/PreviewData/') +
                                  file for file in val_files)

        else:
            train_files = os.listdir(os.path.join(self.data_root, 'train/PreviewData/'))
            train_files.sort()
            self.img_files.extend(os.path.join(self.data_root, 'train/PreviewData/') +
                                  file for file in train_files)

            video_files = os.listdir(os.path.join(self.data_root, 'video/PreviewData/'))
            video_files.sort()
            self.img_files.extend(os.path.join(self.data_root, 'video/PreviewData/') +
                                  file for file in video_files)

        # Get annotation file names
        for file in self.img_files:
            annot_file = file.replace('PreviewData', 'Annotations').replace('.jpeg', '.json')
            self.annot_files.append(annot_file)

        print('Image files: {} Annotation files: {}'.format(len(self.img_files), len(self.annot_files)))
        assert len(self.img_files) == len(self.annot_files)

    def load(self, path):
        with open(path, 'rb') as f:
            with pil.open(f) as img:
                return img.convert('RGB')

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        annot_file = self.annot_files[index]

        img = self.load(img_file)
        img = self.transforms(img)

        print(img.size())

        with open(annot_file) as json_file:
            annotations = json.load(json_file)

        print(annotations)

        return img, annotations


if __name__ == "__main__":
    # testing dataset
    dataset = FlirDataset()

    bs = 12
    workers = 8
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)

    for batch_idx, (inputs, annotations) in enumerate(dataloader):
        pass
