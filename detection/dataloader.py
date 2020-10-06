import torch.tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import os
import json
from PIL import Image as pil


def load(path):
    with open(path, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('RGB')


def collate_fn(batch):
    return tuple(zip(*batch))


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

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        annot_file = self.annot_files[index]

        img = load(img_file)
        img = self.transforms(img)

        with open(annot_file) as json_file:
            annotations = json.load(json_file)

        assert annotations['image']['file_name'] == img_file.split('/')[-1].split('.')[0]

        height = annotations['image']['height']
        width = annotations['image']['width']

        boxes = []
        area = []
        iscrowd = []
        labels = []

        for obj in annotations['annotation']:
            bbox = list(obj['bbox'])

            # convert from xywh to xyxy bbox format
            bbox[2] = min(bbox[0] + bbox[2], width)
            bbox[3] = min(bbox[1] + bbox[3], height)

            boxes.append(bbox)
            area.append(obj['area'])
            iscrowd.append(obj['iscrowd'])
            labels.append(int(obj['category_id']) - 1)

        target = dict()
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['area'] = torch.as_tensor(area, dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.tensor([annotations['image']['id']])

        return img, target


if __name__ == "__main__":
    # testing dataset
    dataset = FlirDataset()

    bs = 8
    workers = 4
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=workers,
                            pin_memory=True, drop_last=True, collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
