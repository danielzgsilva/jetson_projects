from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class FlirDataset(Dataset):
    def __init__():
        self.path = '/groups/mshah/data/FLIR'
        self.train_filenames = []
        train_files = os.listdir(os.path.join(self.path, 'train/PreviewData/'))
        train_files.sort()
        self.train_filenames.extend(os.path.join(self.path, 'train/PreviewData/') +
                               file for file in train_files[1:-1])

        video_files = os.listdir(os.path.join(self.path, 'video/PreviewData/'))
        video_files.sort()
        self.train_filenames.extend(os.path.join(self.path, 'video/PreviewData/') +
                               file for file in video_files[1:-1])

        val_files = os.listdir(os.path.join(self.path, 'valid/PreviewData/'))
        val_files.sort()
        self.val_filenames.extend(os.path.join(self.path, 'valid/PreviewData/') +
                             file for file in val_files[1:-1])

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        img = self.train_filenames[index]
