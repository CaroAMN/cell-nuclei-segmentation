import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TissueNetDataset(Dataset):
    def __init__(self,ids, path= "ds/", apply_trans=False):
        """
        Args:
        assumes the filenames of an image pair (input and label) are img_<ID>.pt and lab_<ID>.pt

        """

        self.inputs = []
        self.labels = []

        for sample in ids:
            pic = torch.load(path + "img_" + str(sample) + ".pt")
            label = torch.load(path + "lab_" + str(sample) + ".pt")

            label = label.astype(np.float32)

            pic = np.expand_dims(pic, axis=0)  # add channel dim

            pic = torch.tensor(pic)
            label = torch.tensor(label)

            self.inputs.append(pic)
            self.labels.append(label)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        pic = self.inputs[idx]
        label = self.labels[idx]

        pair = (pic, label)

        return pair
    

class TissueNetDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of the data module with a train, val and test dataset, as well as a loader for each.
        """

        super(TissueNetDataModule, self).__init__()
        self.df_train = None
        self.df_test = None
        self.df_val = None
        self.train_data_loader = None
        self.test_data_loader = None
        self.val_data_loadern = None
        self.args = kwargs

        # # define train/test sets
        # self.train_ids = [48, 3, 23, 75, 22, 109, 73, 130, 115, 86, 121, 67, 97, 116, 14, 125, 52, 84, 129, 58, 49, 110,
        #                   43, 88, 25, 4, 89, 50, 29, 94, 53, 16, 2, 46, 92, 113, 44, 15, 111, 124, 69, 47, 5, 104, 54, 37, 76,
        #                   119, 13, 34, 21, 103, 80, 91, 82, 35, 19, 6, 72, 59, 105, 83, 20, 128, 120, 57, 101, 30, 28, 24, 8, 41,
        #                   31, 95, 63, 0, 126, 11, 1, 85, 7, 33, 127, 56, 118, 70, 26, 81, 78, 40, 55, 122, 99, 71, 60, 42, 87, 9, 93,
        #                   108, 39, 18, 77, 90, 68, 32, 102, 79, 12, 96, 112, 36, 65, 123, 66, 10, 107, 98]

        # self.test_ids = [17, 64, 27, 114, 74, 45, 61, 38, 106, 100, 117, 51, 62]

        self.train_ids = [0, 1]
        self.val_ids = [0,1]
        self.test_ids = [0, 1]

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data
        :param stage: Stage - training or testing
        """
        self.df_train = TissueNetDataset(self.train_ids, path=self.args['dataset_path'], apply_trans=False)
        self.df_val = TissueNetDataset(self.val_ids, path=self.args['dataset_path'], apply_trans=False)
        self.df_test = TissueNetDataset(self.test_ids, path=self.args['dataset_path'], apply_trans=False)

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return DataLoader(self.df_train, batch_size=self.args['training_batch_size'], num_workers=self.args['num_workers'], shuffle=True)

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        return DataLoader(self.df_test, batch_size=self.args['test_batch_size'], num_workers=self.args['num_workers'], shuffle=False)

    def val_dataloader(self):
        """
        :return: output - Val data loader for the given input
        """
        return DataLoader(self.df_val, batch_size=self.args['test_batch_size'], num_workers=self.args['num_workers'], shuffle=False)    


