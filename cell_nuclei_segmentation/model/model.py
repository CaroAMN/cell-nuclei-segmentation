from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F




def create_model(n_channels, n_class, dropout_val=0.25):
    return UNet3D(n_channels, n_class, dropout_val=dropout_val)
    

class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout_val=0.001): #dropout_val=0.01
        super(ConvBlock, self).__init__()

        self.dropout_value = dropout_val
        self.dropout_1 = nn.Dropout2d(self.dropout_value)
        self.dropout_2 = nn.Dropout2d(self.dropout_value)

        self.non_linearity = nn.ReLU(inplace=False)

        self.conv_1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(out_ch)

        self.conv_2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):

        x = self.dropout_1(x)
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.non_linearity(x)

        x = self.dropout_2(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.non_linearity(x)

        return x


class InputBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout_val=0.001):
        super(InputBlock, self).__init__()

        self.conv_block_1 = ConvBlock(in_ch, out_ch, dropout_val=dropout_val) #, dropout_val=0.2)

    def forward(self, x):

        x = self.conv_block_1(x)

        return x


class DownSamplingBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout_val=0.001):
        super(DownSamplingBlock, self).__init__()

        self.dropout_value = 0.001 #0.25, 0.01
        
        self.down = nn.Sequential(
            nn.Dropout2d(self.dropout_value),
            
            #nn.MaxPool3d(2, stride=2),
            nn.Conv2d(in_ch, in_ch, 2, stride=2),
            
            #nn.BatchNorm3d(in_ch),
            #nn.ReLU(inplace=True),

            ConvBlock(in_ch, out_ch, dropout_val=dropout_val) #, dropout_val=0.2)
        )

    def forward(self, x):
        
        x = self.down(x)
        return x



class UpSamplingBlock(nn.Module):

    def __init__(self, in_ch, cat_ch, out_ch, dropout_val=0.001):
        super(UpSamplingBlock, self).__init__()

        self.dropout_value = 0.001 #0.25, 0.01

        self.up = nn.Sequential(
            nn.Dropout2d(self.dropout_value),

            # nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2),
            nn.Upsample(scale_factor=2, mode='nearest'),

            #nn.BatchNorm3d(in_ch),
            #nn.ReLU(inplace=True)
        )

        self.conv = ConvBlock(in_ch + cat_ch, out_ch, dropout_val=dropout_val) #, dropout_val=0.2)

    def cat_operation(self, dc, syn):
        
        return torch.cat((dc, syn), dim=1)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = self.cat_operation(x1, x2)
        x = self.conv(x)

        return x


class OutputBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(OutputBlock, self).__init__()

        self.conv_1= nn.Conv2d(in_ch, out_ch, 1)
        #self.batch_norm_1 = nn.BatchNorm3d(out_ch)
        
    def forward(self, x):
        
        x = self.conv_1(x)
        #x = self.batch_norm_1(x)
        
        #softmax out?
        #x = F.softmax(x, dim=1)

        return x






"""The Unet2D model """

class UNet2D(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_val=0.001):
        super(UNet2D, self).__init__()


        self.inc = InputBlock(n_channels, 32, dropout_val=dropout_val)
        
        self.down1 = DownSamplingBlock(32, 64, dropout_val=dropout_val)
        self.down2 = DownSamplingBlock(64, 128, dropout_val=dropout_val)

        self.mid = ConvBlock(128, 128, dropout_val=dropout_val)

        self.up1 = UpSamplingBlock(128, 64, 64, dropout_val=dropout_val)
        self.up2 = UpSamplingBlock(64, 32, 32, dropout_val=dropout_val)

        self.outc = OutputBlock(32, n_classes)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)

        return x


         






















class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, len_test_set: int, hparams: dict, **kwargs):
        """
        Initializes the network
        """
        super(LightningMNISTClassifier, self).__init__()
        self.hparams = hparams

        # mnist images are (1, 28, 28) (channels, width, height)
        self.optimizer = None
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.dropout2 = torch.nn.Dropout2d(0.25)
        self.fc2 = torch.nn.Linear(128, 10)
        self.args = kwargs
        self.len_test_set = len_test_set
        self.train_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers', type=int, default=3, metavar='N', help='number of workers (default: 3)')
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
        parser.add_argument('--training-batch-size', type=int, default=64, help='Input batch size for training')
        parser.add_argument('--test-batch-size', type=int, default=1000, help='Input batch size for testing')

        return parser

    def forward(self, x):
        """
        :param x: Input data

        :return: output - mnist digit label for the input image
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output

    def cross_entropy_loss(self, logits, labels):
        """
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        :param train_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        """
        On each training epoch end, log the average training loss
        """
        train_avg_loss = torch.stack([train_output['loss'] for train_output in training_step_outputs]).mean()
        self.log('train_avg_loss', train_avg_loss, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        """
        Predicts on the test dataset to compute the current accuracy of the model.

        :param test_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Testing accuracy
        """

        x, y = test_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        self.test_acc(y_hat, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        # sum up batch loss
        data, target = Variable(x), Variable(y)  # noqa: F841
        test_loss = F.nll_loss(output, target, reduction='sum').data.item()
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).sum()
        return {'test_loss': test_loss, 'correct': correct}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score

        :param outputs: outputs after every epoch end

        :return: output - average test loss
        """
        avg_test_loss = sum([test_output['test_loss'] for test_output in outputs]) / self.len_test_set
        test_correct = float(sum([test_output['correct'] for test_output in outputs]))
        self.log('avg_test_loss', avg_test_loss, sync_dist=True)
        self.log('test_correct', test_correct, sync_dist=True)

    def prepare_data(self):
        """
        Prepares the data for training and prediction
        """
        return {}

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'])
        self.scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-6, verbose=True,
            ),
            'monitor': 'train_avg_loss',
        }
        return [self.optimizer], [self.scheduler]
