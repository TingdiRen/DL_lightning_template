# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import importlib
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from collections import OrderedDict
from easydict import EasyDict


class DInterface(pl.LightningDataModule):
    def __init__(self, dataset_name, *args, **kwargs):
        super().__init__()
        self.register_hyperparameters()
        self.load_data_module()

    def register_hyperparameters(self):
        self.save_hyperparameters()
        self.data_hparams = EasyDict([(hparam.split(".")[-1], v) for hparam, v in self.hparams.items() if
                                      hparam.startswith("init_args")])

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(mode='train')
            self.valset = self.instancialize(mode='val')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(mode='test')

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.data_hparams.batch_size,
                          num_workers=self.data_hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.data_hparams.batch_size,
                          num_workers=self.data_hparams.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.data_hparams.batch_size,
                          num_workers=self.data_hparams.num_workers, shuffle=False)

    def load_data_module(self):
        name = self.hparams.dataset_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                'data.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        '''
        Instancialize a model using the corresponding parameters
        from self.hparams dictionary. You can also input any args
        to overwrite the corresponding value in self.kwargs.

        Args:
            **other_args:
        Returns:
            instancialized data_module
        '''
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        args = EasyDict()
        for arg in class_args:
            if arg in self.data_hparams:
                args[arg] = self.data_hparams[arg]
        args.update(other_args)
        return self.data_module(**args)
