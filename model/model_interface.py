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
import torch
import importlib
import torch.optim.lr_scheduler as lrs
from easydict import EasyDict
from model.utils import to_pil
import lightning.pytorch as pl
import torchmetrics.functional.image as tfi


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss_name, *args, **kwargs):
        super().__init__()
        self.register_hyperparameters()
        self.load_network()
        self.load_loss()

    def register_hyperparameters(self):
        self.save_hyperparameters()
        self.network_hparams = EasyDict([(hparam.split(".")[-1], v) for hparam, v in self.hparams.items() if
                                         hparam.startswith("init_args.network")])
        self.optim_hparams = EasyDict([(hparam.split(".")[-1], v) for hparam, v in self.hparams.items() if
                                       hparam.startswith("init_args.optim")])

    def forward(self, lr_hsi, hr_rgb):
        return self.model(lr_hsi, hr_rgb)

    def training_step(self, batch, batch_idx):
        lr, hr, _ = batch
        sr = self.network(lr)
        loss = self.loss_function(sr, hr)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target, _ = batch
        self.model_output = self.network(input)  # self(lr, hr[:, self.rgb_index, ])
        psnr = tfi.peak_signal_noise_ratio(preds=self.model_output, target=target, data_range=(-1, 1))
        ssim = tfi.structural_similarity_index_measure(preds=self.model_output, target=target, data_range=(-1, 1))
        self.log('psnr', psnr, on_step=True, on_epoch=True, prog_bar=True)
        self.log('ssim', ssim, on_step=True, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        self.logger.log_image(key=f'pred_images', images=to_pil(self.model_output[0, :]),
                              caption=[f"epoch_{self.current_epoch}"])

    def configure_optimizers(self):
        weight_decay = self.optim_hparams.weight_decay if hasattr(self.optim_hparams, 'weight_decay') else 0.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_hparams.lr, weight_decay=weight_decay)

        if self.optim_hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.optim_hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.optim_hparams.lr_decay_steps,
                                       gamma=self.optim_hparams.lr_decay_rate)
            elif self.optim_hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.optim_hparams.lr_decay_steps,
                                                  eta_min=self.optim_hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def load_loss(self):
        name = self.hparams.loss_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            loss = getattr(importlib.import_module(
                'loss.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.loss_function = loss()

    def load_network(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Network = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.network = self.instancialize(Network)

    def instancialize(self, Network):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Network.__init__).args[1:]
        args = EasyDict()
        for arg in class_args:
            if arg in self.network_hparams:
                args[arg] = self.network_hparams[arg]
        return Network(**self.network_hparams)
