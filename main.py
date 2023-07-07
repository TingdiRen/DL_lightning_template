# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
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

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from model import MInterface
from data import DInterface
from utils import get_time

if __name__ == '__main__':
    """ 
    This main entrance of the whole project.  
    """
    logger = WandbLogger(project="demo", name=f'train_{get_time()}', group=f"exp_name", save_dir='exps',
                         log_model="all")
    cli = LightningCLI(model_class=MInterface, datamodule_class=DInterface,
                       trainer_defaults={'logger': logger},
                       save_config_kwargs={"overwrite": True},
                       run=False)
    cli.trainer.fit(cli.model, cli.datamodule)
