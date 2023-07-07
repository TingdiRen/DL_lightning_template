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

from PIL import Image
from pathlib import Path
from natsort import natsorted
import torch.utils.data as data
import torchvision.transforms as transforms
import augly.image as imaugs


class PairedData(data.Dataset):
    def __init__(self, data_dir='dataset',
                 mode='train',
                 no_augment=True):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.get_files()

    def get_files(self):
        dataset_path = Path(self.data_dir, self.mode)
        for i in ["input", "target"]:
            setattr(self, f"{i}_dirs", natsorted((dataset_path / f"{i}").iterdir()))

    def __len__(self):
        return len(self.input_dirs)

    def __getitem__(self, idx):
        batch = {'input': None, 'target': None}
        for k in batch.keys():
            img = Image.open(getattr(self, f"{k}_dirs")[idx])
            if self.mode == 'train' and self.no_augment:
                img = self.img_aug(img)
            batch[k] = self.to_tensor(img)
        return *batch.values(), idx

    def img_aug(self, pil_img, avg_prob=0.15):
        augs = transforms.Compose([
            imaugs.Blur(p=avg_prob),
            imaugs.Rotate(p=avg_prob),
            imaugs.Saturation(p=avg_prob),
            imaugs.RandomNoise(p=avg_prob),
            imaugs.HFlip(p=avg_prob),
            imaugs.VFlip(p=avg_prob),
            imaugs.OverlayText(p=avg_prob)
        ])
        auged_img = augs(pil_img)
        return auged_img

    def to_tensor(self, pil_img):
        transform = transforms.Compose([imaugs.Resize(256, 256),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        tensor_img = transform(pil_img)
        return tensor_img
