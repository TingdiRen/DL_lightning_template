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

import torchvision.transforms.functional as tf


def to_pil(tensor_img, data_range=[-1, 1]):
    if (data_range[0] == -1) and (data_range[1] == 1):
        tensor_img = tensor_img / 2 + 0.5
        pil_img = tf.to_pil_image(tensor_img)
    elif (data_range[0] == 0) and (data_range[1] == 1):
        pil_img = tf.to_pil_image(tensor_img)
    return [pil_img]
