# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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
""" Longformer configuration """

import logging
from typing import List, Union

from .configuration_roberta import RobertaConfig


logger = logging.getLogger(__name__)

LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "allenai/longformer-base-4096": "https://s3.amazonaws.com/models.huggingface.co/bert/allenai/longformer-base-4096/config.json",
    "allenai/longformer-large-4096": "https://s3.amazonaws.com/models.huggingface.co/bert/allenai/longformer-large-4096/config.json",
    "allenai/longformer-large-4096-finetuned-triviaqa": "https://s3.amazonaws.com/models.huggingface.co/bert/allenai/longformer-large-4096-finetuned-triviaqa/config.json",
    "allenai/longformer-base-4096-extra.pos.embd.only": "https://s3.amazonaws.com/models.huggingface.co/bert/allenai/longformer-base-4096-extra.pos.embd.only/config.json",
    "allenai/longformer-large-4096-extra.pos.embd.only": "https://s3.amazonaws.com/models.huggingface.co/bert/allenai/longformer-large-4096-extra.pos.embd.only/config.json",
}


class LongformerConfig(RobertaConfig):
    r"""
        This is the configuration class to store the configuration of a :class:`~transformers1.LongformerModel`.
        It is used to instantiate an Longformer nezha_model according to the specified arguments, defining the nezha_model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the RoBERTa `roberta-base <https://huggingface.co/roberta-base>`__ architecture with a sequence length 4,096.

        The :class:`~transformers1.LongformerConfig` class directly inherits :class:`~transformers1.RobertaConfig`.
        It reuses the same defaults. Please check the parent class for more information.

        Args:
            attention_window (:obj:`int` or :obj:`List[int]`, optional, defaults to 512):
                Size of an attention window around each token. If :obj:`int`, use the same size for all layers.
                To specify a different window size for each layer, use a :obj:`List[int]` where
                ``len(attention_window) == num_hidden_layers``.

        Example::

            from transformers1 import LongformerConfig, LongformerModel

            # Initializing a Longformer configuration
            configuration = LongformerConfig()

            # Initializing a nezha_model from the configuration
            nezha_model = LongformerModel(configuration)

            # Accessing the nezha_model configuration
            configuration = nezha_model.config
    """
    model_type = "longformer"

    def __init__(self, attention_window: Union[List[int], int] = 512, sep_token_id: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.sep_token_id = sep_token_id
