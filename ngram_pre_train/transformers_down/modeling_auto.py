# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Auto Model class. """


import logging
from collections import OrderedDict

from .configuration_auto import (
    AlbertConfig,
    AutoConfig,
    BartConfig,
    BertConfig,
    CamembertConfig,
    CTRLConfig,
    DistilBertConfig,
    ElectraConfig,
    EncoderDecoderConfig,
    FlaubertConfig,
    GPT2Config,
    LongformerConfig,
    OpenAIGPTConfig,
    ReformerConfig,
    RobertaConfig,
    T5Config,
    TransfoXLConfig,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
)
from .configuration_marian import MarianConfig
from .configuration_utils import PretrainedConfig
from .modeling_albert import (
    AlbertForMaskedLM,
    AlbertForPreTraining,
    AlbertForQuestionAnswering,
    AlbertForSequenceClassification,
    AlbertForTokenClassification,
    AlbertModel,
)
from .modeling_bart import BartForConditionalGeneration, BartForSequenceClassification, BartModel
from .modeling_bert import (
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForPreTraining,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
)
from .modeling_camembert import (
    CamembertForMaskedLM,
    CamembertForMultipleChoice,
    CamembertForSequenceClassification,
    CamembertForTokenClassification,
    CamembertModel,
)
from .modeling_ctrl import CTRLLMHeadModel, CTRLModel
from .modeling_distilbert import (
    DistilBertForMaskedLM,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertModel,
)
from .modeling_electra import (
    ElectraForMaskedLM,
    ElectraForPreTraining,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    ElectraModel,
)
from .modeling_encoder_decoder import EncoderDecoderModel
from .modeling_flaubert import (
    FlaubertForQuestionAnsweringSimple,
    FlaubertForSequenceClassification,
    FlaubertModel,
    FlaubertWithLMHeadModel,
)
from .modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from .modeling_longformer import (
    LongformerForMaskedLM,
    LongformerForMultipleChoice,
    LongformerForQuestionAnswering,
    LongformerForSequenceClassification,
    LongformerForTokenClassification,
    LongformerModel,
)
from .modeling_marian import MarianMTModel
from .modeling_openai import OpenAIGPTLMHeadModel, OpenAIGPTModel
from .modeling_reformer import ReformerModel, ReformerModelWithLMHead
from .modeling_roberta import (
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)
from .modeling_t5 import T5ForConditionalGeneration, T5Model
from .modeling_transfo_xl import TransfoXLLMHeadModel, TransfoXLModel
from .modeling_xlm import (
    XLMForQuestionAnsweringSimple,
    XLMForSequenceClassification,
    XLMForTokenClassification,
    XLMModel,
    XLMWithLMHeadModel,
)
from .modeling_xlm_roberta import (
    XLMRobertaForMaskedLM,
    XLMRobertaForMultipleChoice,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
    XLMRobertaModel,
)
from .modeling_xlnet import (
    XLNetForMultipleChoice,
    XLNetForQuestionAnsweringSimple,
    XLNetForSequenceClassification,
    XLNetForTokenClassification,
    XLNetLMHeadModel,
    XLNetModel,
)


logger = logging.getLogger(__name__)


MODEL_MAPPING = OrderedDict(
    [
        (T5Config, T5Model),
        (DistilBertConfig, DistilBertModel),
        (AlbertConfig, AlbertModel),
        (CamembertConfig, CamembertModel),
        (XLMRobertaConfig, XLMRobertaModel),
        (BartConfig, BartModel),
        (LongformerConfig, LongformerModel),
        (RobertaConfig, RobertaModel),
        (BertConfig, BertModel),
        (OpenAIGPTConfig, OpenAIGPTModel),
        (GPT2Config, GPT2Model),
        (TransfoXLConfig, TransfoXLModel),
        (XLNetConfig, XLNetModel),
        (FlaubertConfig, FlaubertModel),
        (XLMConfig, XLMModel),
        (CTRLConfig, CTRLModel),
        (ElectraConfig, ElectraModel),
        (ReformerConfig, ReformerModel),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING = OrderedDict(
    [
        (T5Config, T5ForConditionalGeneration),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForPreTraining),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (BartConfig, BartForConditionalGeneration),
        (LongformerConfig, LongformerForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (BertConfig, BertForPreTraining),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (CTRLConfig, CTRLLMHeadModel),
        (ElectraConfig, ElectraForPreTraining),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING = OrderedDict(
    [
        (T5Config, T5ForConditionalGeneration),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForMaskedLM),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (MarianConfig, MarianMTModel),
        (BartConfig, BartForConditionalGeneration),
        (LongformerConfig, LongformerForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (BertConfig, BertForMaskedLM),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (CTRLConfig, CTRLLMHeadModel),
        (ElectraConfig, ElectraForMaskedLM),
        (EncoderDecoderConfig, EncoderDecoderModel),
        (ReformerConfig, ReformerModelWithLMHead),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (DistilBertConfig, DistilBertForSequenceClassification),
        (AlbertConfig, AlbertForSequenceClassification),
        (CamembertConfig, CamembertForSequenceClassification),
        (XLMRobertaConfig, XLMRobertaForSequenceClassification),
        (BartConfig, BartForSequenceClassification),
        (LongformerConfig, LongformerForSequenceClassification),
        (RobertaConfig, RobertaForSequenceClassification),
        (BertConfig, BertForSequenceClassification),
        (XLNetConfig, XLNetForSequenceClassification),
        (FlaubertConfig, FlaubertForSequenceClassification),
        (XLMConfig, XLMForSequenceClassification),
        (ElectraConfig, ElectraForSequenceClassification),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        (DistilBertConfig, DistilBertForQuestionAnswering),
        (AlbertConfig, AlbertForQuestionAnswering),
        (LongformerConfig, LongformerForQuestionAnswering),
        (RobertaConfig, RobertaForQuestionAnswering),
        (BertConfig, BertForQuestionAnswering),
        (XLNetConfig, XLNetForQuestionAnsweringSimple),
        (FlaubertConfig, FlaubertForQuestionAnsweringSimple),
        (XLMConfig, XLMForQuestionAnsweringSimple),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (DistilBertConfig, DistilBertForTokenClassification),
        (CamembertConfig, CamembertForTokenClassification),
        (XLMConfig, XLMForTokenClassification),
        (XLMRobertaConfig, XLMRobertaForTokenClassification),
        (LongformerConfig, LongformerForTokenClassification),
        (RobertaConfig, RobertaForTokenClassification),
        (BertConfig, BertForTokenClassification),
        (XLNetConfig, XLNetForTokenClassification),
        (AlbertConfig, AlbertForTokenClassification),
        (ElectraConfig, ElectraForTokenClassification),
    ]
)


MODEL_FOR_MULTIPLE_CHOICE_MAPPING = OrderedDict(
    [
        (CamembertConfig, CamembertForMultipleChoice),
        (XLMRobertaConfig, XLMRobertaForMultipleChoice),
        (LongformerConfig, LongformerForMultipleChoice),
        (RobertaConfig, RobertaForMultipleChoice),
        (BertConfig, BertForMultipleChoice),
        (XLNetConfig, XLNetForMultipleChoice),
    ]
)


class AutoModel:
    r"""
        :class:`~transformers1.AutoModel` is a generic nezha_model class
        that will be instantiated as one of the base nezha_model classes of the library
        when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        or the `AutoModel.from_config(config)` class methods.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModel.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base nezha_model classes of the library
        from a configuration.

        Note:
            Loading a nezha_model from its configuration file does **not** load the nezha_model weights.
            It only affects the nezha_model's configuration. Use :func:`~transformers1.AutoModel.from_pretrained` to load
            the nezha_model weights

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The nezha_model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformers1.DistilBertModel` (DistilBERT nezha_model)
                - isInstance of `longformer` configuration class: :class:`~transformers1.LongformerModel` (Longformer nezha_model)
                - isInstance of `roberta` configuration class: :class:`~transformers1.RobertaModel` (RoBERTa nezha_model)
                - isInstance of `bert` configuration class: :class:`~transformers1.BertModel` (Bert nezha_model)
                - isInstance of `openai-gpt` configuration class: :class:`~transformers1.OpenAIGPTModel` (OpenAI GPT nezha_model)
                - isInstance of `gpt2` configuration class: :class:`~transformers1.GPT2Model` (OpenAI GPT-2 nezha_model)
                - isInstance of `ctrl` configuration class: :class:`~transformers1.CTRLModel` (Salesforce CTRL  nezha_model)
                - isInstance of `transfo-xl` configuration class: :class:`~transformers1.TransfoXLModel` (Transformer-XL nezha_model)
                - isInstance of `xlnet` configuration class: :class:`~transformers1.XLNetModel` (XLNet nezha_model)
                - isInstance of `xlm` configuration class: :class:`~transformers1.XLMModel` (XLM nezha_model)
                - isInstance of `flaubert` configuration class: :class:`~transformers1.FlaubertModel` (Flaubert nezha_model)
                - isInstance of `electra` configuration class: :class:`~transformers1.ElectraModel` (Electra nezha_model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            nezha_model = AutoModel.from_config(config)  # E.g. nezha_model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_MAPPING.keys())
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the base nezha_model classes of the library
        from a pre-trained nezha_model configuration.

        The `from_pretrained()` method takes care of returning the correct nezha_model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `t5`: :class:`~transformers1.T5Model` (T5 nezha_model)
            - `distilbert`: :class:`~transformers1.DistilBertModel` (DistilBERT nezha_model)
            - `albert`: :class:`~transformers1.AlbertModel` (ALBERT nezha_model)
            - `camembert`: :class:`~transformers1.CamembertModel` (CamemBERT nezha_model)
            - `xlm-roberta`: :class:`~transformers1.XLMRobertaModel` (XLM-RoBERTa nezha_model)
            - `longformer` :class:`~transformers1.LongformerModel` (Longformer nezha_model)
            - `roberta`: :class:`~transformers1.RobertaModel` (RoBERTa nezha_model)
            - `bert`: :class:`~transformers1.BertModel` (Bert nezha_model)
            - `openai-gpt`: :class:`~transformers1.OpenAIGPTModel` (OpenAI GPT nezha_model)
            - `gpt2`: :class:`~transformers1.GPT2Model` (OpenAI GPT-2 nezha_model)
            - `transfo-xl`: :class:`~transformers1.TransfoXLModel` (Transformer-XL nezha_model)
            - `xlnet`: :class:`~transformers1.XLNetModel` (XLNet nezha_model)
            - `xlm`: :class:`~transformers1.XLMModel` (XLM nezha_model)
            - `ctrl`: :class:`~transformers1.CTRLModel` (Salesforce CTRL  nezha_model)
            - `flaubert`: :class:`~transformers1.FlaubertModel` (Flaubert  nezha_model)
            - `electra`: :class:`~transformers1.ElectraModel` (Electra  nezha_model)

        The nezha_model is set in evaluation mode by default using `nezha_model.eval()` (Dropout modules are deactivated)
        To train the nezha_model, you should first set it back in training mode with `nezha_model.train()`

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained nezha_model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained nezha_model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing nezha_model weights saved using :func:`~transformers1.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/nezha_model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch nezha_model using the provided conversion scripts and loading the PyTorch nezha_model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying nezha_model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers1.PretrainedConfig`:
                Configuration for the nezha_model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the nezha_model is a nezha_model provided by the library (loaded with the ``shortcut-name`` string of a pretrained nezha_model), or
                - the nezha_model was saved using :func:`~transformers1.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the nezha_model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionary for the nezha_model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a nezha_model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers1.PreTrainedModel.save_pretrained` and :func:`~transformers1.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained nezha_model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the nezha_model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the nezha_model.

        Examples::

            nezha_model = AutoModel.from_pretrained('bert-base-uncased')    # Download nezha_model and configuration from S3 and cache.
            nezha_model = AutoModel.from_pretrained('./test/bert_model/')  # E.g. nezha_model was saved using `save_pretrained('./test/saved_model/')`
            assert nezha_model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch nezha_model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            nezha_model = AutoModel.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_MAPPING.keys())
            )
        )


class AutoModelForPreTraining:
    r"""
        :class:`~transformers1.AutoModelForPreTraining` is a generic nezha_model class
        that will be instantiated as one of the nezha_model classes of the library -with the architecture used for pretraining this nezha_model– when created with the `AutoModelForPreTraining.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForPreTraining is designed to be instantiated "
            "using the `AutoModelForPreTraining.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForPreTraining.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base nezha_model classes of the library
        from a configuration.

        Note:
            Loading a nezha_model from its configuration file does **not** load the nezha_model weights.
            It only affects the nezha_model's configuration. Use :func:`~transformers1.AutoModel.from_pretrained` to load
            the nezha_model weights

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The nezha_model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformers1.DistilBertForMaskedLM` (DistilBERT nezha_model)
                - isInstance of `longformer` configuration class: :class:`~transformers1.LongformerForMaskedLM` (Longformer nezha_model)
                - isInstance of `roberta` configuration class: :class:`~transformers1.RobertaForMaskedLM` (RoBERTa nezha_model)
                - isInstance of `bert` configuration class: :class:`~transformers1.BertForPreTraining` (Bert nezha_model)
                - isInstance of `openai-gpt` configuration class: :class:`~transformers1.OpenAIGPTLMHeadModel` (OpenAI GPT nezha_model)
                - isInstance of `gpt2` configuration class: :class:`~transformers1.GPT2LMHeadModel` (OpenAI GPT-2 nezha_model)
                - isInstance of `ctrl` configuration class: :class:`~transformers1.CTRLLMHeadModel` (Salesforce CTRL  nezha_model)
                - isInstance of `transfo-xl` configuration class: :class:`~transformers1.TransfoXLLMHeadModel` (Transformer-XL nezha_model)
                - isInstance of `xlnet` configuration class: :class:`~transformers1.XLNetLMHeadModel` (XLNet nezha_model)
                - isInstance of `xlm` configuration class: :class:`~transformers1.XLMWithLMHeadModel` (XLM nezha_model)
                - isInstance of `flaubert` configuration class: :class:`~transformers1.FlaubertWithLMHeadModel` (Flaubert nezha_model)
                - isInstance of `electra` configuration class: :class:`~transformers1.ElectraForPreTraining` (Electra nezha_model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            nezha_model = AutoModelForPreTraining.from_config(config)  # E.g. nezha_model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_FOR_PRETRAINING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_PRETRAINING_MAPPING.keys())
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the nezha_model classes of the library -with the architecture used for pretraining this nezha_model– from a pre-trained nezha_model configuration.

        The `from_pretrained()` method takes care of returning the correct nezha_model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `t5`: :class:`~transformers1.T5ModelWithLMHead` (T5 nezha_model)
            - `distilbert`: :class:`~transformers1.DistilBertForMaskedLM` (DistilBERT nezha_model)
            - `albert`: :class:`~transformers1.AlbertForMaskedLM` (ALBERT nezha_model)
            - `camembert`: :class:`~transformers1.CamembertForMaskedLM` (CamemBERT nezha_model)
            - `xlm-roberta`: :class:`~transformers1.XLMRobertaForMaskedLM` (XLM-RoBERTa nezha_model)
            - `longformer`: :class:`~transformers1.LongformerForMaskedLM` (Longformer nezha_model)
            - `roberta`: :class:`~transformers1.RobertaForMaskedLM` (RoBERTa nezha_model)
            - `bert`: :class:`~transformers1.BertForPreTraining` (Bert nezha_model)
            - `openai-gpt`: :class:`~transformers1.OpenAIGPTLMHeadModel` (OpenAI GPT nezha_model)
            - `gpt2`: :class:`~transformers1.GPT2LMHeadModel` (OpenAI GPT-2 nezha_model)
            - `transfo-xl`: :class:`~transformers1.TransfoXLLMHeadModel` (Transformer-XL nezha_model)
            - `xlnet`: :class:`~transformers1.XLNetLMHeadModel` (XLNet nezha_model)
            - `xlm`: :class:`~transformers1.XLMWithLMHeadModel` (XLM nezha_model)
            - `ctrl`: :class:`~transformers1.CTRLLMHeadModel` (Salesforce CTRL nezha_model)
            - `flaubert`: :class:`~transformers1.FlaubertWithLMHeadModel` (Flaubert nezha_model)
            - `electra`: :class:`~transformers1.ElectraForPreTraining` (Electra nezha_model)

        The nezha_model is set in evaluation mode by default using `nezha_model.eval()` (Dropout modules are deactivated)
        To train the nezha_model, you should first set it back in training mode with `nezha_model.train()`

        Args:
            pretrained_model_name_or_path:
                Either:

                - a string with the `shortcut name` of a pre-trained nezha_model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained nezha_model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing nezha_model weights saved using :func:`~transformers1.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/nezha_model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch nezha_model using the provided conversion scripts and loading the PyTorch nezha_model afterwards.
            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying nezha_model's ``__init__`` method
            config: (`optional`) instance of a class derived from :class:`~transformers1.PretrainedConfig`:
                Configuration for the nezha_model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the nezha_model is a nezha_model provided by the library (loaded with the ``shortcut-name`` string of a pretrained nezha_model), or
                - the nezha_model was saved using :func:`~transformers1.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the nezha_model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionary for the nezha_model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a nezha_model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers1.PreTrainedModel.save_pretrained` and :func:`~transformers1.PreTrainedModel.from_pretrained` is not a simpler option.
            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained nezha_model
                configuration should be cached if the standard cache should not be used.
            force_download: (`optional`) boolean, default False:
                Force to (re-)download the nezha_model weights and configuration files and override the cached versions if they exists.
            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely received file. Attempt to resume the download if such a file exists.
            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.
            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionary containing missing keys, unexpected keys and error messages.
            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the nezha_model.

        Examples::

            nezha_model = AutoModelForPreTraining.from_pretrained('bert-base-uncased')    # Download nezha_model and configuration from S3 and cache.
            nezha_model = AutoModelForPreTraining.from_pretrained('./test/bert_model/')  # E.g. nezha_model was saved using `save_pretrained('./test/saved_model/')`
            assert nezha_model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch nezha_model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            nezha_model = AutoModelForPreTraining.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_PRETRAINING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_PRETRAINING_MAPPING.keys())
            )
        )


class AutoModelWithLMHead:
    r"""
        :class:`~transformers1.AutoModelWithLMHead` is a generic nezha_model class
        that will be instantiated as one of the language modeling nezha_model classes of the library
        when created with the `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelWithLMHead is designed to be instantiated "
            "using the `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelWithLMHead.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base nezha_model classes of the library
        from a configuration.

        Note:
            Loading a nezha_model from its configuration file does **not** load the nezha_model weights.
            It only affects the nezha_model's configuration. Use :func:`~transformers1.AutoModel.from_pretrained` to load
            the nezha_model weights

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The nezha_model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformers1.DistilBertForMaskedLM` (DistilBERT nezha_model)
                - isInstance of `longformer` configuration class: :class:`~transformers1.LongformerForMaskedLM` (Longformer nezha_model)
                - isInstance of `roberta` configuration class: :class:`~transformers1.RobertaForMaskedLM` (RoBERTa nezha_model)
                - isInstance of `bert` configuration class: :class:`~transformers1.BertForMaskedLM` (Bert nezha_model)
                - isInstance of `openai-gpt` configuration class: :class:`~transformers1.OpenAIGPTLMHeadModel` (OpenAI GPT nezha_model)
                - isInstance of `gpt2` configuration class: :class:`~transformers1.GPT2LMHeadModel` (OpenAI GPT-2 nezha_model)
                - isInstance of `ctrl` configuration class: :class:`~transformers1.CTRLLMHeadModel` (Salesforce CTRL  nezha_model)
                - isInstance of `transfo-xl` configuration class: :class:`~transformers1.TransfoXLLMHeadModel` (Transformer-XL nezha_model)
                - isInstance of `xlnet` configuration class: :class:`~transformers1.XLNetLMHeadModel` (XLNet nezha_model)
                - isInstance of `xlm` configuration class: :class:`~transformers1.XLMWithLMHeadModel` (XLM nezha_model)
                - isInstance of `flaubert` configuration class: :class:`~transformers1.FlaubertWithLMHeadModel` (Flaubert nezha_model)
                - isInstance of `electra` configuration class: :class:`~transformers1.ElectraForMaskedLM` (Electra nezha_model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            nezha_model = AutoModelWithLMHead.from_config(config)  # E.g. nezha_model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_WITH_LM_HEAD_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_WITH_LM_HEAD_MAPPING.keys())
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the language modeling nezha_model classes of the library
        from a pre-trained nezha_model configuration.

        The `from_pretrained()` method takes care of returning the correct nezha_model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `t5`: :class:`~transformers1.T5ModelWithLMHead` (T5 nezha_model)
            - `distilbert`: :class:`~transformers1.DistilBertForMaskedLM` (DistilBERT nezha_model)
            - `albert`: :class:`~transformers1.AlbertForMaskedLM` (ALBERT nezha_model)
            - `camembert`: :class:`~transformers1.CamembertForMaskedLM` (CamemBERT nezha_model)
            - `xlm-roberta`: :class:`~transformers1.XLMRobertaForMaskedLM` (XLM-RoBERTa nezha_model)
            - `longformer`: :class:`~transformers1.LongformerForMaskedLM` (Longformer nezha_model)
            - `roberta`: :class:`~transformers1.RobertaForMaskedLM` (RoBERTa nezha_model)
            - `bert`: :class:`~transformers1.BertForMaskedLM` (Bert nezha_model)
            - `openai-gpt`: :class:`~transformers1.OpenAIGPTLMHeadModel` (OpenAI GPT nezha_model)
            - `gpt2`: :class:`~transformers1.GPT2LMHeadModel` (OpenAI GPT-2 nezha_model)
            - `transfo-xl`: :class:`~transformers1.TransfoXLLMHeadModel` (Transformer-XL nezha_model)
            - `xlnet`: :class:`~transformers1.XLNetLMHeadModel` (XLNet nezha_model)
            - `xlm`: :class:`~transformers1.XLMWithLMHeadModel` (XLM nezha_model)
            - `ctrl`: :class:`~transformers1.CTRLLMHeadModel` (Salesforce CTRL nezha_model)
            - `flaubert`: :class:`~transformers1.FlaubertWithLMHeadModel` (Flaubert nezha_model)
            - `electra`: :class:`~transformers1.ElectraForMaskedLM` (Electra nezha_model)

        The nezha_model is set in evaluation mode by default using `nezha_model.eval()` (Dropout modules are deactivated)
        To train the nezha_model, you should first set it back in training mode with `nezha_model.train()`

        Args:
            pretrained_model_name_or_path:
                Either:

                - a string with the `shortcut name` of a pre-trained nezha_model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained nezha_model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing nezha_model weights saved using :func:`~transformers1.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/nezha_model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch nezha_model using the provided conversion scripts and loading the PyTorch nezha_model afterwards.
            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying nezha_model's ``__init__`` method
            config: (`optional`) instance of a class derived from :class:`~transformers1.PretrainedConfig`:
                Configuration for the nezha_model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the nezha_model is a nezha_model provided by the library (loaded with the ``shortcut-name`` string of a pretrained nezha_model), or
                - the nezha_model was saved using :func:`~transformers1.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the nezha_model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionary for the nezha_model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a nezha_model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers1.PreTrainedModel.save_pretrained` and :func:`~transformers1.PreTrainedModel.from_pretrained` is not a simpler option.
            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained nezha_model
                configuration should be cached if the standard cache should not be used.
            force_download: (`optional`) boolean, default False:
                Force to (re-)download the nezha_model weights and configuration files and override the cached versions if they exists.
            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely received file. Attempt to resume the download if such a file exists.
            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.
            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionary containing missing keys, unexpected keys and error messages.
            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the nezha_model.

        Examples::

            nezha_model = AutoModelWithLMHead.from_pretrained('bert-base-uncased')    # Download nezha_model and configuration from S3 and cache.
            nezha_model = AutoModelWithLMHead.from_pretrained('./test/bert_model/')  # E.g. nezha_model was saved using `save_pretrained('./test/saved_model/')`
            assert nezha_model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch nezha_model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            nezha_model = AutoModelWithLMHead.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_WITH_LM_HEAD_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_WITH_LM_HEAD_MAPPING.keys())
            )
        )


class AutoModelForSequenceClassification:
    r"""
        :class:`~transformers1.AutoModelForSequenceClassification` is a generic nezha_model class
        that will be instantiated as one of the sequence classification nezha_model classes of the library
        when created with the `AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForSequenceClassification is designed to be instantiated "
            "using the `AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForSequenceClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base nezha_model classes of the library
        from a configuration.

        Note:
            Loading a nezha_model from its configuration file does **not** load the nezha_model weights.
            It only affects the nezha_model's configuration. Use :func:`~transformers1.AutoModel.from_pretrained` to load
            the nezha_model weights

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The nezha_model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformers1.DistilBertForSequenceClassification` (DistilBERT nezha_model)
                - isInstance of `albert` configuration class: :class:`~transformers1.AlbertForSequenceClassification` (ALBERT nezha_model)
                - isInstance of `camembert` configuration class: :class:`~transformers1.CamembertForSequenceClassification` (CamemBERT nezha_model)
                - isInstance of `xlm roberta` configuration class: :class:`~transformers1.XLMRobertaForSequenceClassification` (XLM-RoBERTa nezha_model)
                - isInstance of `roberta` configuration class: :class:`~transformers1.RobertaForSequenceClassification` (RoBERTa nezha_model)
                - isInstance of `bert` configuration class: :class:`~transformers1.BertForSequenceClassification` (Bert nezha_model)
                - isInstance of `xlnet` configuration class: :class:`~transformers1.XLNetForSequenceClassification` (XLNet nezha_model)
                - isInstance of `xlm` configuration class: :class:`~transformers1.XLMForSequenceClassification` (XLM nezha_model)
                - isInstance of `flaubert` configuration class: :class:`~transformers1.FlaubertForSequenceClassification` (Flaubert nezha_model)


        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            nezha_model = AutoModelForSequenceClassification.from_config(config)  # E.g. nezha_model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the sequence classification nezha_model classes of the library
        from a pre-trained nezha_model configuration.

        The `from_pretrained()` method takes care of returning the correct nezha_model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `distilbert`: :class:`~transformers1.DistilBertForSequenceClassification` (DistilBERT nezha_model)
            - `albert`: :class:`~transformers1.AlbertForSequenceClassification` (ALBERT nezha_model)
            - `camembert`: :class:`~transformers1.CamembertForSequenceClassification` (CamemBERT nezha_model)
            - `xlm-roberta`: :class:`~transformers1.XLMRobertaForSequenceClassification` (XLM-RoBERTa nezha_model)
            - `roberta`: :class:`~transformers1.RobertaForSequenceClassification` (RoBERTa nezha_model)
            - `bert`: :class:`~transformers1.BertForSequenceClassification` (Bert nezha_model)
            - `xlnet`: :class:`~transformers1.XLNetForSequenceClassification` (XLNet nezha_model)
            - `flaubert`: :class:`~transformers1.FlaubertForSequenceClassification` (Flaubert nezha_model)

        The nezha_model is set in evaluation mode by default using `nezha_model.eval()` (Dropout modules are deactivated)
        To train the nezha_model, you should first set it back in training mode with `nezha_model.train()`

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained nezha_model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained nezha_model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing nezha_model weights saved using :func:`~transformers1.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/nezha_model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch nezha_model using the provided conversion scripts and loading the PyTorch nezha_model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaining positional arguments will be passed to the underlying nezha_model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers1.PretrainedConfig`:
                Configuration for the nezha_model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the nezha_model is a nezha_model provided by the library (loaded with the ``shortcut-name`` string of a pretrained nezha_model), or
                - the nezha_model was saved using :func:`~transformers1.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the nezha_model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionary for the nezha_model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a nezha_model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers1.PreTrainedModel.save_pretrained` and :func:`~transformers1.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained nezha_model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the nezha_model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the nezha_model.

        Examples::

            nezha_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')    # Download nezha_model and configuration from S3 and cache.
            nezha_model = AutoModelForSequenceClassification.from_pretrained('./test/bert_model/')  # E.g. nezha_model was saved using `save_pretrained('./test/saved_model/')`
            assert nezha_model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch nezha_model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            nezha_model = AutoModelForSequenceClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys()),
            )
        )


class AutoModelForQuestionAnswering:
    r"""
        :class:`~transformers1.AutoModelForQuestionAnswering` is a generic nezha_model class
        that will be instantiated as one of the question answering nezha_model classes of the library
        when created with the `AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForQuestionAnswering is designed to be instantiated "
            "using the `AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForQuestionAnswering.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base nezha_model classes of the library
        from a configuration.

        Note:
            Loading a nezha_model from its configuration file does **not** load the nezha_model weights.
            It only affects the nezha_model's configuration. Use :func:`~transformers1.AutoModel.from_pretrained` to load
            the nezha_model weights

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The nezha_model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformers1.DistilBertForQuestionAnswering` (DistilBERT nezha_model)
                - isInstance of `albert` configuration class: :class:`~transformers1.AlbertForQuestionAnswering` (ALBERT nezha_model)
                - isInstance of `bert` configuration class: :class:`~transformers1.BertModelForQuestionAnswering` (Bert nezha_model)
                - isInstance of `xlnet` configuration class: :class:`~transformers1.XLNetForQuestionAnswering` (XLNet nezha_model)
                - isInstance of `xlm` configuration class: :class:`~transformers1.XLMForQuestionAnswering` (XLM nezha_model)
                - isInstance of `flaubert` configuration class: :class:`~transformers1.FlaubertForQuestionAnswering` (XLM nezha_model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            nezha_model = AutoModelForQuestionAnswering.from_config(config)  # E.g. nezha_model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_FOR_QUESTION_ANSWERING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the question answering nezha_model classes of the library
        from a pre-trained nezha_model configuration.

        The `from_pretrained()` method takes care of returning the correct nezha_model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `distilbert`: :class:`~transformers1.DistilBertForQuestionAnswering` (DistilBERT nezha_model)
            - `albert`: :class:`~transformers1.AlbertForQuestionAnswering` (ALBERT nezha_model)
            - `bert`: :class:`~transformers1.BertForQuestionAnswering` (Bert nezha_model)
            - `xlnet`: :class:`~transformers1.XLNetForQuestionAnswering` (XLNet nezha_model)
            - `xlm`: :class:`~transformers1.XLMForQuestionAnswering` (XLM nezha_model)
            - `flaubert`: :class:`~transformers1.FlaubertForQuestionAnswering` (XLM nezha_model)

        The nezha_model is set in evaluation mode by default using `nezha_model.eval()` (Dropout modules are deactivated)
        To train the nezha_model, you should first set it back in training mode with `nezha_model.train()`

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained nezha_model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained nezha_model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing nezha_model weights saved using :func:`~transformers1.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/nezha_model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch nezha_model using the provided conversion scripts and loading the PyTorch nezha_model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying nezha_model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers1.PretrainedConfig`:
                Configuration for the nezha_model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the nezha_model is a nezha_model provided by the library (loaded with the ``shortcut-name`` string of a pretrained nezha_model), or
                - the nezha_model was saved using :func:`~transformers1.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the nezha_model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionary for the nezha_model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a nezha_model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers1.PreTrainedModel.save_pretrained` and :func:`~transformers1.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained nezha_model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the nezha_model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the nezha_model.

        Examples::

            nezha_model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')    # Download nezha_model and configuration from S3 and cache.
            nezha_model = AutoModelForQuestionAnswering.from_pretrained('./test/bert_model/')  # E.g. nezha_model was saved using `save_pretrained('./test/saved_model/')`
            assert nezha_model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch nezha_model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            nezha_model = AutoModelForQuestionAnswering.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_QUESTION_ANSWERING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys()),
            )
        )


class AutoModelForTokenClassification:
    r"""
        :class:`~transformers1.AutoModelForTokenClassification` is a generic nezha_model class
        that will be instantiated as one of the token classification nezha_model classes of the library
        when created with the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTokenClassification is designed to be instantiated "
            "using the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTokenClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base nezha_model classes of the library
        from a configuration.

        Note:
            Loading a nezha_model from its configuration file does **not** load the nezha_model weights.
            It only affects the nezha_model's configuration. Use :func:`~transformers1.AutoModel.from_pretrained` to load
            the nezha_model weights

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The nezha_model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformers1.DistilBertModelForTokenClassification` (DistilBERT nezha_model)
                - isInstance of `xlm` configuration class: :class:`~transformers1.XLMForTokenClassification` (XLM nezha_model)
                - isInstance of `xlm roberta` configuration class: :class:`~transformers1.XLMRobertaModelForTokenClassification` (XLMRoberta nezha_model)
                - isInstance of `bert` configuration class: :class:`~transformers1.BertModelForTokenClassification` (Bert nezha_model)
                - isInstance of `albert` configuration class: :class:`~transformers1.AlbertForTokenClassification` (AlBert nezha_model)
                - isInstance of `xlnet` configuration class: :class:`~transformers1.XLNetModelForTokenClassification` (XLNet nezha_model)
                - isInstance of `camembert` configuration class: :class:`~transformers1.CamembertModelForTokenClassification` (Camembert nezha_model)
                - isInstance of `roberta` configuration class: :class:`~transformers1.RobertaModelForTokenClassification` (Roberta nezha_model)
                - isInstance of `electra` configuration class: :class:`~transformers1.ElectraForTokenClassification` (Electra nezha_model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            nezha_model = AutoModelForTokenClassification.from_config(config)  # E.g. nezha_model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the question answering nezha_model classes of the library
        from a pre-trained nezha_model configuration.

        The `from_pretrained()` method takes care of returning the correct nezha_model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `distilbert`: :class:`~transformers1.DistilBertForTokenClassification` (DistilBERT nezha_model)
            - `xlm`: :class:`~transformers1.XLMForTokenClassification` (XLM nezha_model)
            - `xlm-roberta`: :class:`~transformers1.XLMRobertaForTokenClassification` (XLM-RoBERTa?Para nezha_model)
            - `camembert`: :class:`~transformers1.CamembertForTokenClassification` (Camembert nezha_model)
            - `bert`: :class:`~transformers1.BertForTokenClassification` (Bert nezha_model)
            - `xlnet`: :class:`~transformers1.XLNetForTokenClassification` (XLNet nezha_model)
            - `roberta`: :class:`~transformers1.RobertaForTokenClassification` (Roberta nezha_model)
            - `electra`: :class:`~transformers1.ElectraForTokenClassification` (Electra nezha_model)

        The nezha_model is set in evaluation mode by default using `nezha_model.eval()` (Dropout modules are deactivated)
        To train the nezha_model, you should first set it back in training mode with `nezha_model.train()`

        Args:
            pretrained_model_name_or_path:
                Either:

                - a string with the `shortcut name` of a pre-trained nezha_model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing nezha_model weights saved using :func:`~transformers1.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/nezha_model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch nezha_model using the provided conversion scripts and loading the PyTorch nezha_model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying nezha_model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers1.PretrainedConfig`:
                Configuration for the nezha_model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the nezha_model is a nezha_model provided by the library (loaded with the ``shortcut-name`` string of a pretrained nezha_model), or
                - the nezha_model was saved using :func:`~transformers1.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the nezha_model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionary for the nezha_model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a nezha_model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers1.PreTrainedModel.save_pretrained` and :func:`~transformers1.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained nezha_model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the nezha_model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the nezha_model.

        Examples::

            nezha_model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased')    # Download nezha_model and configuration from S3 and cache.
            nezha_model = AutoModelForTokenClassification.from_pretrained('./test/bert_model/')  # E.g. nezha_model was saved using `save_pretrained('./test/saved_model/')`
            assert nezha_model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch nezha_model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            nezha_model = AutoModelForTokenClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys()),
            )
        )


class AutoModelForMultipleChoice:
    r"""
        :class:`~transformers1.AutoModelForMultipleChoice` is a generic nezha_model class
        that will be instantiated as one of the multiple choice nezha_model classes of the library
        when created with the `AutoModelForMultipleChoice.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForMultipleChoice is designed to be instantiated "
            "using the `AutoModelForMultipleChoice.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForMultipleChoice.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_FOR_MULTIPLE_CHOICE_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_MULTIPLE_CHOICE_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_MULTIPLE_CHOICE_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_MULTIPLE_CHOICE_MAPPING.keys()),
            )
        )
