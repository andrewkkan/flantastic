# Copyright 2023 Flantastic Authors 
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
import datasets
from dataclasses import dataclass
import functools
import itertools
import re
import warnings
from tqdm import tqdm
from typing import Dict, List, Union, Optional, TypeVar, Tuple, Callable, Iterator, Literal
from datasets.utils import logging
from datasets.utils.info_utils import VerificationMode
from datasets.combine import interleave_datasets, concatenate_datasets
from datasets import Dataset, DatasetDict, DatasetInfo, IterableDataset
from datasets.features.features import FeatureType, Features, Value, ClassLabel, _check_if_features_can_be_aligned
from promptsource.templates import DatasetTemplates, Template
from .mixers import Flantastic_Mixture, Flantastic_Template

logger = logging.get_logger(__name__)

DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)

__all__ = [
            'flantastic',
           ]

ORIGINAL_METHOD_SUFFIX = '_original__'
DEFAULT_LABEL_FEATURE_REPLACEMENT = '_string'
_DEFAULT_DESCRIPTION = ''
_DEFAULT_CITATION = ''

class IllegalArgumentError(ValueError):
    pass
class MissingArgumentError(ValueError):
    pass

@dataclass
class _Feature_Alignment_Ouput:
    misalignment: str = "Pass" # "ClassLabel", "Else", "Pass"
    split: str = "" # "", "train", "validation", "test", etc.
    feature_idx_per_split: int = 0 # Each split has its own contigous feature indices 
    feature_name_new: str = "idx" # "label", "label_string", "idx", etc.
    feature_name_orig: str = "idx"
    feature_type_orig: FeatureType = Value(dtype='string') # Value(dtype='string'), ClassLabel(names=['False', 'True']), etc.

def _helper_align_features(features: Union[List[Features], Dict[str, List[Features]]]) -> Iterator[_Feature_Alignment_Ouput]:
    """ The main misalignment type happens with ClassLabel, where it can be ClassLabel(names=['entailment', 'not_entailment'])
    in one dataset and ClassLabel(names=['False', 'True']) in another dataset, as an example, with both defined for feature "label". 
    """
    if isinstance(features, list):
        splits = ['']
    elif isinstance(features, dict):
        splits = list(features.keys())
    name2feature, name2type = {}, {}
    for sp in splits:
        features_list = features[sp] if isinstance(features, dict) else features
        for fidx, fts in enumerate(features_list):
            for fname, ftype in fts.items():
                fvalue = ftype.__class__.__name__
                if isinstance(ftype, Value):
                    fvalue = "__".join([fvalue, 
                        str(ftype.dtype),
                    ])
                elif isinstance(ftype, ClassLabel):
                    fvalue = "__".join([fvalue, 
                        str(ftype.names),
                    ])
                if fname in name2feature:
                    if fvalue in name2feature[fname]:
                        name2feature[fname][fvalue].append((sp, fidx))
                    else:
                        name2feature[fname][fvalue] = [(sp, fidx)]
                    name2type[fname].append(ftype)
                else:
                    name2feature[fname] = {fvalue: [(sp, fidx)]}
                    name2type[fname] = [ftype]
    for fname, fdict in name2feature.items():
        fidx_iter = itertools.chain(*fdict.values())
        if len(fdict) > 1:
            new_fname = fname + DEFAULT_LABEL_FEATURE_REPLACEMENT
            logger.warning(f"Feature {fname} has different types in the datasets.  We will rename the feature to {new_fname} in the datasets.")
            for sp, fidx in fidx_iter:
                if isinstance(features[sp][fidx][fname], ClassLabel):
                    yield _Feature_Alignment_Ouput(
                        misalignment="ClassLabel",
                        split=sp,
                        feature_idx_per_split=fidx,
                        feature_name_new=new_fname,
                        feature_name_orig=fname,
                        feature_type_orig=name2type[fname].pop(),
                    )
                else:
                    yield _Feature_Alignment_Ouput(
                        misalignment="Else",
                        split=sp,
                        feature_idx_per_split=fidx,
                        feature_name_new=new_fname,
                        feature_name_orig=fname,
                        feature_type_orig=name2type[fname].pop(),
                    )
        else:
            fidx = next(fidx_iter)
            yield _Feature_Alignment_Ouput(
                misalignment="Pass",
                split="",
                feature_idx_per_split=0,
                feature_name_new="",
                feature_name_orig=fname,
                feature_type_orig=name2type[fname][0],
            )

def _modify_feature_list_by_template(
    builder_list_per_split: Dict[str, List[datasets.DatasetBuilder]], 
    templates_dict: Dict[str, List[Flantastic_Template]],
) -> Dict[str, List[Features]]:
    """ 
    This function modifies the feature list of each dataset in the mixture, based on the templates_dict.  
    The input builder_list_per_split is a dict of lists of builders, where the key is the split name and the value is the list of builders.
    The input templates_dict is a dict of lists of templates, where the key is the config_name of the builder and the value is the list of templates.
    The output is a dict of lists of features, where the key is the split name and the value is the list of features.
    """
    features_dict = {}
    for sp, builder_list in builder_list_per_split.items():
        features_list = []
        for builder in builder_list:
            if sp not in builder.info.splits:
                continue
            features = builder.info.features
            if templates_dict and templates_dict[builder.info.config_name]:
                for template in templates_dict[builder.info.config_name]:
                    modified_features = template.eval(features)
                    features_list.append(modified_features)
        features_dict[sp] = features_list
    return features_dict

def _normalize_ratios(nums: List[float]) -> List[float]:
    s = sum(nums)
    return [num/s for num in nums]

"""
Reminders for flantastic devs:
- Do not be confused between DatasetBuilder instance name (self.name) and the name provided to load_dataset or load_dataset_builder.  They are different.  Builder instance name is 'super_glue' and 
    the name provided to load_dataset is 'boolq', for example.  
    'boolq' can be obtained by builder_instance.info.config_name.  'super_glue' can be obtained by builder_instance.name or builder_instance.info.builder_name.
    builder_instance.info.name does not work.
"""
def flantastic(mixture: Flantastic_Mixture=None) -> Callable:

    if mixture is None:
        return MissingArgumentError('mixture is required')
    elif isinstance(mixture, list):
        mixture = Flantastic_Mixture(mixture_list=mixture)
    elif not isinstance(mixture, Flantastic_Mixture):
        return IllegalArgumentError('mixture must be a list of dict or Flantastic_Mixture') 

    def class_decorator(cls: datasets.DatasetBuilder=None) -> Callable:
        if cls is None:
            return MissingArgumentError('cls is required')
        cls.__BUILDER_MIXTURE__: Flantastic_Mixture = mixture
        cls.__NAME__: str = cls.__name__
        cls.__VERSION__: str = "0.0.0"

        def _generic_wraps(new_fn):
            try:
                original_fn = getattr(cls, new_fn.__name__)
            except AttributeError:
                def wrap(self, *args, **kwargs):
                    return new_fn(self, *args, **kwargs)
            else:
                setattr(cls, original_fn.__name__ + ORIGINAL_METHOD_SUFFIX, original_fn)
                @functools.wraps(original_fn)
                def wrap(self, *args, **kwargs):
                    return new_fn(self, *args, **kwargs)
            setattr(cls, new_fn.__name__, wrap)
            return wrap

        @_generic_wraps
        def __init__(self, *args, **kwargs):
            self.modified_feature_list_per_split = _modify_feature_list_by_template(self.__BUILDER_MIXTURE__.builder_list_per_split, self.__BUILDER_MIXTURE__.templates_dict)
            self.features_aligned_itr: List[_Feature_Alignment_Ouput] = list(_helper_align_features(self.modified_feature_list_per_split))
            aligned_features = Features()
            for fao in self.features_aligned_itr:
                if 'ClassLabel' in fao.misalignment or 'Else' in fao.misalignment:
                    aligned_features.update({fao.feature_name_new: Value(dtype='string')})
                elif 'Pass' in fao.misalignment:
                    aligned_features.update({fao.feature_name_orig: fao.feature_type_orig})
            if not hasattr(self, 'info'):
                self.info = datasets.DatasetInfo()
            self.info.features = aligned_features
            self.seed = 42
            cls.__init___original__(self, *args, features=aligned_features, **kwargs, )
            if not self.info.splits:
                if 'split' in kwargs:
                    self.info.splits = datasets.SplitDict({kwargs['split']: datasets.SplitInfo(name=kwargs['split'])})
                else:
                    self.info.splits = datasets.SplitDict({splitname: datasets.SplitInfo(name=splitname) for splitname in self.__BUILDER_MIXTURE__.splits})

        @_generic_wraps
        def _create_builder_config(
            self, 
            config_name=None, 
            custom_features=None, 
            **config_kwargs,
        ) -> Tuple[datasets.BuilderConfig, str]:
            """Modified from original datasets.DatasetBuilder._create_builder_config to accomodate mixture.
            """
            """Original annotation: Create and validate BuilderConfig object as well as a unique config id for this config.
            Raises ValueError if there are multiple builder configs and name and DEFAULT_CONFIG_NAME are None.
            config_kwargs override the defaults kwargs in config
            """
            self.BUILDER_CONFIGS: List[datasets.BuilderConfig] = self.__BUILDER_MIXTURE__.configs() # Deprecate BUILDER_CONFIGS and rebuild it based on given mixture, in case it is needed by legacy code.
            self.NAME_RATIOS: Dict[str, float] = self.__BUILDER_MIXTURE__.ratios()
            builder_config = self.BUILDER_CONFIG_CLASS()
            if hasattr(self, "name"):
                setattr(builder_config, "name", getattr(self, "__NAME__"))
            if hasattr(self, "version"):
                setattr(builder_config, "version", getattr(self, "__VERSION__"))
            setattr(builder_config, "description", "Flantastic mixture for datasets " + ", ".join(self.__BUILDER_MIXTURE__.config_names()))
            setattr(builder_config, "mixture_seed", str(self.__BUILDER_MIXTURE__.ratios())+str(self.seed)) # This is intended to pass a unique signature to create_config_id, so that the config_id is unique for each mixture.
            if config_kwargs:
                # builder_config = copy.deepcopy(builder_config) # This is not needed, since we are not modifying the original builder_config from config_dict, i.e. we never called self.builder_configs.get(name).
                for key, value in config_kwargs.items():
                    if value is not None:
                        if not hasattr(builder_config, key):
                            raise ValueError(f"BuilderConfig {builder_config} doesn't have a '{key}' key.")
                        setattr(builder_config, key, value)
                config_kwargs.update(builder_config.__dict__) # config_kwargs still takes precedence, but whatever it doesn't have, we get from builder_config.
            else:
                config_kwargs = builder_config.__dict__
            # compute the config id that is going to be used for caching
            config_id = builder_config.create_config_id(config_kwargs, custom_features=custom_features)
            is_custom = (config_id not in self.builder_configs) and config_id != "default"
            if is_custom:
                logger.info(f"Using custom data configuration {config_id}")
            else:
                if (
                    builder_config.name in self.builder_configs
                    and builder_config != self.builder_configs[builder_config.name]
                ):
                    raise ValueError(
                        "Cannot name a custom BuilderConfig the same as an available "
                        f"BuilderConfig. Change the name. Available BuilderConfigs: {list(self.builder_configs.keys())}"
                    )
                if not builder_config.version:
                    raise ValueError(f"BuilderConfig {builder_config.name} must have a version")
                # if not builder_config.description:
                #     raise ValueError(f"BuilderConfig {builder_config.name} must have a description"  )
            return builder_config, config_id

        @_generic_wraps
        def download_and_prepare(self, *args, **kwargs) -> None:
            """ Modified from original datasets.DatasetBuilder.download_and_prepare to accomodate mixture.
                Used by load_dataset. 
            """
            for cmpnt in self.__BUILDER_MIXTURE__:
                cmpnt["builder"].download_and_prepare(*args, **kwargs)
            # The original download_and_prepare will call _split_generators and _generate_examples when a previously cached copy is not found.
            cls.download_and_prepare_original__(self, *args, verification_mode = VerificationMode.NO_CHECKS, **kwargs)
        
        @_generic_wraps
        def _save_infos(self, *args, **kwargs) -> DatasetInfo:
            """ Modified from original datasets.DatasetBuilder._safe_infos to accomodate mixture.
                Used by load_dataset.
            """
            is_local = not is_remote_filesystem(self._fs)
            if is_local:
                lock_path = self._output_dir + "_infos.lock"
            with FileLock(lock_path) if is_local else contextlib.nullcontext():
                DatasetInfosDict(**{self.config.name: self.info}).write_to_directory(self.get_imported_module_dir())
            return cls._safe_infos_original__(self, *args, **kwargs)
        
        @_generic_wraps
        def _info(self, *args, **kwargs) -> DatasetInfo:
            """Override this.
            """
            return datasets.DatasetInfo(
                description=_DEFAULT_DESCRIPTION + self.config.description,
                features=datasets.Features(self.info.features), 
                # homepage=self.config.url,
                # citation=self.config.citation + "\n" + _DEFAULT_CITATION,
            )

        @_generic_wraps
        def _flantastic_magic(self):
            def _force_align_features(dataset_list: List[Dataset], split=None) -> List[Dataset]:
                if not split:
                    split = ''
                for fao in self.features_aligned_itr:
                    if split in fao.split:
                        if 'Else' in fao.misalignment or 'ClassLabel' in fao.misalignment:
                            if 'ClassLabel' in fao.misalignment:
                                mapped_dataset = dataset_list[fao.feature_idx_per_split].map(lambda x: {fao.feature_name_new: self.modified_feature_list_per_split[split][fao.feature_idx_per_split][fao.feature_name_orig].names[x[fao.feature_name_orig]]}, remove_columns=[fao.feature_name_orig])
                            elif 'Else' in fao.misalignment:
                                mapped_dataset = dataset_list[fao.feature_idx_per_split].map(lambda x: {fao.feature_name_new: str(x[fao.feature_name_orig])}, remove_columns=[fao.feature_name_orig])
                            dataset_list[fao.feature_idx_per_split] = mapped_dataset
                return dataset_list

            dataset_list = []
            for cmpnt in self.__BUILDER_MIXTURE__:
                dataset = cmpnt["builder"].as_dataset()
                template_list = self.__BUILDER_MIXTURE__.templates_dict[cmpnt["builder"].info.config_name]
                if template_list:
                    dataset_with_prompts = []
                    for template in template_list:
                        fnames = cmpnt["builder"].info.features.keys() # Original feature names
                        # template_keywords = " ".join(re.findall(r"\{\{(.+)\}\}", template.template.jinja))
                        columns_to_remove = fnames # [fname for fname in fnames if fname in template_keywords]
                        dataset_with_prompts.append(
                            template.apply(dataset, columns_to_remove=columns_to_remove), 
                        )
                    if isinstance(dataset, DatasetDict):
                        dataset_dict = DatasetDict()
                        for split in dataset.keys():
                            dataset_dict.update({
                                split: concatenate_datasets([ddict[split] for ddict in dataset_with_prompts])
                            })
                        dataset_list.append(dataset_dict)
                    else:
                        dataset_list.append(concatenate_datasets(dataset_with_prompts))
                else:
                    dataset_list.append(dataset)

            # dataset_list = [cmpnt["builder"].as_dataset(*args, **kwargs) for cmpnt in self.__BUILDER_MIXTURE__]             
            all_splits = list(set().union(*dataset_list))
            self.combined_dataset = DatasetDict({
                split: interleave_datasets(
                    datasets=_force_align_features([datasetdict[split] for datasetdict in dataset_list if split in datasetdict.keys()], split), 
                    probabilities=_normalize_ratios([self.NAME_RATIOS[builder.info.config_name] for builder in self.__BUILDER_MIXTURE__.builder_list_per_split[split]]), 
                    seed=self.seed, 
                    stopping_strategy='all_exhausted'
                ) for split in all_splits
            })

        @_generic_wraps
        def _split_generators(self, *args, **kwargs):
            try: 
                self.combined_dataset
            except:
                self._flantastic_magic()
            generator_list = []
            for split in self.__BUILDER_MIXTURE__.splits:
                generator_list.append(
                    datasets.SplitGenerator(
                        name=split,
                        gen_kwargs={'dataset': self.combined_dataset[split]},
                    )
                )
            return generator_list
        
        @_generic_wraps
        def _generate_examples(self, dataset):
            # For non Seq2Seq tasks, the features need to be modified from "INPUT" and "OUTPUT", to "text" and "label" for example.
            for id_, row in enumerate(dataset):
                yield id_, {"INPUT": row['INPUT'], "OUTPUT": row['OUTPUT']}

        return cls
    return class_decorator
