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
from datasets import Dataset, DatasetDict, DatasetInfo, IterableDataset
from datasets.features.features import FeatureType, Features, Value, ClassLabel, _check_if_features_can_be_aligned
from promptsource.templates import DatasetTemplates, Template

logger = logging.get_logger(__name__)

DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)

__all__ = [
            'Flantastic_Mixture',
            'Flantastic_Template',
           ]

_DEFAULT_INPUT_FEATURE_NAME = 'INPUT'
_DEFAULT_OUTPUT_FEATURE_NAME = 'OUTPUT'

class IllegalArgumentError(ValueError):
    pass
class MissingArgumentError(ValueError):
    pass

@dataclass
class Flantastic_Template:
    template: Template
    input_feature_name: str = _DEFAULT_INPUT_FEATURE_NAME
    output_feature_name: str = _DEFAULT_OUTPUT_FEATURE_NAME

    def apply(self, dataset: Union[Dataset, DatasetDict], columns_to_remove: List[str]=None) -> Union[Dataset, DatasetDict]:
        def _apply_template(x):
            input_result, output_result = [], []
            features = list(x.keys())
            for idx, _ in enumerate(x[features[0]]):
                example = {features[0]: x[features[0]][idx]}
                for ft in features[1:]:
                    example.update({ft: x[ft][idx]})
                result = self.template.apply(example)
                input_result.append(result[0])
                output_result.append(result[1])
            return {
                self.input_feature_name: input_result,
                self.output_feature_name: output_result,
            }
        dataset_with_prompt = dataset.map(_apply_template, batched=True, remove_columns=columns_to_remove)
        return dataset_with_prompt
    
    def eval(self, features: Features) -> Tuple[Features]:
        return Features({
            self.input_feature_name: Value(dtype='string'),
            self.output_feature_name: Value(dtype='string'),
        })
    
class Flantastic_Mixture:
    def __init__(
        self, 
        mixture_list: List[
            Dict[str, Union[
                float,                          # 'ratio'
                datasets.DatasetBuilder,        # 'builder'
                List[                           # 'templates'
                    Dict[str, Union[
                        str,                    # 'template_name', 'input_feature_name', 'output_feature_name'
                        Literal[True, False]    # 'remove_applied_features'
                    ]]
                ]
            ]]
        ]
    ) -> None:
        if not isinstance(mixture_list, list):
            raise IllegalArgumentError('mixture_list must be a list')
        elif all('builder' not in cmpnt.keys() or 'ratio' not in cmpnt.keys() for cmpnt in mixture_list):
            raise IllegalArgumentError('mixture_list must be a list of dict with keys "builder" and "ratio"')
        elif not all([isinstance(cmpnt['builder'], datasets.DatasetBuilder) for cmpnt in mixture_list]):
            raise IllegalArgumentError('mixture_list must contain only DatasetBuilder instances in "builder" key')
        elif not all([isinstance(cmpnt['ratio'], float) for cmpnt in mixture_list]):
            raise IllegalArgumentError('mixture_list must contain only float in "ratio" key')
        self._mixture = mixture_list
        splits = []
        try:
            for cmpnt in mixture_list:
                splits.extend([*cmpnt['builder'].info.splits.keys()])
        except:
            splits = ['']
        self.splits = set(splits)
        self.builder_list_per_split = {}
        for sp in self.splits:
            for cmpnt in mixture_list:
                if sp in cmpnt['builder'].info.splits:
                    if sp in self.builder_list_per_split:
                        self.builder_list_per_split[sp].append(cmpnt['builder'])
                    else:
                        self.builder_list_per_split[sp] = [cmpnt['builder']]
        # Check for config names of each component.
        builderconfig_default_name = datasets.BuilderConfig().name
        for cmpnt in mixture_list:
            if cmpnt["builder"].info.config_name == builderconfig_default_name:
                cmpnt["builder"].info.config_name = cmpnt["builder"].name
        self.templates_dict = {}
        for cmpnt in mixture_list:
            template_list = []
            if cmpnt["templates"]:
                if not isinstance(cmpnt["templates"], list):
                    raise IllegalArgumentError('Templates must be a list of strings in the form "dataset_name/template_name" or "dataset_name/subset_name/template_name".')
                for tname in cmpnt["templates"]:
                    if not isinstance(tname, str):
                        raise IllegalArgumentError('Templates must be a list of strings in the form "dataset_name/template_name" or "dataset_name/subset_name/template_name".')
                    split_prompt_name = tname.split("/")
                    if len(split_prompt_name) == 2:
                        dataset_name, template_name = split_prompt_name
                        template: Template = DatasetTemplates(dataset_name)[template_name]
                    elif len(split_prompt_name) == 3:
                        dataset_name, subset_name, template_name = split_prompt_name
                        template: Template = DatasetTemplates(f"{dataset_name}/{subset_name}")[template_name]
                    else:
                        raise IllegalArgumentError("Prompt name must be of the form 'dataset_name/template_name' or 'dataset_name/subset_name/template_name' in this current version of Flantastic_Mixture.")
                    template_list.append(Flantastic_Template(template=template))
            self.templates_dict[cmpnt['builder'].info.config_name] = template_list    

    def __iter__(self):
        return iter(self._mixture)
    def __len__(self):
        return len(self._mixture)
    def __getitem__(self, index):
        return self._mixture[index] 
    def items(self) -> List[Dict[str, Union[float, datasets.DatasetBuilder]]]:
        return self._mixture
    def builder_names(self) -> List[str]:
        return [cmpnt['builder'].name for cmpnt in self.__iter__()]
    def config_names(self) -> List[str]:
        return [cmpnt['builder'].info.config_name for cmpnt in self.__iter__()]
    def versions(self) -> Dict[str, str]:
        return {cmpnt['builder'].info.config_name: cmpnt['builder'].info.version for cmpnt in self.__iter__()}
    def hashes(self) -> Dict[str, Optional[str]]:
        return {cmpnt['builder'].info.config_name: cmpnt['builder'].hash for cmpnt in self.__iter__()}
    def base_paths(self) -> Dict[str, str]:
        return {cmpnt['builder'].info.config_name: cmpnt['builder'].base_path for cmpnt in self.__iter__()}
    def ratios(self) -> Dict[str, float]:
        return {cmpnt['builder'].info.config_name: cmpnt['ratio'] for cmpnt in self.__iter__()}
    def builders(self) -> List[datasets.DatasetBuilder]:
        return [cmpnt['builder'] for cmpnt in self.__iter__()]
    def configs(self) -> List[datasets.BuilderConfig]:
        return [cmpnt['builder'].config for cmpnt in self.__iter__()]

