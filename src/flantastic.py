import datasets
from dataclasses import dataclass
import functools
import itertools
from typing import Dict, List, Union, Optional, TypeVar, Tuple, Callable
from datasets.utils import logging
from datasets.combine import interleave_datasets
from datasets import Dataset, DatasetDict, DatasetInfo, IterableDataset
from datasets.features.features import FeatureType, Features, Value, ClassLabel

logger = logging.get_logger(__name__)

DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)

__all__ = [
            'flantastic',
            'flantastic_mixture',
           ]

ORIGINAL_METHOD_SUFFIX = '_original__'
DEFAULT_LABEL_FEATURE_REPLACEMENT = '_string'
_DEFAULT_DESCRIPTION = ''
_DEFAULT_CITATION = ''


"""
Outstanding issues:
- Create an internal class structure for _helper_align_features output.
- Urgent: Be able to handle datasets with missing splits.  Two errors are observed:
    - With tb.download_and_prepare(), the error occurs in _force_align_features when handling all splits.
    - With tb.download_and_prepare(split='train'), the error occurs in the following line in the method ds_dataset:
        dataset_list = [cmpnt["builder"].as_dataset(*args, **kwargs) for cmpnt in self.__BUILDER_MIXTURE__]
"""

class IllegalArgumentError(ValueError):
    pass
class MissingArgumentError(ValueError):
    pass

class flantastic_mixture:
    def __init__(self, mixture_list: List[Dict[str, Union[float, datasets.DatasetBuilder]]]) -> None:
        # A clumsy way to check mixture_list.
        if  not isinstance(mixture_list, list):
            return IllegalArgumentError('mixture_list must be a list')
        elif all('builder' not in cmpnt.keys() or 'ratio' not in cmpnt.keys() for cmpnt in mixture_list):
            return IllegalArgumentError('mixture_list must be a list of dict with keys "builder" and "ratio"')
        elif not all([isinstance(cmpnt['builder'], datasets.DatasetBuilder) for cmpnt in mixture_list]):
            return IllegalArgumentError('mixture_list must contain only DatasetBuilder instances in "builder" key')
        elif not all([isinstance(cmpnt['ratio'], float) for cmpnt in mixture_list]):
            return IllegalArgumentError('mixture_list must contain only float in "ratio" key')
        self._mixture = mixture_list
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

def _helper_align_features(features_list: List[Features]) -> Tuple[str, Union[int, str, str, Dict[str, FeatureType]]]:
    """ The main misalignment type happens with ClassLabel, where it can be ClassLabel(names=['entailment', 'not_entailment'])
    in one dataset and ClassLabel(names=['False', 'True']) in another dataset, as an example, with both defined for feature "label". 
    """
    name2feature, name2type = {}, {}
    for fidx, features in enumerate(features_list):
        for fname, ftype in features.items():
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
                    name2feature[fname][fvalue].append(fidx)
                else:
                    name2feature[fname][fvalue] = [fidx]
                name2type[fname].append(ftype)
            else:
                name2feature[fname] = {fvalue: [fidx]}
                name2type[fname] = [ftype]
    for fname, fdict in name2feature.items():
        if len(fdict) > 1:
            new_fname = fname + DEFAULT_LABEL_FEATURE_REPLACEMENT
            logger.warning(f"Feature {fname} has different types in the datasets.  We will rename the feature to {new_fname} in the datasets.")
            fidx_iter = itertools.chain(*fdict.values())
            for fidx in fidx_iter:
                if isinstance(features_list[fidx][fname], ClassLabel):
                    yield 'ClassLabel', fidx, new_fname, fname, name2type[fname].pop()
                else:
                    yield 'Else', fidx, new_fname, fname, name2type[fname].pop()
        else:
            yield 'Pass', "", "", fname, name2type[fname][0]

"""
Reminders:
- Do not be confused between DatasetBuilder instance name (self.name) and the name provided to load_dataset or load_dataset_builder.  They are different.  Builder instance name is 'super_glue' and 
    the name provided to load_dataset is 'boolq', for example.  
    'boolq' can be obtained by builder_instance.info.config_name.  'super_glue' can be obtained by builder_instance.name or builder_instance.info.builder_name.
    builder_instance.info.name does not work.
"""
def flantastic(mixture: Union[flantastic_mixture, List[Dict[str, Union[float, datasets.DatasetBuilder]]]]=None) -> Callable:

    if mixture is None:
        return MissingArgumentError('mixture is required')
    elif isinstance(mixture, list):
        mixture = flantastic_mixture(mixture_list=mixture)
    elif not isinstance(mixture, flantastic_mixture):
        return IllegalArgumentError('mixture must be a list of dict or flantastic_mixture') 

    def class_decorator(cls: datasets.DatasetBuilder=None) -> Callable:
        if cls is None:
            return MissingArgumentError('cls is required')
        cls.__BUILDER_MIXTURE__: flantastic_mixture = mixture

        def _generic_wraps(new_fn):
            original_fn = getattr(cls, new_fn.__name__)
            @functools.wraps(original_fn)
            def wrap(self, *args, **kwargs):
                return new_fn(self, *args, **kwargs)
            setattr(cls, new_fn.__name__, wrap)
            setattr(cls, original_fn.__name__ + ORIGINAL_METHOD_SUFFIX, original_fn)
            return wrap

        @_generic_wraps
        def __init__(self, *args, **kwargs):
            self.feature_list = [builder.info.features for builder in self.__BUILDER_MIXTURE__.builders()] # Watch out for those datasets with missing splits.
            self.features_aligned_itr = list(_helper_align_features(self.feature_list))
            aligned_features = Features()
            for iter_obj in self.features_aligned_itr:
                if 'ClassLabel' in iter_obj[0] or 'Else' in iter_obj[0]:
                    aligned_features.update({iter_obj[2]: Value(dtype='string')})
                elif 'Pass' in iter_obj[0]:
                    aligned_features.update({iter_obj[3]: iter_obj[4]})
            if not hasattr(self, 'info'):
                self.info = datasets.DatasetInfo()
            self.info.features = aligned_features
            cls.__init___original__(self, *args, features=aligned_features, **kwargs, )

        @_generic_wraps
        def _create_builder_config(
            self, 
            name=None, 
            custom_features=None, 
            **config_kwargs,
        ) -> Tuple[datasets.BuilderConfig, str]:
            """Modified from original datasets.DatasetBuilder._create_builder_config to accomodate mixture.
            """
            if name is not None or len(self.__BUILDER_MIXTURE__) == 1:
                # Let original method handle the single case, where either it is implicitly specified by having only 1 config, or when it is explicitly specified by a name.
                if not self.BUILDER_CONFIGS:
                    self.BUILDER_CONFIGS: List[datasets.BuilderConfig] = self.__BUILDER_MIXTURE__.configs()
                return cls._create_builder_config_original__(self, name=name, custom_features=custom_features, **config_kwargs)
            else:
                """Original annotation: Create and validate BuilderConfig object as well as a unique config id for this config.
                Raises ValueError if there are multiple builder configs and name and DEFAULT_CONFIG_NAME are None.
                config_kwargs override the defaults kwargs in config
                """
                self.BUILDER_CONFIGS: List[datasets.BuilderConfig] = self.__BUILDER_MIXTURE__.configs() # Deprecate BUILDER_CONFIGS and rebuild it based on given mixture, in case it is needed by legacy code.
                self.NAME_RATIOS: Dict[str, float] = self.__BUILDER_MIXTURE__.ratios()
                builder_config = self.BUILDER_CONFIG_CLASS()
                if hasattr(self, "name"):
                    setattr(builder_config, "name", getattr(self, "name"))
                if hasattr(self, "VERSION"):
                    setattr(builder_config, "version", getattr(self, "VERSION"))
                setattr(builder_config, "description", "Flantastic mixture for datasets " + ", ".join(self.__BUILDER_MIXTURE__.config_names()))
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

        @_generic_wraps
        def as_dataset(self, *args, **kwargs) -> Union[Dataset, DatasetDict]:
            """ Modified from original datasets.DatasetBuilder.as_dataset to accomodate mixture.
                Used by load_dataset.
            """
            dataset_list = [cmpnt["builder"].as_dataset(*args, **kwargs) for cmpnt in self.__BUILDER_MIXTURE__]
            ratios = [cmpnt["ratio"] for cmpnt in self.__BUILDER_MIXTURE__]
            def _force_align_features(dataset_list):
                for iter_obj in self.features_aligned_itr:
                    if 'Else' in iter_obj[0] or 'ClassLabel' in iter_obj[0]:
                        if 'ClassLabel' in iter_obj[0]:
                            mapped_dataset = dataset_list[iter_obj[1]].map(lambda x: {iter_obj[2]: self.feature_list[iter_obj[1]][iter_obj[3]].names[x[iter_obj[3]]]}, remove_columns=[iter_obj[3]])
                        elif 'Else' in iter_obj[0]:
                            mapped_dataset = dataset_list[iter_obj[1]].map(lambda x: {iter_obj[2]: str(x[iter_obj[3]])}, remove_columns=[iter_obj[3]])
                        dataset_list[iter_obj[1]] = mapped_dataset
                return dataset_list
            if isinstance(dataset_list[0], DatasetDict):
                all_splits = list(set().union(*dataset_list))
                # For now, just do sampling on all splits.
                combined_dataset = DatasetDict({
                    split: interleave_datasets(
                        datasets=_force_align_features([datasetdict[split] for datasetdict in dataset_list if split in datasetdict.keys()]), 
                        probabilities=ratios, 
                        seed=self.seed, 
                        stopping_strategy='all_exhausted'
                    ) for split in all_splits
                })
            else:
                combined_dataset = interleave_datasets(
                    datasets=_force_align_features(dataset_list), 
                    probabilities=ratios, 
                    seed=self.seed, 
                    stopping_strategy='all_exhausted'
                )
            self.info.features = combined_dataset.features 
            return combined_dataset
        
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
        def _split_generators(self, *args, **kwargs) -> DatasetInfo:
            """Override this.
            """
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={})]

        return cls
    return class_decorator
