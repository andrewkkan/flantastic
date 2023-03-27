import datasets
from dataclasses import dataclass
import functools
import itertools
from typing import Dict, List, Union, Optional, TypeVar, Tuple, Callable, Iterator
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
        splits = []
        try:
            for cmpnt in mixture_list:
                splits.extend([*cmpnt['builder'].info.splits.keys()])
        except:
            splits = ['']
        splits = set(splits)
        self.builder_list_per_split, self.feature_list_per_split = {}, {}
        for sp in splits:
            for cmpnt in mixture_list:
                if sp in cmpnt['builder'].info.splits:
                    if sp in self.builder_list_per_split:
                        self.builder_list_per_split[sp].append(cmpnt['builder'])
                        self.feature_list_per_split[sp].append(cmpnt['builder'].info.features)
                    else:
                        self.builder_list_per_split[sp] = [cmpnt['builder']]
                        self.feature_list_per_split[sp] = [cmpnt['builder'].info.features]

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
    def feature_list_per_split(self) -> Dict[str, List[Features]]:
        return self.feature_list_per_split
    def builder_list_per_split(self) -> Dict[str, List[datasets.DatasetBuilder]]:
        return self.builder_list_per_split

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

def _normalize_ratios(nums: List[float]) -> List[float]:
    s = sum(nums)
    return [num/s for num in nums]

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
            self.feature_list_per_split = self.__BUILDER_MIXTURE__.feature_list_per_split
            self.features_aligned_itr: List[_Feature_Alignment_Ouput] = list(_helper_align_features(self.feature_list_per_split))
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
            if 'split' in kwargs:
                dataset_list = [cmpnt["builder"].as_dataset(*args, **kwargs) for cmpnt in self.__BUILDER_MIXTURE__ if kwargs['split'] in cmpnt["builder"].info.splits]
            else:
                dataset_list = [cmpnt["builder"].as_dataset(*args, **kwargs) for cmpnt in self.__BUILDER_MIXTURE__]
            def _force_align_features(dataset_list, split=None):
                if not split:
                    split = ''
                for fao in self.features_aligned_itr:
                    if split in fao.split:
                        if 'Else' in fao.misalignment or 'ClassLabel' in fao.misalignment:
                            if 'ClassLabel' in fao.misalignment:
                                mapped_dataset = dataset_list[fao.feature_idx_per_split].map(lambda x: {fao.feature_name_new: self.feature_list_per_split[split][fao.feature_idx_per_split][fao.feature_name_orig].names[x[fao.feature_name_orig]]}, remove_columns=[fao.feature_name_orig])
                            elif 'Else' in iter_obj[0]:
                                mapped_dataset = dataset_list[fao.feature_idx_per_split].map(lambda x: {fao.feature_name_new: str(x[fao.feature_name_orig])}, remove_columns=[fao.feature_name_orig])
                            dataset_list[fao.feature_idx_per_split] = mapped_dataset
                return dataset_list
            if isinstance(dataset_list[0], DatasetDict):
                all_splits = list(set().union(*dataset_list))
                # For now, just do sampling on all splits.
                combined_dataset = DatasetDict({
                    split: interleave_datasets(
                        datasets=_force_align_features([datasetdict[split] for datasetdict in dataset_list if split in datasetdict.keys()], split), 
                        probabilities=_normalize_ratios([self.NAME_RATIOS[builder.info.config_name] for builder in self.__BUILDER_MIXTURE__.builder_list_per_split[split]]), 
                        seed=self.seed, 
                        stopping_strategy='all_exhausted'
                    ) for split in all_splits
                })
            else:
                if 'split' in kwargs:
                    split = kwargs['split']
                else:
                    split = ['']
                combined_dataset = interleave_datasets(
                    datasets=_force_align_features(dataset_list, split), 
                    probabilities=_normalize_ratios([self.NAME_RATIOS[builder.info.config_name] for builder in self.__BUILDER_MIXTURE__.builder_list_per_split[split]]), 
                    seed=self.seed, 
                    stopping_strategy='all_exhausted'
                )
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
