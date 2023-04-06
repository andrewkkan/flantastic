# flantastic
Mixture of datasets intended for multi-task training, by combining pre-packaged or custom packaged datasets with prompt templates into a single dataset.
Powered by [Huggingface Datasets](https://github.com/huggingface/datasets/) and [PromptSource](https://github.com/bigscience-workshop/promptsource).

## Objective
Inspired by [Google's SeqIO](https://github.com/google/seqio), [Google's FLAN usage of SeqIO](https://github.com/google-research/FLAN/tree/main/flan/v2), as well as BigScience Workshop [P3](https://github.com/bigscience-workshop/promptsource) (Public Pool of Prompts).
Target usage for [HF Transformers training pipeline](https://github.com/huggingface/transformers/tree/main/src/transformers).

### Usage
Below is an example of how to use the `flantastic` decorator to create a mixture of datasets.
```
import datasets
from datasets import load_dataset_builder
from flantastic import flantastic, Flantastic_Mixture

fm = Flantastic_Mixture(
    mixture_list=[
        {
            'builder': load_dataset_builder("banking77"),
            'ratio': 1.0,
            'templates': [
                'banking77/help_page_topic',
                'banking77/direct_to_which_department',
            ],
        },
        {
            'builder': load_dataset_builder("ag_news"),
            'ratio': 1.0,
            'templates': [
                'ag_news/classify',
                'ag_news/which_section',
            ],
        },
        {
            'builder': load_dataset_builder("squad"),
            'ratio': 0.5,
            'templates': [
                'squad/given_context_generate_question',
                'squad/jeopardy',
            ],
        },
    ],
)

@flantastic(mixture=fm)
class TestBuilder(datasets.GeneratorBasedBuilder):
    pass

tb = TestBuilder()
tb.download_and_prepare()
ds = tb.as_dataset()
```

This results in a dataset with the following structure:
```
DatasetDict({
    train: Dataset({
        features: ['INPUT', 'OUTPUT'],
        num_rows: 875331
    })
    test: Dataset({
        features: ['INPUT', 'OUTPUT'],
        num_rows: 30235
    })
    validation: Dataset({
        features: ['INPUT', 'OUTPUT'],
        num_rows: 21140
    })
})
```

### Install depedencies
```bash
git clone https://github.com/andrewkkan/flantastic.git
pip install -r requirements.txt
```
Python version: '<3.10,>=3.7' to satisfy promptsource dependency.




