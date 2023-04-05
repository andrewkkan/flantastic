# flantastic
Mixture of datasets intended for multi-task training, by combining pre-packaged or custom packaged datasets with prompt templates into a single dataset.

## Objective

Inspired by [Google's SeqIO](https://github.com/google/seqio), and [Google's FLAN usage of SeqIO](https://github.com/google-research/FLAN/tree/main/flan/v2).
Idea adapted for [HF Transformers training pipeline](https://github.com/huggingface/transformers/tree/main/src/transformers).

### Usage

```
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

### Install depedencies
```bash
git clone https://github.com/andrewkkan/flantastic.git
pip install -r requirements.txt
```
### Run training
```python TBD
```



