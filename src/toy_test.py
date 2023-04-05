import datasets
from datasets import load_dataset_builder
from flantastic import flantastic, Flantastic_Mixture

def test():

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
    ds, ds_train = tb.as_dataset(), tb.as_dataset(split='train')
    print(ds)
    print(ds_train[0:10])

if __name__ == '__main__':
    test()


