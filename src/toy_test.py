import datasets
from datasets import load_dataset_builder
from flantastic import flantastic, Flantastic_Mixture
from IPython import embed

def test():

    fm = Flantastic_Mixture(mixture_list=[
        {
            'builder': load_dataset_builder("banking77"),
            'ratio': 1.0,
            'templates': [
                {   
                    'template_name': 'banking77/help_page_topic',
                    'input_feature_name': 'INPUT',
                    'output_feature_name': 'OUTPUT',
                    'remove_applied_features': [True, True],
                }, 
                {
                    'template_name': 'banking77/direct_to_which_department',
                    'input_feature_name': 'INPUT',
                    'output_feature_name': 'OUTPUT',
                    'remove_applied_features': [True, True],
                },
            ],
        }, 
        {
            'builder': load_dataset_builder("ag_news"),
            'ratio': 1.0,
            'templates': [
                {   
                    'template_name': 'ag_news/classify',
                    'input_feature_name': 'INPUT',
                    'output_feature_name': 'OUTPUT',
                    'remove_applied_features': [False, True],
                }, 
                {
                    'template_name': 'ag_news/which_section',
                    'input_feature_name': 'INPUT',
                    'output_feature_name': 'OUTPUT',
                    'remove_applied_features': [True, False],
                },
            ],            
        },
    ])

    @flantastic(mixture=fm)
    class TestBuilder(datasets.GeneratorBasedBuilder):
        pass

    tb = TestBuilder()
    tb.download_and_prepare()
    ds, ds_train, ds_test = tb.as_dataset(), tb.as_dataset(split='train'), tb.as_dataset(split='test')
    embed()

if __name__ == '__main__':
    test()


