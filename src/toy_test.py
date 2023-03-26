import datasets
from datasets import load_dataset_builder
import flantastic
from IPython import embed

def test():

    axb_builder = load_dataset_builder("super_glue", "axb")
    boolq_builder = load_dataset_builder('super_glue', 'boolq')
    fm = flantastic.flantastic_mixture(mixture_list=[{'builder':axb_builder,'ratio':1.0}, {'builder':boolq_builder,'ratio':1.0}])

    @flantastic.flantastic(mixture=fm)
    class TestBuilder(datasets.GeneratorBasedBuilder):
        pass

    tb = TestBuilder()
    tb.download_and_prepare()
    ds, ds_train, ds_test = tb.as_dataset(), tb.as_dataset(split='train'), tb.as_dataset(split='test')
    # embed()

if __name__ == '__main__':
    test()


