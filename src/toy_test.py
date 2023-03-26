import datasets
from datasets import load_dataset_builder
from super_glue import flantastic
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
    ds = tb.as_dataset(split='train')
    embed()

