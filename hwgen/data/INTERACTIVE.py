import os
from hwgen.data.utils import show, display
from textgen.basic_text_dataset import BasicTextDataset
from textgen.trivial_dataset import TrivialDataset
from hwgen.data.hw_generator import HWGenerator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
dataset = BasicTextDataset(TrivialDataset("T-his ~ == _ = is .. -- some -- data right --- here"))
render_text_pair_gen =HWGenerator(next_text_dataset=dataset,
                                        batch_size=1)
for x in render_text_pair_gen:
    show(x["word_imgs"][0])