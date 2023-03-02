from hwgen.generators.Generator import *
from textgen.basic_text_dataset import VOCABULARY, ALPHA_VOCABULARY
from hwgen.data.hw_generator import HWGenerator
import os
from hwgen.data.utils import show

os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = get_model(models[DEFAULT_MODEL] if DEFAULT_MODEL in models else DEFAULT_MODEL)

def create_hw(batch_size = 1):
    words_dataset = WikipediaEncodedTextDataset(
        use_unidecode=True,
        shuffle_articles=True,
        random_starting_word=True,
        dataset=load_dataset("wikipedia", "20220301.en")["train"],
        vocabulary=set(VOCABULARY),  # set(self.model.netconverter.dict.keys())
        exclude_chars="",
        symbol_replacement_dict={
            "}": ")",
            "{": "(",
            "]": ")",
            "[": "(",
            "–": "-",
            " ()": "",
            "\n": " "
        }
    )
    hw_gen = HWGenerator(next_text_dataset=words_dataset,
                batch_size=batch_size,
                model="IAM",
                device=None,
                style="IAM",
                )

    output = next(iter(hw_gen))
    show(output['word_imgs'][0])

if __name__ == "__main__":
    create_hw()