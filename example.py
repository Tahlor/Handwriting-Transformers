from hwgen.generators.Generator import *
from hwgen.data.hw_generator import *

if __name__=="__main__":
    uni = Unigrams(csv_file="./data/datasets/unigram_freq.csv")
    trivial = TrivialDataset("This is some data right here")

    # Load novel text next_text_dataset
    text_data = trivial

    basic_text_dataset = Wikipedia(
                dataset=load_dataset("wikipedia", "20220301.en")["train"],
                vocabulary=set(VOCABULARY),  # set(self.model.netconverter.dict.keys())
                encode_function=Wikipedia.encode,
                min_sentence_length=60,
                max_sentence_length=64
            )

    g = HWGenerator(model="IAM",
                    next_text_dataset=basic_text_dataset,
    )
    # Get random style
    #style = next(iter(g.style_image_and_text_dataset))
    words = basic_text_dataset.sample() # just return text

    # Get specific style
    author_id = g.style_image_and_text_dataset.random_author()
    style = g.style_image_and_text_dataset.get_one_author(n=batch_size, author_id=author_id)
    i = 0
    master_list = None
    while True:
        master_list = g.generate_new_samples(style, save_path=f"./data/datasets/synth_hw_wiki/style_{author_id}_samples_{i}.npy", master_list=master_list)
        i+=1
        print(i)

