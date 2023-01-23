from queue import Queue
import shlex
import argparse
import os
from textgen.unigram_dataset import Unigrams
from textgen.wikipedia_dataset import Wikipedia
from hwgen.data.hw_generator import HWGenerator
import threading
from pathlib import Path

ROOT = Path(__file__).parent.absolute()


def create_parser():
    global OUTPUT_DICT, OUTPUT_OCR_JSON
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_handwriting_data",
                        action="store", const="sample", nargs="?",
                        help="Path to saved handwriting, 'sample' or 'eng_latest' to pull from S3")
    parser.add_argument("--saved_handwriting_model",
                        action="store", const="IAM", nargs="?",
                        help="Path to HWR model, OR 'CVL' or 'IAM'",
                        )
    parser.add_argument("--unigrams", action="store_const", const=True,
                        help="Path to unigram frequency file, if 'true' it will be downloaded from S3")
    parser.add_argument("--wikipedia", action="store", const="20220301.en", nargs="?",
                        help="20220301.en, 20220301.fr, etc.")
    parser.add_argument("--batch_size", default=12, type=int, help="Batch size for processing")
    parser.add_argument("--resume", action="store_const", const=-1, help="Resuming from previous process")
    parser.add_argument("--freq", default=5000, type=int, help="Frequency of processing")
    parser.add_argument("--output_folder", default=ROOT / "output", help="Path to output directory")
    parser.add_argument("--output_json", default=None, help="Path to output directory")
    parser.add_argument("--incrementer", default=True, help="Increment output folder")
    parser.add_argument("--debug", action="store_true", help="Debugging mode")

    return parser

def generate_images(model, image_queue):
    while True:
        image = model.generate()  # generate a new image
        image_queue.put(image)
def main(args=None):
    global IDX
    from hwgen.data.basic_text_dataset import VOCABULARY
    parser = create_parser()
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(shlex.split(args))

    basic_text_dataset = Unigrams(
        csv_file=args.unigrams,
    )

    renderer = HWGenerator(next_text_dataset=basic_text_dataset,
                           batch_size=args.batch_size,
                           model="IAM")

    master_list = g.generate_new_samples(style,
                                         save_path=f"./data/datasets/synth_hw_wiki/style_{author_id}_samples_{i}.npy",
                                         master_list=master_list)

    # Create a queue to hold the generated images
    image_queue = Queue(block=False, maxsize=32000)


    # Start the model in a separate thread
    threading.Thread(target=generate_images, kwargs={"model":renderer, "image_queue":image_queue}).start()

    # In another process
    while True:
        image = image_queue.get()
        # Do something with the image

if __name__=='__main__':
    output = ROOT / "output"
    command = f"""
    --output_folder {output}
    --batch_size 16 
    --freq 5000 
    --unigrams
    --saved_handwriting_data sample"""

    main()

