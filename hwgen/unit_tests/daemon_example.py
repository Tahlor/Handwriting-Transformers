import time
from queue import Queue
import queue
import shlex
import argparse
import os
from textgen.unigram_dataset import Unigrams
from textgen.wikipedia_dataset import WikipediaEncodedTextDataset
from hwgen.data.hw_generator import HWGenerator
from datasets import load_dataset
import threading
from pathlib import Path
from time import sleep
import torch
from hwgen.daemon import Daemon
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).parent.absolute()


def create_parser():
    global OUTPUT_DICT, OUTPUT_OCR_JSON
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_handwriting_data",
                        action="store", const="sample", nargs="?",
                        help="Path to saved handwriting, 'sample' or 'eng_latest' to pull from S3")
    parser.add_argument("--saved_handwriting_model", default="IAM",
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
    parser.add_argument("--min_chars", default=8, help="Min chars in batch-item")
    parser.add_argument("--max_chars", default=200, help="Max chars in batch-item")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cpu, cuda, cuda:0, etc.")


    return parser


def main(args=None):
    global IDX
    parser = create_parser()
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(shlex.split(args))

    VOCABULARY = """Only thewigsofrcvdampbkuq.A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/"""
    basic_text_dataset = WikipediaEncodedTextDataset(
        dataset=load_dataset("wikipedia", args.wikipedia)["train"],
        vocabulary=set(VOCABULARY),  # set(self.model.netconverter.dict.keys())
        exclude_chars="0123456789()+*;#:!/,.",
        use_unidecode=True,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    renderer = HWGenerator(next_text_dataset=basic_text_dataset,
                           batch_size=args.batch_size,
                           model=args.saved_handwriting_model,
                           device=args.device)

    daemon = Daemon(renderer)
    daemon.start()

    # Consume items from the queue
    sleep(10)
    for i in range(1000):
        try:
            item = daemon.queue.get(timeout=1)
            print(f'{i} {time.strftime("%H:%M:%S", time.localtime())}')
            # print current time
            # Do something with the item
        except queue.Empty:
            # If the queue is empty for too long, stop consuming.
            print("Queue is empty, waiting 1 second")
            sleep(1)

    daemon.stop()
    daemon.join()


if __name__ == '__main__':
    main()

