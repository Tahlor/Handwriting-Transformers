from data.basic_text_dataset import BasicTextDataset
import random
from text_utils import *

class Wikipedia(BasicTextDataset):
    def __init__(self,
                 dataset,
                 vocabulary,
                 encode_function,
                 min_sentence_length=32,
                 max_sentence_length=64,
                 epoch_size=100000,
                 ):
        super().__init__(dataset,vocabulary,encode_function)
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.length = epoch_size

    def __len__(self):
        return self.length

    @staticmethod
    def chunk_letters(filtered_sentence, length):
        """ Search rightward for next space at beginning/end of line

        Args:
            filtered_sentence:
            length:

        Returns:

        """
        try:
            start = random.randint(0, len(filtered_sentence) - length)
            new_start = filtered_sentence[start:].find(" ")+start+1
            fs = filtered_sentence[new_start:new_start+length]
            new_end = fs.rfind(" ")
            fs = fs[:new_end]
        except Exception as e:
            fs = filtered_sentence[:length]
            new_end = fs.rfind(" ") if fs.rfind(" ") > 0 else None
            fs = fs[:new_end].strip()
        return fs

    @staticmethod
    def chunk_words(filtered_sentence, length, force_new_sentence=1, force_end_sentence=1):
        """ Search rightward for next sentence and end of sentence.
        Args:
            filtered_sentence:
            length:
            force_new_sentence [0,1] probability: 1 - force text to begin at the beginning of a sentence
            force_end_sentence [0,1] probability: 1 - force text to end at end of sentence
        Returns:

        It's possible it returns a blank.
        """
        def find_next_period(sentence, index):
            # Incorrect attempt to search within selection for sentence end
            #end = -next(i for i,word in enumerate(filtered_sentence[end:start:-1]) if word[-1] == ".") + end

            try: # Search rightward for next period
                index += next(i for i, word in enumerate(sentence[index:]) if word and word[-1] == ".") + 1
            except StopIteration: # None found
                index = None
            return index

        force_new_sentence = random.random() < force_new_sentence # 1 means force
        force_end_sentence = random.random() < force_end_sentence # 1 means force

        _filtered_sentence = filtered_sentence.split(" ")
        start = random.randint(0, max(0,len(_filtered_sentence) - length))
        if force_new_sentence:
            start = find_next_period(filtered_sentence, start)
        end = start + length if start else length
        if force_end_sentence:
            end = find_next_period(_filtered_sentence, end)

        new_sentence = " ".join(_filtered_sentence[start:end]).strip()
        return new_sentence


    def _get_text(self, sentence, key="text", target_length=None, unit="words"):
        """ Target length is usually a minimum length if we need to force it to end at a sentence end.
            However, the returned sentence can be shorter than the target length, if the initial start point is
                at the end of a document and we force to start at sentence beginning.
                    [Instead of looking forward to start sentence, we could look backward, then target_length would be minimum length.]
                    [We could reject articles that are too short]
        Args:
            sentence:
            key:
            target_length:
            unit:

        Returns:

        """
        if target_length is None:
            target_length = self.max_sentence_length

        text = sentence[key]

        # Remove section headings
        filtered_sentence = replace_symbols(filter_lines_to_sentences(text))

        # Remove OOV symbols
        filtered_sentence = " " + self.filter_sentence(filtered_sentence) + " "
        if unit == "characters":
            fs = self.chunk_letters(filtered_sentence, target_length)
        elif unit == "words":
            fs = self.chunk_words(filtered_sentence, target_length)
        return fs

    def get_text(self, idx, unit="words"):
        """ Choose random length sentence satisfying (min,vocab_size).
            After processing, make sure it is still long enough.

        Args:
            idx:
            unit:

        Returns:

        """
        length = random.randint(self.min_sentence_length, self.max_sentence_length)
        while True:
            sentence = self._get_text(self.dataset[idx], target_length=length)
            s = sentence if unit == "characters" else sentence.split()
            s = s[:self.max_sentence_length]
            if len(s) >= self.min_sentence_length: # make sure article is long enough
                break
            else:
                idx = random.randint(0,len(self)) # try a new random article
        return s if unit == "characters" else " ".join(s)
