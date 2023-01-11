import re
import unicodedata

def filter_short_lines(text, minimum_length, end_punctuation):
    """ Exclude lines with fewer than `minimum_length` words

    Returns:

    """
    out = []
    for line in text.split('\n'):
        words = len(line.split(' '))
        if words > minimum_length:
            out += [line]
    return "\n".join(out)


def filter_lines_to_sentences(text,end_punctuation="."):
    """ Exclude lines with fewer than `minimum_length` words

    Returns:

    """
    return "\n".join([line for line in text.split('\n') if line and line[-1]=="."])

def filter_with_punctuation():
    lower_case = re.compile(r'[^a-z0-9 \.]+')
    double_space = re.compile(r'\s\s+')
    space_period = re.compile(r'\s+\.')
    return lambda sentence: space_period.sub(".",
                                            double_space.sub(" ",
                                            lower_case.sub("", sentence)).strip())

def filter_vocab(vocab=None):
    if vocab is None:
        return lambda x: x
    else:
        #vocab = set(vocab)
        #return lambda string: "".join([v for v in string if v in vocab])
        vocab = re.escape(str(vocab))
        vocab_regex = re.compile(fr"""[^{vocab}]*""")
        return lambda x: vocab_regex.sub("", x)


"""
Excluded:
$<=>@[\]^_`{|}~
Included:
'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
"""
symbol_replacements = {"}":")",
"{":"(",
"]":")",
"[":"(",
"\n":" ",
"(":"(",
")":")",
#" ()":""
}


re_funcs = []
for key in symbol_replacements.keys():
    k = f"\{key}" if key in "()[]" else key
    whitespace = "[\s]*" if key in "([{" else ""
    func = re.compile(f"{k}{whitespace}")
    print(k,key, func)
    re_funcs.append([func, symbol_replacements[key]])

def replace_symbols(text):
    text = unicodedata.normalize('NFKD', text)
    # text.replace("[","(").replace("]", ")").replace("\n", " ").replace(
    #     "{","(").replace("}",")")
    for f,r in re_funcs:
        text = f.sub(r,text)
    text = text.replace(" ()", "")

    return text

def test(text):
    x = "\n".join([line for line in text.split('\n') if line.rstrip() and line.rstrip()[-1]=="."])
    y = "\n".join([line for line in text.split('\n') if line and line[-1]=="."])
    assert x==y



if __name__ == '__main__':
    x = str.translate("this ($)%*&)#", symbol_replacements)
    print(x)