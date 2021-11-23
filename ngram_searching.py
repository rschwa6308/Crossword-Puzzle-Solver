from typing import List, Tuple
import nltk
from nltk.corpus import words
import re


# nltk.download('words')
ENGLISH_WORDS = set(w.upper() for w in words.words())

ENGLISH_WORDS_STRING = "\n".join(ENGLISH_WORDS)

def word_search(slot: str, single=False) -> List[str]:
    pattern = "".join("." if c == " " else c.upper() for c in slot)
    # print(pattern)
    if single:
        if m := re.search(
            f"^({pattern})$",
            ENGLISH_WORDS_STRING,
            re.MULTILINE
        ):
            return [m.group(0)]
        else:
            return []
    else:
        return re.findall(
            f"^({pattern})$",
            ENGLISH_WORDS_STRING,
            re.MULTILINE
        )



def ngram_search(n: int, slot: str, single=False) -> List[Tuple[str, ...]]:
    if n == 1:
        if words := word_search(slot):
            return [(w,) for w in words]
        else:
            return []
    
    output = []
    for i in range(1, len(slot)):
        head, tail = slot[:i], slot[i:]
        # print(head, tail)
        # cartesian product of head and tail results
        head_words = word_search(head, single)
        tail_ngrams = ngram_search(n-1, tail)
        for head_word in head_words:
            for tail_ngram in tail_ngrams:
                # print(head_word, tail_ngram)
                output.append((head_word,) + tail_ngram)
                if single: return output
        
    return output




if __name__ == "__main__":
    from pprint import pprint
    from timeit import timeit
    # print(timeit(
    #     'print(ngram_search(2, "T     F"))',
    #     setup="from __main__ import ngram_search",
    #     number=10
    # ) / 10)
    pprint(ngram_search(2, "E TSWAY"))
    pprint(ngram_search(2, "E TSWAY", single=True))
    