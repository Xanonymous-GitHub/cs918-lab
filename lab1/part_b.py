# 1. Compute N (number of tokens) and V (vocabulary size).

# 2. List the top 25 trigrams based on the number of occurrences on the entire corpus.

# 3. Using the lists of positive and negative words provided with the corpus,
# compute the number of positive and negative word counts in the corpus.

# 4. Compute the number of news stories with more positive than negative words, as well as
# the number of news stories with more negative than positive words. News stories with a tie
# (same number of positive and negative words) should not be counted.


from collections import Counter
from pprint import pprint
from re import findall

from nltk import trigrams
from nltk.corpus import PlaintextCorpusReader

from utils.files import runtime_dir


def tokenize_of(string: str) -> tuple[int, [str]]:
    return len(tokens := findall(r'\w+', string)), tokens


def is_positive(string: str) -> bool:
    return string in ('positive', 'positive\n')


def start_nltk():
    corpus = PlaintextCorpusReader(runtime_dir, r'\w+\.txt')

    # Token size
    words = corpus.words()
    print(f'Token Size: {len(words)}')

    # Vocabulary size
    total_vocab = Counter(words)
    print(f'Vocabulary Size: {total_vocab.__len__()}')

    # Top 25 trigrams
    trigram = trigrams(words)
    print('Top 25 Trigrams: ', end='')
    pprint(Counter(trigram).most_common(25))

    positive_words = frozenset(corpus.words(f'{runtime_dir}/positive-words.txt'))
    negative_words = frozenset(corpus.words(f'{runtime_dir}/negative-words.txt'))

    print(len(positive_words))
    print(len(negative_words))


if __name__ == '__main__':
    try:
        start_nltk()
    except KeyboardInterrupt:
        pass
