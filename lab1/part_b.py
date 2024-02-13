# 1. Compute N (number of tokens) and V (vocabulary size).

# 2. List the top 25 trigrams based on the number of occurrences on the entire corpus.

# 3. Using the lists of positive and negative words provided with the corpus,
# compute the number of positive and negative word counts in the corpus.

# 4. Compute the number of news stories with more positive than negative words, as well as
# the number of news stories with more negative than positive words. News stories with a tie
# (same number of positive and negative words) should not be counted.


import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from pprint import pprint
from re import findall
from typing import Any, Generator

from nltk import trigrams, FreqDist
from nltk.downloader import download as nltk_download
from nltk.tokenize import WhitespaceTokenizer

from utils.files import runtime_dir


def get_text_from_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read().lower()


def get_tokens_by_regexp_from_text(text: str) -> [str]:
    return findall(r'\w+', text)


def get_tokens_from_text(text: str) -> Generator[str, Any, None]:
    # Performance: Use low-level function instead of word_tokenize.
    # return Generator instead of List to save memory and save time.
    spans = WhitespaceTokenizer().span_tokenize(text)
    return (text[begin: end] for (begin, end) in spans)


def analyze_tokens_from(tokens: Generator) -> FreqDist[str]:
    fd = FreqDist()
    for token in tokens:
        fd[token] += 1
    return fd


async def start_nltk():
    file_path = f'{runtime_dir}/processed_text_part_a.txt'
    text = get_text_from_file(file_path)

    tokens = get_tokens_by_regexp_from_text(text)

    def show_token_analysis(tokens_: Generator):
        token_analysis = analyze_tokens_from(tokens_)

        # Token size
        print(f'Token Size: {token_analysis.N()}')

        # Vocabulary size
        print(f'Vocabulary Size: {token_analysis.B()}')

    def show_trigrams(tokens_: Generator):
        trigram = trigrams(tokens_)
        most_common_trigrams = Counter(trigram).most_common(25)
        print('Top 25 Trigrams: ')
        pprint(most_common_trigrams)

    def count_positive_words(tokens_: Generator):
        positive_words = frozenset(
            get_tokens_by_regexp_from_text(
                get_text_from_file(f'{runtime_dir}/positive-words.txt')
            )
        )

        positive_words_count = sum(1 for token in tokens_ if token in positive_words)
        print(f'There are {positive_words_count} positive words in the corpus.')

    def count_negative_words(tokens_: Generator):
        negative_words = frozenset(
            get_tokens_by_regexp_from_text(
                get_text_from_file(f'{runtime_dir}/negative-words.txt')
            )
        )

        negative_words_count = sum(1 for token in tokens_ if token in negative_words)
        print(f'There are {negative_words_count} negative words in the corpus.')

    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()

        # Note: We need to copy the tokens because the generator used in other functions
        # Ensure this step is done before the other functions are called.

        await asyncio.gather(
            loop.run_in_executor(executor, show_token_analysis, tokens),
            loop.run_in_executor(executor, show_trigrams, copy(tokens)),
            loop.run_in_executor(executor, count_positive_words, copy(tokens)),
            loop.run_in_executor(executor, count_negative_words, copy(tokens)),
        )


if __name__ == '__main__':
    try:
        nltk_download('punkt')
        asyncio.run(start_nltk())
    except KeyboardInterrupt:
        pass
