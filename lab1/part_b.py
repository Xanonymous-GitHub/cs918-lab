# 1. Compute N (number of tokens) and V (vocabulary size).

# 2. List the top 25 trigrams based on the number of occurrences on the entire corpus.

# 3. Using the lists of positive and negative words provided with the corpus,
# compute the number of positive and negative word counts in the corpus.

# 4. Compute the number of news stories with more positive than negative words, as well as
# the number of news stories with more negative than positive words. News stories with a tie
# (same number of positive and negative words) should not be counted.


import asyncio
import multiprocessing
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from pprint import pprint
from re import findall
from threading import Lock
from typing import Any, Generator

from nltk import trigrams, FreqDist
from nltk.downloader import download as nltk_download
from nltk.tokenize import WhitespaceTokenizer, RegexpTokenizer

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


def calculate_number_of_token_in_set(token: Generator, token_set: frozenset[str]) -> int:
    return sum(1 for t in token if t in token_set)


async def count_each_news_pos_neg_words(
        text: str,
        positive_words: frozenset[str],
        negative_words: frozenset[str]
) -> tuple[int, int]:
    new_line_tokenizer = RegexpTokenizer(r'\n', gaps=True)
    span = new_line_tokenizer.span_tokenize(text)
    lines = (text[begin: end] for (begin, end) in span)

    def find_pos_and_neg_number_in_line(line: str) -> tuple[int, int]:
        tokens = get_tokens_from_text(line)
        pos_words_count = calculate_number_of_token_in_set(tokens, positive_words)
        neg_words_count = calculate_number_of_token_in_set(tokens, negative_words)
        return pos_words_count, neg_words_count

    pos_more_than_neg_counter = 0
    neg_more_than_pos_counter = 0

    # spawn maximum threads to calculate the number of a line that has more positive or negative words.
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        loop = asyncio.get_event_loop()
        # Find `pos, neg` in each line, once a line is done, check if it has more positive or negative words.
        # Line and lines are not related, so we can do this in parallel.
        # Use a deque to store the results of the tasks, and then process them in the main thread.
        tasks = (loop.run_in_executor(executor, find_pos_and_neg_number_in_line, line) for line in lines)

        thread_lock = Lock()
        for pos, neg in await asyncio.gather(*tasks):
            if pos > neg:
                with thread_lock:
                    pos_more_than_neg_counter += 1
            elif neg > pos:
                with thread_lock:
                    neg_more_than_pos_counter += 1

    return pos_more_than_neg_counter, neg_more_than_pos_counter


async def start_nltk():
    file_path = f'{runtime_dir}/processed_text_part_a.txt'
    text = get_text_from_file(file_path)

    tokens = get_tokens_by_regexp_from_text(text)

    thread_lock = Lock()

    def show_token_analysis(tokens_: Generator):
        token_analysis = analyze_tokens_from(tokens_)

        with thread_lock:
            # Token size
            print(f'Token Size: {token_analysis.N()}')

        with thread_lock:
            # Vocabulary size
            print(f'Vocabulary Size: {token_analysis.B()}')

    def show_trigrams(tokens_: Generator):
        trigram = trigrams(tokens_)
        most_common_trigrams = Counter(trigram).most_common(25)

        with thread_lock:
            print('Top 25 Trigrams: ')
            pprint(most_common_trigrams)

    def count_positive_words(tokens_: Generator):
        positive_words = frozenset(
            get_tokens_by_regexp_from_text(
                get_text_from_file(f'{runtime_dir}/positive-words.txt')
            )
        )

        positive_words_count = calculate_number_of_token_in_set(tokens_, positive_words)

        with thread_lock:
            print(f'There are {positive_words_count} positive words in the corpus.')

        return positive_words

    def count_negative_words(tokens_: Generator):
        negative_words = frozenset(
            get_tokens_by_regexp_from_text(
                get_text_from_file(f'{runtime_dir}/negative-words.txt')
            )
        )

        negative_words_count = calculate_number_of_token_in_set(tokens_, negative_words)

        with thread_lock:
            print(f'There are {negative_words_count} negative words in the corpus.')

        return negative_words

    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()

        # Note: We need to copy the tokens because the generator used in other functions
        # Ensure this step is done before the other functions are called.

        _, __, pos_words, neg_words = await asyncio.gather(
            loop.run_in_executor(executor, show_token_analysis, tokens),
            loop.run_in_executor(executor, show_trigrams, copy(tokens)),
            loop.run_in_executor(executor, count_positive_words, copy(tokens)),
            loop.run_in_executor(executor, count_negative_words, copy(tokens)),
        )

        pos_more_than_neg, neg_more_than_pos = await count_each_news_pos_neg_words(text, pos_words, neg_words)
        print(f'News stories with more positive than negative words: {pos_more_than_neg}')
        print(f'News stories with more negative than positive words: {neg_more_than_pos}')


if __name__ == '__main__':
    try:
        nltk_download('punkt')
        asyncio.run(start_nltk())
    except KeyboardInterrupt:
        pass
