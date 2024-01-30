# 1. Compute N (number of tokens) and V (vocabulary size).

# 2. List the top 25 trigrams based on the number of occurrences on the entire corpus.

# 3. Using the lists of positive and negative words provided with the corpus,
# compute the number of positive and negative word counts in the corpus.

# 4. Compute the number of news stories with more positive than negative words, as well as
# the number of news stories with more negative than positive words. News stories with a tie
# (same number of positive and negative words) should not be counted.


from collections import Counter
from collections.abc import Sequence
from pprint import pprint
from re import findall

from utils.files import read_file_lines_from


def tokenize_of(string: str) -> tuple[int, [str]]:
    return len(tokens := findall(r'\w+', string)), tokens


def vocabulary_size_of(tokens: Sequence[str]) -> tuple[int, Counter[str]]:
    counter = Counter(tokens)
    return len(counter), counter


def start():
    raw_file_lines = read_file_lines_from('processed_text_part_a.txt')

    very_long_single_line = ' '.join(raw_file_lines)
    total_num_of_tokens, tokens = tokenize_of(very_long_single_line)
    total_vocab_size, total_vocab = vocabulary_size_of(tokens)

    print(f'N: {total_num_of_tokens}')
    print(f'V: {total_vocab_size}')
    print(f'Top 25 trigrams: ')
    pprint(total_vocab.most_common(25))


if __name__ == '__main__':
    try:
        start()
    except KeyboardInterrupt:
        pass
