import re
from collections.abc import Sequence
from re import Pattern

from nltk.downloader import download as nltk_download
from nltk.stem import WordNetLemmatizer
from pandas import read_json

from utils.files import runtime_dir, write_file_lines_to


def filter_text(src: str, /, *, patterns: Sequence[Pattern]) -> str:
    filtered = src

    for pattern in patterns:
        filtered = pattern.sub('', filtered)

    return filtered


def lemmatize_text(src: str, /) -> str:
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(src)


def start():
    lines = read_json(f'{runtime_dir}/signal-news1.jsonl', lines=True)['content']

    pattern_alphanumeric = re.compile(r'[^a-zA-Z0-9 ]')
    pattern_only_one_char = re.compile(r'\b[a-zA-Z]\b')
    pattern_fully_numeric = re.compile(r'\b\d+\b')
    pattern_url = re.compile(
        r'^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b[-a-zA-Z0-9()@:%_\\+.~#?&/=]*'
    )

    patterns = (
        pattern_alphanumeric,
        pattern_only_one_char,
        pattern_fully_numeric,
        pattern_url,
    )

    filtered_lines = [filter_text(line.lower(), patterns=patterns) for line in lines]
    lemmatized_lines = [' '.join([lemmatize_text(word) for word in line.split()]) for line in filtered_lines]
    write_file_lines_to('processed_text_part_a.txt', lines=lemmatized_lines)


if __name__ == '__main__':
    try:
        nltk_download('wordnet')
        start()
    except KeyboardInterrupt:
        pass
