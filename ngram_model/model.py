"""Core count-based n-gram language model implementation."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable, List, Sequence, Tuple


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<UNK>"


@dataclass(frozen=True)
class Prediction:
    """A possible next word and its conditional probability."""

    word: str
    probability: float


class NGramLanguageModel:
    """Predict a word from the previous ``context_size`` words.

    A context size of 2, for example, learns trigram probabilities:
    P(word_3 | word_1, word_2).
    """

    def __init__(self, context_size: int = 2, min_count: int = 1) -> None:
        if context_size < 1:
            raise ValueError("context_size must be at least 1")
        if min_count < 1:
            raise ValueError("min_count must be at least 1")

        self.context_size = context_size
        self.min_count = min_count
        self.ngram_counts: DefaultDict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.context_counts: Counter[Tuple[str, ...]] = Counter()
        self.vocabulary: set[str] = set()
        self._prediction_vocabulary: set[str] = set()
        self.training_token_count = 0
        self._trained = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Convert text to lowercase word tokens."""
        return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]

    def fit(self, text: str) -> "NGramLanguageModel":
        """Train the model from text, replacing rare words with ``<UNK>``."""
        raw_tokens = self.tokenize(text)
        if not raw_tokens:
            raise ValueError("The training text does not contain any words.")

        frequencies = Counter(raw_tokens)
        self.vocabulary = {
            word for word, count in frequencies.items() if count >= self.min_count
        }
        if not self.vocabulary:
            raise ValueError(
                "No words meet min_count; lower min_count or use a larger corpus."
            )
        tokens = [word if word in self.vocabulary else UNK_TOKEN for word in raw_tokens]

        self.ngram_counts.clear()
        self.context_counts.clear()
        self._prediction_vocabulary = self.vocabulary | {UNK_TOKEN, EOS_TOKEN}
        self.training_token_count = len(tokens)

        padded_tokens = [BOS_TOKEN] * self.context_size + tokens + [EOS_TOKEN]
        for index in range(self.context_size, len(padded_tokens)):
            context = tuple(padded_tokens[index - self.context_size : index])
            next_token = padded_tokens[index]
            self.ngram_counts[context][next_token] += 1
            self.context_counts[context] += 1

        self._trained = True
        return self

    def fit_file(self, path: str | Path, encoding: str = "utf-8") -> "NGramLanguageModel":
        """Read a text file and train the model on its contents."""
        text = Path(path).read_text(encoding=encoding)
        return self.fit(text)

    def probability(self, next_word: str, context: Sequence[str] | str) -> float:
        """Return the add-one-smoothed probability of a next word."""
        self._require_trained()
        normalized_context = self._normalize_context(context)
        normalized_word = self._normalize_word(next_word)
        vocabulary_size = len(self._prediction_vocabulary)

        next_count = self.ngram_counts[normalized_context][normalized_word]
        context_count = self.context_counts[normalized_context]
        return (next_count + 1) / (context_count + vocabulary_size)

    def predict_next(
        self, context: Sequence[str] | str, top_k: int = 5
    ) -> List[Prediction]:
        """Return the most probable lexical next words for a context."""
        self._require_trained()
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.vocabulary:
            raise ValueError(
                "No lexical words remain in the vocabulary; lower min_count and retrain."
            )

        predictions = [
            Prediction(word, self.probability(word, context))
            for word in self.vocabulary
        ]
        predictions.sort(key=lambda item: (-item.probability, item.word))
        return predictions[:top_k]

    def perplexity(self, sentence: str | Sequence[str]) -> float:
        """Calculate sentence perplexity, including its end-of-sentence event."""
        self._require_trained()
        if isinstance(sentence, str):
            raw_tokens = self.tokenize(sentence)
        else:
            raw_tokens = [str(token).lower() for token in sentence]
        if not raw_tokens:
            raise ValueError("The sentence must contain at least one word.")

        tokens = [self._normalize_word(word) for word in raw_tokens]
        padded_tokens = [BOS_TOKEN] * self.context_size + tokens + [EOS_TOKEN]
        log_probability_sum = 0.0
        event_count = 0

        for index in range(self.context_size, len(padded_tokens)):
            context = tuple(padded_tokens[index - self.context_size : index])
            next_word = padded_tokens[index]
            probability = self.probability(next_word, context)
            log_probability_sum += math.log(probability)
            event_count += 1

        return math.exp(-log_probability_sum / event_count)

    def _normalize_context(self, context: Sequence[str] | str) -> Tuple[str, ...]:
        if isinstance(context, str):
            words = self.tokenize(context)
        else:
            words = [str(word).lower() for word in context]

        normalized = [self._normalize_word(word) for word in words]
        if len(normalized) < self.context_size:
            normalized = [BOS_TOKEN] * (self.context_size - len(normalized)) + normalized
        return tuple(normalized[-self.context_size :])

    def _normalize_word(self, word: str) -> str:
        if word in {BOS_TOKEN, EOS_TOKEN, UNK_TOKEN}:
            return word
        lowered = word.lower()
        return lowered if lowered in self.vocabulary else UNK_TOKEN

    def _require_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("Train the model before requesting probabilities.")
