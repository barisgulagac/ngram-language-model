
A dependency-free Python n-gram language model that trains on a user-provided `.txt` file. It predicts word `n+1` from the preceding `n` words, reports each prediction's Laplace-smoothed probability, and calculates sentence perplexity.
The vocabulary includes the unknown-word and end-of-sentence tokens. Suggestions exclude internal control tokens, so the CLI displays actual words only.
"no stupid backoff implementation yet"

Sentence perplexity includes the probability of the end-of-sentence event:
Lower perplexity means that the model considers the sentence more likely.


## Running

```bash
python3 main.py path/to/corpus.txt --context-size 2
```

`--context-size 2` uses the previous two words to predict the third word. In standard terminology, this is a trigram model. At the prompt, enter at least two words; if you enter more, the final two are used to predict the next word.

## Example

```text
Training complete: 12000 tokens, 2100 known words, context size 2.

Enter at least 2 words. The last 2 words will be used to predict the next word.

> the weather
Next-word predictions:
  1. is                   0.125000
  2. was                  0.062500
Sentence perplexity: 42.381920
```
## Quit
Use `:quit` to leave the program.

## Corpus and Training Data 
you can train the model on any plain text file ('.txt') file.
## Option 1:
Create or open a `.txt` file (e.g., `sample.txt`) in the project directory and paste your desired training text.
## Option 2:
Use `curl` in your terminal to fetch public-domain books directly from [Project Gutenberg](https://www.gutenberg.org/):
```bash
# Download "The Adventures of Sherlock Holmes"
curl -o sample.txt [https://www.gutenberg.org/files/1661/1661-0.txt](https://www.gutenberg.org/files/1661/1661-0.txt)

# Run the model on the downloaded corpus
python3 main.py sample.txt --context-size 2
```
## Options 

-n, --context-size N   Number of context words (default: 2)
--min-count COUNT      Replace less frequent words with <UNK> (default: 1)
--top-k K              Number of predictions shown (default: 5)
--encoding NAME        Corpus character encoding (default: utf-8)

For meaningful unknown-word learning, use `--min-count 2` or higher on a reasonably large corpus. Rare training words are then represented by `<UNK>`. Even with the default, unseen input words map to `<UNK>` and receive nonzero probability from smoothing.



