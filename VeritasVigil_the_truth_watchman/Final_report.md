# Final Report: VeritasVigil: The truth Watchman

## 1. Custom Tokenizer and Lemmatizer Design

- **Custom Tokenizer**:
  - Converts text to lowercase.
  - Expands contractions using a custom dictionary.
  - Replaces emoticons with a placeholder token <EMOTICON>.
  - Removes punctuation marks.
  - Normalizes elongated words (e.g., “goooood” becomes go <REPEAT:5>).
  - Tokenizes text by simple whitespace splitting.

- **Mini POS Tagger**:
  - Uses a rule-based system based on suffix patterns
   - Verb Identification Rules:
      - If a word is in the list of common auxiliary verbs:
        - is, are, was, were, be, being, been, am, have, has, had, do, does, did: tag as verb.
      - If a word ends with:
        - ing: tag as verb (present participle)
        - ed: tag as verb (past tense)
      
    - Adjective Identification Rules:
      - If a word ends with:
          - ly,ous, ful, able, ible, ic, ive: tag as adj
    - Default Rule:  
      - If none of the above rules apply: tag the word as noun.

- **Custom Lemmatizer**:
  - Reduces irregular verbs based on a manually defined dictionary.
  - Simplifies verbs by removing common suffixes (ing, ed, s).
  - Converts plural nouns to singular by removing trailing "s".
  - Simplifies adjectives by removing common suffixes (ly,ful)

## 2. Impact of Repeated-Character Normalization and POS-Guided Lemmatization

- **Repeated-Character Normalization**:
  - Reduces vocabulary size by preventing multiple exaggerated forms of the same word.
  - Improves model generalization by limiting overfitting to rare, noisy tokens.

- **POS-Guided Lemmatization**:
  - Reduces words to their base forms.
  - Simplifies irregular verb forms.
  - Boosts accuracy by unifying word representations without losing context.


## 3. Comparison with Off-the-Shelf Pipelines 

- While NLP libraries like NLTK and spaCy offer powerful tokenization, POS tagging, and lemmatization, they are designed for clean, well-formed text and can struggle with noisy, informal datasets like fake news or social media posts.

- Unlike off-the-shelf tokenizers that primarily rely on whitespace and punctuation, my custom tokenizer effectively handles repeated characters, emoticons and normalises exaggerated spellings into consistent tokens. This reduces noise and vocabulary size.

- For POS tagging, libraries like spaCy leverage statistical models trained on standard corpora, achieving high accuracy on formal text but inconsistent performance on informal writing. My rule-based mini POS tagger, while simpler, reliably identifies grammatical roles based on suffix patterns and predefined verb lists,thus proving sufficient for lemmatization in fake news classification tasks.

- In lemmatization, tools like NLTK’s WordNet and spaCy's lookup tables excel at standard English but often miss slang and informal variations. My custom lemmatizer, guided by POS tags, simplifies verbs and handles irregular forms explicitly, offering better control.

In summary, while off-the-shelf pipelines are convenient and efficient for typical text, custom pipeline outperforms them in handling informal, noisy, and exaggerated fake news content, resulting in a cleaner feature space, improved classification accuracy, and enhanced interpretability.
