import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import List, Tuple

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class POSTagger:
    """The is a class that generates POS ngrams. It is used for feature extractions.
    """
    def __init__(self, text: str):
        """Initializes the POSTagger class.

        Args:
            text (str): The text to extract the POS tags from.
        """
        self.text = text
        self.sentences = sent_tokenize(text)
        self.tokens = [word_tokenize(sentence) for sentence in self.sentences]
        self.pos_tags = [nltk.pos_tag(token) for token in self.tokens]

    def get_pos_ngrams(self, n) -> List[Tuple]:
        pos_sentences = [list(ngrams(sentence, n)) for sentence in self.pos_tags]	

        # get only the tags not the words
        pos_sentences = [[tuple([tag for word, tag in ngram]) for ngram in sentence] for sentence in pos_sentences]

        output = []

        [output.extend(sentence) for sentence in pos_sentences]

        return output