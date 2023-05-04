from symspellpy import SymSpell, Verbosity
from symspellpy.editdistance import LevenshteinFast


class SpellChecker:
    """Spell checker class used for feature extraction.
    """
    def __init__(self, max_edit_distance: int=2, dictionary_path: str='language_checkers/frequency_dictionary_en_82_765.txt') -> None:
        """Initialize the spell checker.
        Args:
            max_edit_distance (int, optional): The maximum edit distance used for matching corrections. Defaults to 2.
            dictionary_path (str, optional): The path to the dictionary file. Defaults to 'language_checkers/frequency_dictionary_en_82_765.txt'.
        """
        self.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.max_edit_distance = max_edit_distance
        self.distance_algorithm = LevenshteinFast()

    def get_closest_correction(self, word: str) -> str:
        """Get the closest correction for a word.
        Args:
            word (str): The word to correct.
        Returns:
            str: The closest matched correction or the word itself if the word is correct or no correction is found.
        """
        suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=self.max_edit_distance)

        try:
            return suggestions[0].term

        except IndexError: # No suggestions
            return word

    def get_levenshtein_distance(self, word1: str, word2: str) -> int:
        """Get the levenshtein distance between two words.
        Args:
            word1 (str): The first word.
            word2 (str): The second word.
        Returns:
            int: The levenshtein distance between the two words.
        """
        return self.distance_algorithm.distance(word1, word2, self.max_edit_distance) 
        
        