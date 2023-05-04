import string
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from language_checkers.spell_checker import SpellChecker
from language_checkers.grammar_checker import GrammarChecker
from language_checkers.pos_tagger import POSTagger
import difflib
from collections import Counter 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class NormalFeatureExtractor:
    """A feature extractor that uses a combination of word ngrams, character ngrams, edit distance, substitutions, grammar mistakes,
        function word use, POS-ngrams and sentence length to create features.
    """

    def __init__(self,
                 word_ngram_range: Tuple[int,int]=(1, 1),
                 word_ngram_max_features: int=1000,
                 remove_stopwords: bool=True,
                 char_ngram_range: Tuple[int,int]=(3, 3),
                 char_ngram_max_features: int=1000,
                 max_edit_distance: int=2,
                 substitution_max_features: int=400,
                 function_words_path: str='feature_extractors/function_words.txt',
                 max_function_words_features: int=None,
                 max_pos_features: int=300,
                 pos_ngram_size: int=3,
                 log_file: str=None) -> None:
        """Initialize the feature extractor.

        Args:
            word_ngram_range (Tuple[int,int], optional): the range of word ngrams to create. The first element in the tuple is the minimum ngram size, the second is the maximum. Defaults to (1, 1).
            word_ngram_max_features (int, optional): the maximum number of word feature to create. This creates the top n features. Defaults to 1000.
            remove_stopwords (bool, optional): whether or not to remove stopwords when creating word ngrams. Defaults to True.
            char_ngram_range (Tuple[int,int], optional): the range of character ngrams to create. The first element in the tuple is the minimum ngram size, the second is the maximum. Defaults to (3, 3).
            char_ngram_max_features (int, optional): the maximum number of character feature to create. This creates the top n features. Defaults to 1000.
            max_edit_distance (int, optional): the maximum edit distance to use when doing spell checking. Defaults to 2.
            substitution_max_features (int, optional): the maximum number of substitution features to create. Defaults to 400.
            function_words_path (str, optional): the path to the file containing the function words. Defaults to 'feature_extractors/function_words.txt'.
            max_function_words_features (int, optional): the maximum number of function word features to create. If not set, all the function words will be used. Defaults to None.
            max_pos_features (int, optional): the maximum number of part of speech features to create. Defaults to 300.
            pos_ngram_size (int, optional): the size of the part of speech ngrams to create. Defaults to 3.
        """
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.word_ngram_max = word_ngram_max_features
        self.char_ngram_max = char_ngram_max_features
        if remove_stopwords:
            stop_words = 'english'
        else:
            stop_words = None
        self.word_vectorizer = CountVectorizer(ngram_range=self.word_ngram_range, max_features=self.word_ngram_max, stop_words=stop_words)
        self.char_vectorizer = CountVectorizer(analyzer='char', ngram_range=self.char_ngram_range, max_features=self.char_ngram_max)
        self.substitution_max_features = substitution_max_features
        self._spell_checker = SpellChecker(max_edit_distance=max_edit_distance)
        self.substitutions = None
        self.grammar_mistake_rules = None
        self.grammar_mistakes_per_chunk = None
        with open(function_words_path, 'r') as f:
            self.function_words = f.read().splitlines()
        self.max_function_words_features = max_function_words_features
        self.pos_features = None
        self.max_pos_features = max_pos_features
        self.pos_ngram_size = pos_ngram_size
        self.log_file = log_file

    def fit(self,
            X: List[str],
            fit_vectorizers: bool=True,
            fit_substitutions: bool=True,
            fit_grammar_mistakes: bool=True,
            fit_pos_ngrams: bool=True) -> None:
        """Fit the feature extractor to the training data. This first initializes the word and char CountVectorizers,
        then it discovers the most common substitutions, finds all the grammar mistakes

        Args:
            X (List[str]): A list of strings, each representing to the text of a chunk.
            fit_vectorizers (bool, optional): whether or not to fit the word and char CountVectorizers. Defaults to True.
            fit_substitutions (bool, optional): whether or not to discover the most common substitutions. Defaults to True.
            fit_grammar_mistakes (bool, optional): whether or not to find all the grammar mistakes (this can take several hours). Defaults to True.
            fit_pos_ngrams (bool, optional): whether or not to find the most common part of speech ngrams. Defaults to True.
        """
        X = [self._clean_text(x) for x in X]
        
        if fit_vectorizers:
            print('Fitting Vectorizers')
            self.word_vectorizer.fit(X)
            self.char_vectorizer.fit(X)
        
        if fit_substitutions:
            collected_substitutions = []    
            [collected_substitutions.extend(self._get_substitutions(x)) for x in tqdm(X, desc='Discovering Substitutions', file=(open(self.log_file, 'w') if self.log_file else None))]

            self.substitutions = Counter(collected_substitutions).most_common(self.substitution_max_features)
        
        if fit_grammar_mistakes:
            self.grammar_mistakes_per_chunk = {}
            total_grammar_mistakes = []
            
            grammar_checker = GrammarChecker()
            grammar_checker.init()
            
            for id, chunk_text in tqdm(enumerate(X), total=len(X), desc='Discovering Grammar Mistakes', file=(open(self.log_file, 'w') if self.log_file else None)):
                mistakes_in_chunk = self._get_grammar_mistake_rules_for_chunk(chunk_text, grammar_checker, num_processes=4)
                self.grammar_mistakes_per_chunk[id] = mistakes_in_chunk
                total_grammar_mistakes.extend(mistakes_in_chunk)
            
            grammar_checker.close()
        
            self.grammar_mistake_rules = Counter(total_grammar_mistakes).most_common()
        
        if fit_pos_ngrams:
            all_pos_ngrams = []
            
            for x in tqdm(X, desc='Discovering POS Ngrams', file=(open(self.log_file, 'w') if self.log_file else None)):
                tagger = POSTagger(x)
                all_pos_ngrams.extend(tagger.get_pos_ngrams(self.pos_ngram_size)) 
            
            self.pos_features = Counter(all_pos_ngrams).most_common(self.max_pos_features)
                
    def transform(self,X: List[str],
                  normalize: bool=False,
                  grammar_counts: bool=True,
                  word_ngrams: bool=True,
                  char_ngrams: bool=True,
                  edit_distance: bool=True,
                  substitutions: bool=True,
                  grammar_mistakes: bool=True,
                  function_words: bool=True,
                  pos_ngrams: bool=True,
                  average_sentence_length: bool=True,
                  log_file: str=None) -> pd.DataFrame:

        """Transform the given data into a pandas DataFrame containing the features.

        Args:
            X (List[str]): A list of strings, each representing to the text of a chunk.
            normalize (bool, optional): whether or not to normalize the word and char ngram features. Defaults to False.
            grammar_counts (bool, optional): whether or not to include the number of grammar mistakes in each chunk, if set to False this feature will be binary. Defaults to True.
            word_ngrams (bool, optional): whether or not to include the word ngram features. Defaults to True.
            char_ngrams (bool, optional): whether or not to include the char ngram features. Defaults to True.
            edit_distance (bool, optional): whether or not to include the edit distance features. Defaults to True.
            substitutions (bool, optional): whether or not to include the substitution features. Defaults to True.
            grammar_mistakes (bool, optional): whether or not to include the grammar mistake features. Defaults to True.
            function_words (bool, optional): whether or not to include the function word features. Defaults to True.
            pos_ngrams (bool, optional): whether or not to include the part of speech ngram features. Defaults to True.
            average_sentence_length (bool, optional): whether or not to include the average sentence length feature. Defaults to True.
            log_file (str, optional): A path to a log file to write progress to. Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the features.
        """
        features = []
        feature_names = [] 

        if word_ngrams:
            features.append(self.get_word_ngram_features(X, normalize=normalize))
            feature_names.extend([f'word_ngram: {name}' for name in self.word_vectorizer.get_feature_names_out()])
        
        if char_ngrams:
            features.append(self.get_char_ngram_features(X, normalize=normalize))
            feature_names.extend([f'char_ngram: {name}' for name in self.char_vectorizer.get_feature_names_out()])

        if edit_distance:
            features.append(self.get_edit_distance(X, normalize=normalize, log_file=log_file).reshape(-1,1))
            feature_names.append('edit_distance')
        
        if substitutions:
            features.append(self.get_substitution_features(X, normalize=normalize, log_file=log_file))
            feature_names.extend([f'substitution: {str(substitution[0])}' for substitution in self.substitutions])

        if grammar_mistakes:
            features.append(self.get_grammar_features(counts=grammar_counts, log_file=log_file))
            feature_names.extend([f'grammar_mistake: {str(mistake[0])}' for mistake in self.grammar_mistake_rules])
        
        if function_words:
            features.append(self.get_function_word_features(X, normalize=normalize, log_file=log_file))
            feature_names.extend([f'function_word: {word}' for word in self.function_words])
        
        if pos_ngrams:
            features.append(self.get_pos_features(X, normalize=normalize, log_file=log_file))
            feature_names.extend([f'pos_ngram: {str(pos_ngram[0])}' for pos_ngram in self.pos_features])
        
        if average_sentence_length:
            features.append(self.get_average_sentence_length(X, log_file=log_file).reshape(-1,1))
            feature_names.append('average_sentence_length')
        
        return pd.DataFrame(np.concatenate(features, axis=1), columns=feature_names)
            
            
    
    def get_word_ngram_features(self, X: List[str], normalize: bool=False) -> np.ndarray:
        """Get the word ngram features for the given data.

        Args:
            X (List[str]): A list of strings, each representing the text of a chunk.
            normalize (bool, optional): Whether or not to normalize the word ngram counts by the number of words in the chunk. Defaults to False.

        Returns:
            np.ndarray: A numpy array of shape (len(X), self.word_ngram_max) containing the word ngram counts.
        """
        X = [self._clean_text(x) for x in X]
        ngram_counts = np.array([self._get_ngram_count(x, 'word') for x in X])

        print('Creating word ngram features...')

        if normalize:
            output = self.word_vectorizer.transform(X).toarray() / ngram_counts[:, np.newaxis]
        
        else:
            output = self.word_vectorizer.transform(X).toarray()
        
        assert output.shape == (len(X), self.word_ngram_max), f'Expected output shape to be {(len(X), self.word_ngram_max)}, but got {output.shape}'

        return output
    
    def get_char_ngram_features(self, X: List[str], normalize: bool=False) -> np.ndarray:
        """Get the character ngram features for the given data.

        Args:
            X (List[str]): A list of strings, each representing the text of a chunk.
            normalize (bool, optional): Whether or not to normalize the character ngram counts by the number of characters in the chunk. Defaults to False.

        Returns:
            np.ndarray: A numpy array of shape (len(X), self.char_ngram_max) containing the character ngram counts.
        """

        X = [self._clean_text(x) for x in X]
        ngram_counts = np.array([self._get_ngram_count(x, 'char') for x in X]) 

        print('Creating char ngram features...')
        
        if normalize:
            output = self.char_vectorizer.transform(X).toarray() / ngram_counts[:, np.newaxis]
        
        else:
            output = self.char_vectorizer.transform(X).toarray()
        
        assert output.shape == (len(X), self.char_ngram_max), f'Expected output shape to be {(len(X), self.char_ngram_max)}, but got {output.shape}'

        return output

    def get_edit_distance(self, X: List[str], normalize: bool=False, log_file: str=None) -> np.array:
        """Get the total edit distance for each chunk.

        Args:
            X (List[str]): A list of strings, each representing to the text of a chunk.
            normalize (bool, optional): Whether or not to normalize the edit distance by the number of words in the chunk. Defaults to False.
            log_file (str, optional): A name of a log file to create. When not set, no log file is created. Defaults to None.

        Returns:
            np.array: A numpy array of shape (len(X),) containing the total edit distance for each chunk.
        """
        X = [self._remove_punctuation(self._clean_text(x)) for x in X]
        
        edit_distances = []

        for chunk_text in tqdm(X, desc='Calculating edit distance', file=open(log_file, 'w') if log_file else None):
            edit_distance = 0
            number_of_words_in_chunk = len(chunk_text.split())
            
            for word in chunk_text.split():
                edit_distance += self._get_edit_distance_for_word(word, self._spell_check_word(word))

            if normalize:
                edit_distance /= number_of_words_in_chunk

            edit_distances.append(edit_distance)

        output = np.array(edit_distances, dtype=np.float32)

        assert output.shape == (len(X),), f'Expected output shape to be {(len(X),)}, got {output.shape}'
        
        return output
    
    def get_substitution_features(self, X: List[str], normalize: bool=False, log_file: str=None) -> np.array:
        """Get the counts of the most common substitutions.

        Args:
            X (List[str]): a list of strings, each representing the text of a chunk.
            normalize (bool, optional): whether or not to normalize the features. Defaults to False.
            log_file (str, optional): a name of a log file to create. When not set, no log file is created. Defaults to None.

        Returns:
            np.array[np.int32]: A numpy array of shape (len(X), self.substitution_max_features) containing the normalized counts of the most common substitutions.
        """
         
        X = [self._remove_punctuation(self._clean_text(x)) for x in X]
        
        substitutions = [substitution[0] for substitution in self.substitutions]
        
        features_per_chunk = []
        
        for chunk_text in tqdm(X, desc='Extracting Substitution Features', file=open(log_file, 'w') if log_file else None):
            substitutions_in_chunk = []
            features = []
            
            for word in chunk_text.split():
                substitutions_in_chunk.extend(self._get_substitutions_for_word(word))
                
            for substitution in substitutions:
                features.append(substitutions_in_chunk.count(substitution))
            
            if normalize: 
                number_of_words_in_chunk = len(chunk_text.split())
                features = np.array(features, dtype=np.float32) / number_of_words_in_chunk
            
            else: 
                features = np.array(features, dtype=np.int32)
            
            features_per_chunk.append(features)
            
        output = np.array(features_per_chunk)

        assert output.shape == (len(X), self.substitution_max_features), f'Expected output shape to be {(len(X), self.substitution_max_features)}, got {output.shape}'
        
        return output
           
    def get_grammar_features(self, chunk_texts: List[str]=None, counts: bool=False, log_file: str=None) -> np.array:
        """Get the grammar features for the given data.

        Args:
            X (List[str], optional): The text for a single chunk. If not passed, will return feature for all the chunks the extractor was fit on.
            counts (bool, optional): If set to True the features will be count of the number of each grammar mistake and not binary values. Defaults to False.
            log_file (str, optional): A name of a log file to create. When not set, no log file will be created. Defaults to None.

        Raises:
            RuntimeError: If the feature extractor was not fit on any data.

        Returns:
            np.array: A numpy array of shape (len(X), len(self.grammar_mistake_rules)) containing the grammar features.
        """
        feature_names = [t[0] for t in self.grammar_mistake_rules]

        if not chunk_texts:
            if not self.grammar_mistakes_per_chunk:
                raise RuntimeError('Grammar mistakes per chunk not found. Please initialize the feature extractor first.')

            feature_matrix = np.zeros((len(self.grammar_mistakes_per_chunk.keys()), len(feature_names)), dtype=np.int8)
            
            for i in tqdm(range(len(self.grammar_mistakes_per_chunk.keys())), desc='Extracting Grammar Features', file=open(log_file, 'w') if log_file else None):
                for j, feature_name in enumerate(feature_names):
                    if counts:
                        feature_matrix[i, j] = self.grammar_mistakes_per_chunk[i].count(feature_name)
                    else:
                        feature_matrix[i, j] = 1 if feature_name in self.grammar_mistakes_per_chunk[i] else 0
                        
            return feature_matrix
            
        else:
            feature_matrix = np.zeros((len(chunk_texts),len(feature_names)), dtype=np.int8)        
            grammar_checker = GrammarChecker()
            grammar_checker.init()

            for i, chunk in tqdm(enumerate(chunk_texts), total=len(chunk_texts), desc='Extracting Grammar Features', file=open(log_file, 'w') if log_file else None):
                mistakes = self._get_grammar_mistake_rules_for_chunk(chunk, grammar_checker=grammar_checker)
                
                for j, feature_name in enumerate(feature_names):
                    if counts:
                        feature_matrix[i][j] = mistakes.count(feature_name)
                    else:
                        feature_matrix[i][j] = 1 if feature_name in mistakes else 0

            grammar_checker.close()

            return feature_matrix 

    def get_function_word_features(self, X: List[str], normalize: bool=False, log_file: str=None) -> np.array:
        """Get the function word features for the given data.

        Args:
            X (List[str]): A list of strings, each representing the text of a chunk.
            normalize (bool, optional): Whether or not to normalize the features by the number of words in each chunk. Defaults to False.
            log_file (str, optional): A name of a log file to create. When not set, no log file is created. Defaults to None.

        Returns:
            np.array: A numpy array of shape (len(X), self.function_word_max_features) containing the counts of the function words.
        """
        if self.max_function_words_features:
            number_of_features = self.max_function_words_features
        else:
            number_of_features = len(self.function_words)
            
        output = np.zeros((len(X), number_of_features), dtype=np.float32)
        
        for i, chunk_text in tqdm(enumerate(X), desc='Extracting Function Word Features', file=open(log_file, 'w') if log_file else None):
            words_in_chunk = self._remove_punctuation(self._clean_text(chunk_text)).split()

            for j, function_word in enumerate(self.function_words[:number_of_features]):
                output[i, j] = words_in_chunk.count(function_word) 

            if normalize:
                output[i,:] /= len(words_in_chunk)        

        return output
                
    def get_pos_features(self, X: List[str], normalize: bool=False, log_file: str=None) -> np.array:
        """Get the part of speech features for the given data.

        Args:
            X (List[str]): A list of strings, each representing the text of a chunk.
            normalize (bool, optional): Whether or not to normalize the features by the number of words in each chunk. Defaults to False.
            log_file (str, optional): A name of a log file to create. When not set, no log file is created. Defaults to None.

        Returns:
            np.array: A numpy array of shape (len(X), self.pos_max_features) containing the counts of the part of speech tags.
        """
        if not self.pos_features:
            raise RuntimeError('Part of speech features not found. Please initialize the feature extractor first.')
         
        output = np.zeros((len(X), len(self.pos_features)), dtype=np.float32)
        
        pos_ngrams = [feature[0] for feature in self.pos_features]
        
        for i, chunk_text in tqdm(enumerate(X), desc='Extracting POS Features', file=open(log_file, 'w') if log_file else None):
            chunk_text = self._clean_text(chunk_text)
            words_in_chunk = self._remove_punctuation(chunk_text).split()
            
            pos_tagger = POSTagger(chunk_text)
            pos_ngrams_in_chunk = pos_tagger.get_pos_ngrams(n=self.pos_ngram_size)
            
            for j, pos_ngram in enumerate(pos_ngrams):
                output[i, j] = pos_ngrams_in_chunk.count(pos_ngram)

            if normalize:
                output[i,:] /= len(words_in_chunk)        

        return output
    
    def get_average_sentence_length(self, X: List[str], log_file: str=None) -> np.array:
        """Get the average sentence length for each chunk.

        Args:
            X (List[str]): List of strings, each representing the text of a chunk.
            log_file: log file to create. When not set, no log file is created. Defaults to None.

        Returns:
            np.array: an array of shape (len(X),) containing the average sentence length for each chunk.
        """
        output = np.zeros((len(X),), dtype=np.float32)
        
        for i, chunk_text in tqdm(enumerate(X), desc='Extracting Average Sentence Length', total=len(X), file=open(log_file, 'w') if log_file else None):
            sentences = chunk_text.split('\n')
            output[i] = np.mean([len([word for word in sentence.split() if len(word) > 1 or (len(word) == 1 and word.isalnum())]) for sentence in sentences])
            
        return output
            
    def _get_substitutions(self, text: str) -> List[Tuple[str, str] or Tuple[str, str, str]]:
        text = self._remove_punctuation(text)
        substitutions = [] 

        [substitutions.extend(self._get_substitutions_for_word(word)) for word in text.split()]

        return substitutions
        
    def _get_substitutions_for_word(self, word: str) -> List[Tuple[str, str] or Tuple[str, str, str]]:
        substitutions = []
        
        original = word
        corrected = self._spell_check_word(word)
        
        if original == corrected:
            return substitutions
        
        matcher = difflib.SequenceMatcher(None, original, corrected)
        
        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            if opcode == 'insert':
                substitutions.append(('insert', corrected[j1:j2]))

            elif opcode == 'delete':
                substitutions.append(('delete', original[i1:i2]))

            elif opcode == 'replace':
                substitutions.append(('replace', original[i1:i2], corrected[j1:j2]))
        
        return substitutions

    def _get_ngram_count(self, text: str, analyzer: str) -> int:
        if analyzer == 'char':
            vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=self.char_ngram_range)
        if analyzer == 'word':
            vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=self.word_ngram_range)

        return len(vectorizer.build_analyzer()(text))

    def _spell_check_word(self, word: str) -> str:
        return self._spell_checker.get_closest_correction(word)
    
    def _get_edit_distance_for_word(self, word1: str, word2: str) -> int:
        return self._spell_checker.get_levenshtein_distance(word1, word2) 
    
    def _get_grammar_mistake_rules_for_chunk(self, chunk_text: str, grammar_checker: GrammarChecker ,num_processes: int=4) -> List[str]: 
        sentences = chunk_text.split('\n')
         
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            grammar_mistake_rules = executor.map(grammar_checker.get_grammar_mistake_rules, sentences) 
        
        output = []
        [output.extend(mistake_rules) for mistake_rules in grammar_mistake_rules]
        
        return output
    
    
    def _clean_text(self, text: str) -> str:
        return text.replace('\n', ' ')

    def _remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))
    
    