import logging
import random
from collections import defaultdict
import os
import statistics
from typing import Tuple

class DataBalancer:
    """A class to balance the text chunks data. 
    """
    def __init__(self, random_seed: int=None):
        """Initialize the data balancer.

        Args:
            random_seed (int, optional): Optional random seed to use. Defaults to None.
        """
        self.random_seed = random_seed
        if self.random_seed:
            random.seed(random_seed)
        self.label_to_language = {
            "Austria" : "German",
            "Germany" : "German",
            "Australia" : "English",
            "Ireland" : "English",
            "NewZealand" : "English",
            "UK" : "English",
            "US" : "English",
            "Bulgaria" : "Bulgarian",
            "Croatia" : "Croatian",
            "Czech" : "Czech",
            "Estonia" : "Estonian",
            "Finland" : "Finish",
            "France" : "French",
            "Greece" : "Greek",
            "Hungary" : "Hungarian",
            "Italy" : "Italian",
            "Lithuania" : "Lithuanian",
            "Netherlands" : "Dutch",
            "Norway" : "Norwegian",
            "Poland" : "Polish",
            "Portugal" : "Portuguese",
            "Romania" : "Romanian",
            "Russia" : "Russian",
            "Serbia" : "Serbian",
            "Slovenia" : "Slovenian",
            "Spain" : "Spanish",
            "Mexico" : "Spanish",
            "Sweden" : "Swedish",
            "Turkey" : "Turkish",
        }
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO)
        self.logger = logging.getLogger(__name__)

    def create_balanced_folder(self, raw_data_dir: str, output_dir: str, authors_per_language: int, max_chunks_per_author: int, excluded_users: set=None) -> None:
        """Create a folder that will be of the following structure:
        output_dir
        ├── language1
        │   ├── username1
        │   │   ├── chunk1
        │   │   ├── chunk2
        |	|   ...
        │   │   └── chunkN
        │   ├── username2
        │   │   ├── chunk1
        │   │   ├── chunk2
        |   |   ...
        │   │   └── chunkN
        │   ...
        │   language2
        │   ├── username1
        │   │   ├── chunk1
        ...

        Args:
            raw_data_dir (str): Path to the raw text chunks data folder you want to downsample. The folder should be of the following structure:
            
            input_dir
            ├── country1
            │   ├── username1
            │   │   ├── chunk1
            │   │   ├── chunk2
            |   |   ...
            │   │   └── chunkN
            │   ├── username2
            │   │   ├── chunk1
            │   │   ├── chunk2
            |   |   ...
            │   │   └── chunkN
            │   ...
            │── country2
            │   ├── username1
            │   │   ├── chunk1
            │   │   ├── chunk2
            ... 
            
            
            output_dir (str): Path to the folder where you want to save the downsampled data.
            authors_per_language (int): Number of authors per language to sample.
            max_chunks_per_author (int): Maximum number of chunks per author to sample.
            excluded_users (set, optional): An optional set of users to exclude from the downsampled data. Defaults to None.
        """
        self.logger.info('Running!')
 
        os.makedirs(output_dir, exist_ok=True)

        language_to_authors, author_to_chunks = self._get_data_from_dir(raw_data_dir,
                                                                           max_chunks_per_author=max_chunks_per_author,
                                                                           authors_per_language=authors_per_language,
                                                                           excluded_users=excluded_users ) 

        self._create_folders(output_dir=output_dir, language_to_authors=language_to_authors, author_to_chunks=author_to_chunks)
        
        self.logger.info('Done!')
        
    def get_statistics(self, raw_data_dir: str, balanced: bool=False) -> None:
        """ Get statistics about the raw data folder. These include the number of authors, the number of authors per language,
        the number of chunks, the number of chunks per author.
        
        Args:
            raw_data_dir (str): Path to the raw text chunks data folder you want to downsample.
            balanced (bool, optional): Whether the data is balanced or not. Defaults to False.
        """
        language_authors = {}
        language_chunks = {}
        language_author_chunks = defaultdict(list)
        print('Statistics: ') 
        print(f'Found {len(os.listdir(raw_data_dir))} languages in {raw_data_dir}')
        for folder in os.listdir(raw_data_dir):
            if not balanced:
                label = folder.split('.')[1]
                if label == 'Ukraine':
                    continue # Ignoring Ukraine because it is multilingual
                language = self.label_to_language[label]
            else:
                language = folder

            authors = os.listdir(f'{raw_data_dir}/{folder}')
            language_authors[language] = len(authors)

            num_chunks = 0
            for author in authors:
                chunks = os.listdir(f'{raw_data_dir}/{folder}/{author}')
                num_chunks += len(chunks)
                language_author_chunks[language].append(len(chunks))
            language_chunks[language] = num_chunks

        print(' ')
        print(f"""
        Total number of authors is: {sum(language_authors.values())}
        Max number of authors per language is: {max(language_authors.values())} for {max(language_authors, key=language_authors.get)}
        Min number of authors per language is: {min(language_authors.values())} for {min(language_authors, key=language_authors.get)}
        Average number of authors per language is: {sum(language_authors.values()) / len(language_authors)}
        Median number of authors per language is: {statistics.median(language_authors.values())}

        Total number of chunks is: {sum(language_chunks.values())}
        Max number of chunks per language: {max(language_chunks.values())} for {max(language_chunks, key=language_chunks.get)}
        Min number of chunks per language: {min(language_chunks.values())} for {min(language_chunks, key=language_chunks.get)}
        Average number of chunks per language is: {sum(language_chunks.values()) / len(language_chunks)}
        Median number of chunks per language is: {statistics.median(language_chunks.values())}

        Max number of chunks per author is: {max([max(chunks) for chunks in language_author_chunks.values()])}
        Min number of chunks per author is: {min([min(chunks) for chunks in language_author_chunks.values()])}
        Average number of chunks per author is {sum([sum(chunks) for chunks in language_author_chunks.values()]) / sum(language_authors.values())}
        Median number of chunks per author is {statistics.median([number for chunks in language_author_chunks.values() for number in chunks])}
              """)

        return language_authors, language_chunks, language_author_chunks

    def _get_data_from_dir(self, data_dir: str, max_chunks_per_author: int, authors_per_language: int, excluded_users: set=None) -> Tuple[dict, dict]:
        language_to_authors = defaultdict(set)
        author_to_chunks = defaultdict(list)

        for label_folder in os.listdir(f'{data_dir}'):
            self.logger.info(f'Dowsampling {label_folder} in {data_dir}')

            label = label_folder.split('.')[1]
            if label == 'Ukraine':
                continue # Ignoring Ukraine because it is multilingual
            language = self.label_to_language[label]

            for author_name in os.listdir(f'{data_dir}/{label_folder}'):
                if excluded_users and author_name in excluded_users:
                    continue

                language_to_authors[language].add(author_name)
                user_chunks = []
                
                # Randomly sample chunks for each author
                for chunk in os.listdir(f'{data_dir}/{label_folder}/{author_name}'):
                    with open(os.path.join(data_dir, label_folder, author_name, chunk), 'r') as f:
                        text = ''.join(f.readlines()).lower()
                        user_chunks.append(text)

                author_to_chunks[author_name] = random.sample(user_chunks, min(max_chunks_per_author, len(user_chunks)))

        # Randomly sample authors for each language
        for language, authors in language_to_authors.items():
            language_to_authors[language] = random.sample(authors, authors_per_language)

        sampled_users = set()

        for language, authors in language_to_authors.items():
            if not len(authors) == authors_per_language:
                self.logger.warning(f'{language} does not have {authors_per_language} authors, it has {len(authors)}')
            assert len(authors) == authors_per_language

            sampled_users = sampled_users.union(authors)

        assert len(sampled_users) == authors_per_language * 23, f'Total number of authors is: {len(sampled_users)}, should be {authors_per_language * 23}' 

        # Remove chunks from authors that are not sampled
        author_names = [key for key in author_to_chunks.keys()]
        for author_name in author_names:
            if not author_name in sampled_users:
                author_to_chunks.pop(author_name)

        assert all(len(authors) == authors_per_language for authors in language_to_authors.values()), f'All languages should have {authors_per_language} authors, but some dont\'t'
        assert all([len(chunks) <= max_chunks_per_author for chunks in author_to_chunks.values()]), f'All authors should have {max_chunks_per_author} chunks, but some have more'
        return language_to_authors, author_to_chunks

    def _create_folders(self, output_dir: str, language_to_authors: dict, author_to_chunks: dict) -> None:
        for author_name, chunks in author_to_chunks.items():
            author_label = None

            for language in language_to_authors.keys():
                if author_name in language_to_authors[language]:
                    author_label = language
                    break
            language_folder = f'{output_dir}/{author_label}'

            if not os.path.exists(language_folder):
                os.makedirs(language_folder)

            author_folder = f'{output_dir}/{language}/{author_name}'
            if not os.path.exists(author_folder):
                os.makedirs(author_folder)

            for chunk_num, chunk in enumerate(chunks):
                with open(f'{author_folder}/chunk{chunk_num + 1}', 'w') as f:
                    f.write(chunk)
