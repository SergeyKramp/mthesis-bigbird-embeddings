import os
from collections import defaultdict
import random
from transformers import AutoTokenizer
from typing import Tuple
from .data_chunk import Chunk
from .reddit_dataset import RedditDataset


class DataProcessor:
    """Class that processes the data from the Reddit dataset. It discovers the chunks and creates a training and test dataset object.
    """

    def __init__(self, tokenizer_str: str, clean_up_chunk_text: bool=True) -> None:
        """Initialize the data processor.

        Args:
            tokenizer_str (str): The name of the tokenizer to use. Can either be a local path or a name from the huggingface transformers library.
            clean_up_chunk_text (bool, optional): Whether or not to clean up the chunk's text by removing blank spaces and replacing URLs with the word "URL". Defaults to True.
        """
        self.lang2usernames = defaultdict(list)
        self.user2chunks = defaultdict(list)
        self.clean_up_chunk_text = clean_up_chunk_text
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        self.number_of_languages = 23
        self.language2label = {
                'English': 0,
                'German': 1,
                'Bulgarian': 2,
                'Croatian': 3,
                'Czech': 4,
                'Estonian': 5,
                'Finish': 6,
                'French': 7,
                'Greek': 8,
                'Hungarian': 9,
                'Italian': 10,
                'Lithuanian': 11,
                'Dutch': 12,
                'Norwegian': 13,
                'Polish': 14,
                'Portuguese': 15,
                'Romanian': 16,
                'Russian': 17,
                'Serbian': 18,
                'Slovenian': 19,
                'Spanish': 20,
                'Swedish': 21,
                'Turkish': 22
            }


    def discover_chunks(self, data_dir: str) -> None:
        """Discovers the chunks from a directory which is of the following structure:
        data_dir
        ├── language1
        │   ├── username1
        │   │   ├── chunk1
        │   │   ├── chunk2	
        │   │   └── chunk3
        │   ├── username2
        │   │   ├── chunk1
        │   │   ├── chunk2
        │   │   └── chunk3
        │   ...
        │   language2
        │   ├── username1
        │   │   ├── chunk1
        ...
           
        

        Args:
            data_dir (str, optional): path to data directory.
        """

        for language_folder in os.listdir(data_dir):
            language = language_folder

            for username in os.listdir(f'{data_dir}/{language_folder}'):
                user_chunks = []

                self.lang2usernames[language].append(username)

                for chunk in os.listdir(f'{data_dir}/{language_folder}/{username}'):
                    full_path = f'{data_dir}/{language_folder}/{username}/{chunk}' 

                    with open(full_path, 'r') as f:
                        id = f'{username}_{chunk}'

                        user_chunks.append(Chunk(id=id,
                                                 author_name=username,
                                                 text=f.read(),
                                                 clean_up_chunk_text=self.clean_up_chunk_text,
                                                 label=self.language2label[language],
                                                 tokenizer=self.tokenizer,
                                                 ))
                        
                self.user2chunks[username] = user_chunks

        assert len(self.lang2usernames) == self.number_of_languages, f'Number of languages is not correct. Should be {self.number_of_languages} but is {len(self.lang2usernames)}'


    def get_train_test_datasets(self, train_size: float=0.8, sequence_length: int=2048, seed: int=None, split_by_chunks: bool=False) -> Tuple[RedditDataset, RedditDataset]:
        """Creates a training and test RedditDataset object tuple from the discovered chunks.

        Args:
            train_size (float, optional): The proportion of the data that will be used for training. Defaults to 0.8.
            sequence_length (int, optional): The size of the input tensor for the transformer model. Defaults to 2048.
            seed (int, optional): Random seed to set. Defaults to None.
            split_by_chunks (bool, optional): Determines if the data will be split by chunks. If False the data is split by users. Defaults to False.

        Raises:
            Exception: If chunks are not discovered yet.

        Returns:
            Tuple[RedditDataset, RedditDataset]: Training and test RedditDataset object tuple.
        """
        if not self.user2chunks:
            raise Exception('Chunks not discovered yet. Please call discover_chunks() first.')

        if seed:
            random.seed(seed)
        
        if split_by_chunks:
            chunks = [chunk for chunks in self.user2chunks.values() for chunk in chunks]
            random.shuffle(chunks)

            training_chunks = chunks[:int(len(chunks) * train_size)]
            test_chunks = chunks[int(len(chunks) * train_size):]
            
            assert len(training_chunks) + len(test_chunks) == len(chunks), 'Training and test chunks do not add up to the total number of chunks'
            
            assert len(training_chunks) == int(len(chunks) * train_size)\
                or len(training_chunks) == int(len(chunks) * train_size) + 1\
                or len(training_chunks) == int(len(chunks) * train_size) - 1\
                , f'Training size is not correct. Should be ~ {int(len(chunks) * train_size)} but is {len(training_chunks)}'
                
            assert len(test_chunks) == int(len(chunks) * (1 - train_size))\
                or len(test_chunks) == int(len(chunks) * (1 - train_size)) + 1\
                or len(test_chunks) == int(len(chunks) * (1 - train_size)) - 1\
                , f'Test size is not correct. Should be ~ {int(len(chunks) * (1 - train_size))} but is {len(test_chunks)}'
            
        else:
            users = list(self.user2chunks.keys())
            random.shuffle(users)
            
            training_chunks = [chunk for user in users[:int(len(users) * train_size)] for chunk in self.user2chunks[user]]
            test_chunks = [chunk for user in users[int(len(users) * train_size):] for chunk in self.user2chunks[user]]    

            chunks = [chunk for chunks in self.user2chunks.values() for chunk in chunks]
            
            assert len(training_chunks) + len(test_chunks) == len(chunks)\
                , 'Training and test chunks do not add up to the total number of chunks'
                
            training_users = set([chunk.author_name for chunk in training_chunks])
            testing_users = set([chunk.author_name for chunk in test_chunks])
            
            assert training_users.intersection(testing_users) == set(), 'Training and test users are not disjoint'
            assert len(training_users) == int(len(users) * train_size)\
                or len(training_users) == int(len(users) * train_size) + 1\
                or len(training_users) == int(len(users) * train_size) - 1\
                    , f'Training size is not correct. Should be ~ {int(len(users) * train_size)} but is {len(training_users)}'
                    
            assert len(testing_users) == int(len(users) * (1 - train_size))\
                or len(testing_users) == int(len(users) * (1 - train_size)) + 1\
                or len(testing_users) == int(len(users) * (1 - train_size)) - 1\
                    , f'Test size is not correct. Should be ~ {int(len(users) * (1 - train_size))} but is {len(testing_users)}'


        return RedditDataset(chunks=training_chunks, max_length=sequence_length), RedditDataset(chunks=test_chunks, max_length=sequence_length)

    def split_dataset(self, dataset: RedditDataset, train_size: float=0.8, shuffle: bool=True, seed: int=None) -> Tuple[RedditDataset, RedditDataset]:
        """Splits a RedditDataset object into a training and test RedditDataset object tuple.

        Args:
            dataset (RedditDataset): The dataset to split. If the dataset contains encodings it will be split by encodings
            and the labels will be split accordingly. If the dataset does not contain encodings it will be split by chunks.
            
            NOTE: when splitting by encodings the new datasets will not contain chunks!
            
            train_size (float, optional): The proportion of the data that will be used for training. Defaults to 0.8.
            shuffle (bool, optional): Determines if the data will be shuffled before splitting. Defaults to True.
            random_seed (int, optional): Random seed to set. Defaults to None.

        Returns:
            Tuple[RedditDataset, RedditDataset]: Training and test RedditDataset object tuple.
        """
        if seed:
            random.seed(seed)

        if dataset.encodings:
            index_list = list(range(len(dataset)))
            
            if shuffle:
                random.shuffle(index_list)
            
            training_encodings = {key: [value[i] for i in index_list[:int(len(index_list) * train_size)]] for key, value in dataset.encodings.items()}
            training_labels = [dataset.labels[i] for i in index_list[:int(len(index_list) * train_size)]]
            
            testing_encodings = {key: [value[i] for i in index_list[int(len(index_list) * train_size):]] for key, value in dataset.encodings.items()}
            testing_labels = [dataset.labels[i] for i in index_list[int(len(index_list) * train_size):]]
            
            assert len(training_encodings['input_ids']) + len(testing_encodings['input_ids']) == len(dataset.encodings['input_ids'])\
                , 'Training and test encodings do not add up to the total number of encodings'
            assert len(training_labels) + len(testing_labels) == len(dataset.labels)\
                , 'Training and test labels do not add up to the total number of labels'
            
            return RedditDataset(encodings=training_encodings, labels=training_labels, max_length=dataset.max_length),\
                    RedditDataset(encodings=testing_encodings, labels=testing_labels, max_length=dataset.max_length)
        else:
            index_list = list(range(len(dataset.chunks)))
            if shuffle:       
                random.shuffle(index_list)
            
            training_chunks = dataset.chunks[:int(len(index_list) * train_size)]
            test_chunks = dataset.chunks[int(len(index_list) * train_size):]

            assert len(training_chunks) + len(test_chunks) == len(dataset.chunks), 'Training and test chunks do not add up to the total number of chunks'

        return RedditDataset(training_chunks, max_length=dataset.max_length), RedditDataset(test_chunks, max_length=dataset.max_length)
        

