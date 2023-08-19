
from typing import Set, List
from torch.utils.data import Dataset
import torch
import pandas as pd
from .data_chunk import Chunk

class RedditDataset(Dataset):
    def __init__(self, chunks: List[Chunk]=None, max_length: int=512, encodings: dict=None, labels: List[int]=None) -> None:
        """This is a dataset object that contains chunks used for either training or testing. It can be used as input for a PyTorch DataLoader object or for creating a Pandas DataFrame out of the chunks.

        Args:
            chunks (List[Chunk], optional): The chunks to store in the dataset. Defaults to None.
            max_length (int, optional): The maximum length of the chunks. Defaults to 512.
            encodings (dict, optional): A dictionary of encodings containing input_ids and attention masks. Defaults to None.
            labels (List[int], optional): The labels for each input_id attention mask pair. Defaults to None.
        """
        self.chunks = chunks
        self.max_length = max_length
        self.encodings = encodings
        self.labels = labels
        self.dataframe = None


    def __getitem__(self, idx):
        if not self.encodings or not self.labels:
           raise Exception('Dataset object not initialized. Call extract_encodings_and_labels_from_chunks() method to initialize the dataset object.')
        
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


    def __len__(self):
        if not self.labels:
            raise Exception('Dataset object not initialized. Call extract_encodings_and_labels_from_chunks() method to initialize the dataset object.')
        return len(self.labels)
   
    
    def extract_encodings_and_labels_from_chunks(self) -> None:
        """Extracts the encodings and labels from the chunks and stores them in the dataset object. This is methods needs to be called to make this object usable by a PyTorch DataLoader object.
        """
        if self.encodings and self.labels:
            return
        
        encodings = {'input_ids': [], 'attention_mask': []}
        labels = []
        for chunk in self.chunks:
            input_ids, attention_mask = chunk.get_encoding(max_length=self.max_length)
            
            if isinstance(input_ids[0], list):
                [encodings['input_ids'].append(input_id) for input_id in input_ids]
                [encodings['attention_mask'].append(attention_mask) for attention_mask in attention_mask]
                labels.extend([chunk['label']] * len(input_ids))
            
            else:
                encodings['input_ids'].append(input_ids)
                encodings['attention_mask'].append(attention_mask)
                labels.append(chunk.label)
        
        assert len(encodings['input_ids']) == len(encodings['attention_mask']), 'Input ids and attention mask are not the same length'
        assert len(encodings['input_ids']) == len(labels), 'Input ids and labels are not the same length'

        self.encodings = encodings
        self.labels = labels
        
        
    def get_dataframe(self) -> pd.DataFrame:
        """ Creates a Pandas DataFrame out of the chunks.

        Returns:
            pd.DataFrame: Pandas DataFrame with the following columns: chunk_id, text, label, author_name
        """
        return pd.DataFrame({'chunk_id': [chunk.id for chunk in self.chunks],
                             'text': [chunk.text for chunk in self.chunks],
                             'label': [chunk.label for chunk in self.chunks],
                             'author_name': [chunk.author_name for chunk in self.chunks]})
        
    
    def get_author_names(self) -> Set[str]:
        """Returns a list of author names that are present in the chunks.

        Returns:
            Set[str]: List of author names.
        """
        return set([chunk.author_name for chunk in self.chunks])
        
