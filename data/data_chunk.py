from typing import List, Tuple
from typing import Union
from transformers import AutoTokenizer
import re

class Chunk:
    def __init__(self, id: str=None, author_name: str=None, text: str=None, clean_up_chunk_text: bool=False, label: int=None, tokenizer: AutoTokenizer=None, tokens: List[str]=None, token_ids: List[int]=None) -> None:
        """ This is a class to store a text chunk by an author and its metadata.

        Args:
            id (str, optional): This should be a string consisting of the name of the author and the chunk number. Defaults to None.
            author_name (str, optional): A string containing the authors username. Defaults to None.
            text (str, optional): The context of all 100 lines of text in the chunk. Defaults to None.
            clean_up_chunk_text (bool, optional): Whether or not to clean up the chunk's text by removing blank spaces and replacing URLs with the word "URL". Defaults to False.
            label (int, optional): A number representing the native language of the author of the text chunk. Defaults to None.
            tokenizer (BertTokenizer, optional): The associated tokenizer that will be used for tokenization and token id conversion. Defaults to None.
            tokens (List[str], optional): Token representation of the text. Defaults to None.
            token_ids (List[int], optional): The BERT token ids associated with the tokens. Defaults to None.
        """
        self.id = id
        self.author_name = author_name
        if clean_up_chunk_text:
            temp_text = text.replace(" \n  ", "\n").replace("  ", " ").replace(" '", "'").replace(" n'", "n'").replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!").replace(" :", ":").replace(" ;", ";")
            self.text = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', 'URL', temp_text)
        else:
            self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.token_ids = token_ids
       
    def __getitem__(self, key) -> Union[str, int, List[str], List[int]]:
        if key == 'tokens' and not self.tokens:
            self._tokenize()
        if key == 'token_ids' and not self.token_ids:
            self._tokenize()
            self._convert_tokens_to_ids()

        return self.__dict__[key]
   
    def get_encoding(self, max_length: int=512) -> Tuple[Union[List[int], List[List[int]]], Union[List[int], List[List[int]]]]:
        """ Returns the token ids and the associated attention mask for the chunk.

        Args:
            max_length (int, optional): The maximum length of the list of token ids (This corresponds to the maximum input length of the BERT model).
            If the length of the text is more than 512 tokens, the chunk will be broken down to subchunks. Defaults to 512.

        Returns:
            Tuple[Union[List[int], List[List[int]]], Union[List[int], List[List[int]]]]: A tuple containing the token ids and the attention mask.
            If the chunk is longer than the max length, the token ids and attention mask will be a list of lists containing the token_ids and attention_mask for each subchunk.
            Token_id lists and attention_mask lists that are shorter than the max length are padded to the max length. 
        """
        if not self.token_ids:
            self._tokenize()
            self._convert_tokens_to_ids()

        if len(self.token_ids) > max_length:
            input_ids_list = []
            attention_mask_list = []
            
            subchunks = self.split_to_subchunks(max_length=max_length, split_by_tokens=True)
            for subchunk in subchunks:
                input_ids, attention_mask = subchunk.get_encoding(max_length=max_length)
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
            
            assert all([len(input_ids) == len(attention_mask) for input_ids, attention_mask in zip(input_ids_list, attention_mask_list)]), 'Some input ids and attention masks are not the same length'
            assert all([len(input_ids) == max_length for input_ids in input_ids_list]), 'Some input ids are not of the max length'
                
            return input_ids_list, attention_mask_list 
        
        else:
            input_ids = self._pad_token_ids(max_length)
            attention_mask = self._create_attention_mask(self.token_ids, max_length)
            
            assert len(input_ids) == len(attention_mask), 'Input ids and attention mask are not the same length'
            assert len(input_ids) == max_length, 'Input ids are not the max length'
            assert all(mask == 0 for id, mask in zip(input_ids, attention_mask) if id == 0), 'Attention mask is not correct. Some padding tokens dont have an attention mask of 0'
            assert all(mask == 1 for id, mask in zip(input_ids, attention_mask) if id != 0), 'Attention mask is not correct. Some non-padding tokens dont have an attention mask of 1'

            return input_ids, attention_mask             

    def split_to_subchunks(self, max_length: int=512, split_by_tokens: bool=True) -> List['Chunk']:
        """ Splits the chunk into subchunks of a maximum length.

        Args:
            max_length (int, optional): The maximum length each sub chunk should be. Defaults to 512 (corresponding to BERT's input size).
            split_by_tokens (bool, optional): If True, the chunk will be split by BERT tokens. If False, the chunk will be split by words. Note: the subchunks will have an empty text
            attribute if this is set to true. Defaults to True.
        
        Returns:
            List[Chunk]: A list of chunks. 
        """
        if split_by_tokens and not self.tokenizer:
            print('Cannot split by tokens. Tokenizer not set')
            return
        
        if split_by_tokens and not self.tokens:
            self._tokenize()
        
        if split_by_tokens and not self.token_ids:
            self._convert_tokens_to_ids()
            
        total_length = len(self.tokens) if self.token_ids else len(self.tokens)
        subchunks = []
        
        if total_length <= max_length:
            subchunks.append(self)
        else:
            if split_by_tokens:
                for chunk_number, i in enumerate(range(0, total_length, max_length)):
                    cls_token = self.tokenizer.cls_token
                    sep_token = self.tokenizer.sep_token
                    cls_token_id = self.tokenizer.cls_token_id
                    sep_token_id = self.tokenizer.sep_token_id
                    
                    if i != 0:
                        tokens = [cls_token] + self.tokens[i:i+max_length-2] + [sep_token]
                        token_ids = [cls_token_id] + self.token_ids[i:i+max_length-2] + [sep_token_id]
                    else:
                        tokens = self.tokens[i:i+max_length-1] + [sep_token]
                        token_ids = self.token_ids[i:i+max_length-1] + [sep_token_id]
                        
                    subchunk = Chunk(id=self.id + f'_{chunk_number}', author_name=self.author_name, label=self.label, tokenizer=self.tokenizer, tokens=tokens, token_ids=token_ids)
                    subchunks.append(subchunk)
            else:
                text_words = self.text.split(' ')
                for chunk_number, i in enumerate(range(0, total_length, max_length)):
                    text = ' '.join(text_words[i:i+max_length])
                    subchunk = Chunk(id=self.id + f'_{chunk_number}', author_name=self.author_name, text=text, label=self.label, tokenizer=self.tokenizer)
                    subchunks.append(subchunk)
        
        assert all(len(subchunk.tokens) <= max_length for subchunk in subchunks if split_by_tokens), 'Subchunk length is greater than max length'
        assert all(len(subchunk.tokens) == len(subchunk.token_ids) for subchunk in subchunks if split_by_tokens), 'Subchunk tokens and token ids are not the same length'
        assert all(subchunk.tokens[0] == self.tokenizer.cls_token for subchunk in subchunks if split_by_tokens), 'Subchunks tokens dont start with [CLS]'
        assert all(subchunk.tokens[-1] == self.tokenizer.sep_token for subchunk in subchunks if split_by_tokens), 'Subchunks tokens dont end with [SEP]'
        assert all(subchunk.token_ids[0] == self.tokenizer.cls_token_id for subchunk in subchunks if split_by_tokens), 'Subchunks token ids dont start with cls_token_id'
        assert all(subchunk.token_ids[-1] == self.tokenizer.sep_token_id for subchunk in subchunks if split_by_tokens), 'Subchunks token ids dont end with sep_token_id'

        assert all(len(subchunk.text.split(' ')) <= max_length for subchunk in subchunks if not split_by_tokens), 'Subchunk length is greater than max length'
        
        return subchunks

    def _tokenize(self) -> None:
        if not self.text:
            print ('Cannot tokenize. Text not set')
        else:
            text = self.text.replace('\n', self.tokenizer.sep_token)
            text = self.tokenizer.cls_token + text + self.tokenizer.sep_token
            self.tokens = self.tokenizer.tokenize(text)
      
    def _convert_tokens_to_ids(self) -> None:
        if not self.tokens:
            print('Cannot convert tokens to ids. Tokens not created')
        if not self.tokenizer:
            print('Cannot convert tokens to ids. Tokenizer not set')
        if self.tokens and self.tokenizer:
            self.token_ids = self.tokenizer.convert_tokens_to_ids(self.tokens)
        
    def _create_attention_mask(self, token_ids, length: int=512) -> List[int]:
        if len(token_ids) < length:
            return [1] * len(token_ids) + [0] * (length - len(token_ids))
        else:
            return [1] * length
        
    def _pad_tokens(self, max_length: int=512) -> List[str]:
        if len(self.tokens) < max_length:
            result = self.tokens + [self.tokenizer.pad_token] * (max_length - len(self.token_ids))
        else:
            result = self.tokens
        
        assert len(result) == max_length, 'Padded tokens length is not equal to max length'
        
        return result

    def _pad_token_ids(self, max_length: int=512) -> List[int]:
        if len(self.token_ids) < max_length:
            result = self.token_ids + [self.tokenizer.pad_token_id] * (max_length - len(self.token_ids))
        else:
            result = self.token_ids

        assert len(result) == max_length, 'Padded token ids length is not equal to max length'
        
        return result


    def __str__(self) -> str:
        return f'ID: {self.id}, Author: {self.author_name}, Label: {self.label}, Tokens: {self.tokens}, Token IDs: {self.token_ids}'

    def __repr__(self) -> str:
        return str({
            'id': self.id,
            'author_name': self.author_name,
            'text': self.text,
            'label': self.label,
            'tokens': self.tokens,
            'token_ids': self.token_ids
        })
    