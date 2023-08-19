import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
import numpy as np
from tqdm import tqdm


class TransformerFeatureExtractor:
    """ This class is used to extract features using a transformer model.
    """
    def __init__(self, model_name: str, max_length: int=2048):
        """Initialize the transformer feature extractor.

        Args:
            model_name (str): A pretrained model name from the huggingface model hub or a local path.
            max_length (int, optional): The input length the model expects. Defaults to 2048.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def transform(self, X : List[str] or str, log_file=None) -> np.ndarray:
        """Transform a list of strings or a single string into a feature vector.

        Args:
            X (List[str]or str): A list of strings or a single string. Each corresponding to the text of a chunk.
            log_file (str, optional): A name of a log file to create. If not set, no file will be created. Defaults to None.

        Returns:
            np.ndarray: A feature vector of shape (len(X), model.config.hidden_size).
        """
        
        X_tokenized = self._tokenize(X)
        
        X_tokenized_truncated = self._truncate_tokens(X_tokenized)

        X_tokenized_truncated_padded = self._pad_tokens(X_tokenized_truncated)
       
        assert all([len(x) == self.max_length for x in X_tokenized_truncated_padded])

        input_ids = self._convert_tokens_to_ids(X_tokenized_truncated_padded)
        attention_masks = self._create_attention_mask(input_ids)

        input_id_tensors = [torch.tensor(x).to(self.device) for x in input_ids]
        attention_mask_tensors = [torch.tensor(x).to(self.device) for x in attention_masks]
        
        self.model.to(self.device)
        self.model.eval()
        
        features = []
        
        with torch.no_grad():
            for input_ids, attention_masks in tqdm(zip(input_id_tensors, attention_mask_tensors),
                                                   total=len(input_id_tensors),
                                                   desc='Extracting features',
                                                   file=open(log_file, 'w') if log_file else None):
                outputs = self.model(input_ids.unsqueeze(0), attention_mask=attention_masks.unsqueeze(0))
                features.append(outputs['last_hidden_state'][:,0,:].cpu().numpy())
        
        self.model.to('cpu')
        torch.cuda.empty_cache()
        
        features = np.concatenate(features, axis=0)
        
        assert features.shape == (len(X), self.model.config.hidden_size)
        
        return features
        
    def _tokenize(self, X: List[str] or str) -> List[List[str]] or List[str]:
         
        if isinstance(X, str):
            X = X.replace('\n', self.tokenizer.sep_token)
            X = self.tokenizer.cls_token + X + self.tokenizer.sep_token

            return self.tokenizer.tokenize(X)

        tokens = []

        for x in X:
            x = x.replace('\n', self.tokenizer.sep_token)
            x = self.tokenizer.cls_token + x + self.tokenizer.sep_token
            tokens.append(self.tokenizer.tokenize(x))

        return tokens

    def _truncate_tokens(self, X: List[List[str]] or List[str]) -> List[List[str]] or List[str]:
        if isinstance(X, str):
            if len(X) > self.max_length:
                return X[:self.max_length -1] + [self.tokenizer.sep_token]

        return [x[:self.max_length -1] + [self.tokenizer.sep_token] if len(x) > self.max_length else x for x in X]
        
    def _pad_tokens(self, X: List[List[str]] or List[str]) -> List[List[str]] or List[str]:
        if isinstance(X[0], str):
            if len(X) < self.max_length: 
                return X + [self.tokenizer.pad_token] * (self.max_length - len(X))

        return [x + [self.tokenizer.pad_token] * (self.max_length - len(x)) if len(x) < self.max_length else x for x in X]

    def _convert_tokens_to_ids(self, X: List[List[str]] or List[str]) -> List[List[int]] or List[int]:
        if isinstance(X[0], str):
            return self.tokenizer.convert_tokens_to_ids(X)

        return [self.tokenizer.convert_tokens_to_ids(x) for x in X]

    def _create_attention_mask(self, X: List[List[int]] or List[int]) -> List[List[int]] or List[int]:
        if isinstance(X[0], int):
            return [1 if x != self.tokenizer.pad_token_id else 0 for x in X]

        return [[1 if id != self.tokenizer.pad_token_id else 0 for id in x] for x in X]