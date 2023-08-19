# Native Language Identification with Big Bird Embeddings

This is the repository for the code used in the paper [Native Language Identification with Big Bird Embeddings](https://arxiv.org/abs/2104.05247) by [Sergey Kramp](sergey), [Giovanni Cassani](https://research.tilburguniversity.edu/en/persons/giovanni-cassani) and [Chris Emmery](https://research.tilburguniversity.edu/en/persons/chris-emmery). The code is released under the [MIT license](https://opensource.org/licenses/MIT). The Reddit L2 dataset used in the work is available on [here](http://cl.haifa.ac.il/projects/L2/).
For citing this work, please use the following bibtex entry:

```
TBD
```

## TLDR
In this work we used embeddings from a fine-tuned Big Bird [_(Zaheer et al., 2020)_](https://arxiv.org/abs/2007.14062) model to perform Native Language Identification (NLI) on the Reddit L2 dataset.
Here you will find the code used to sample the data, script used for fine-tuning Big Bird, and notebooks containing the experiments described in the paper. 

What you will not find in this repository is the data used. You can download the Reddit L2 dataset [here](http://cl.haifa.ac.il/projects/L2/) (it appears as __Reddit-L2 chunks__).

## How to use the code
- **data**: contains classes for working with the data.
  - **databalancer.py**: contains a DataBalancer class that is used to balance the data by sampling chunks or authors from the Reddit L2 dataset directory and creating folders by label (the author's native language instead of the author's country as it is in the original dataset). To work with this class you need to have the Reddit L2 dataset downloaded and extracted.
  - **dataprocessor.py**: contains a DataProcessor class that is used to discover the text chunks from the Reddit L2 dataset directory and create datasets.
  - **data_chunk.py**: contains a Chunk class, which is created by the DataProcessor and contains the text chunk, the metadata for the chunk, such as the author name and the label. It also contains methods for tokenizing the text and getting token ids.
  - **reddit_dataset.py**: contains a RedditDataset, which is created by the DataProcessor and contains Chunks. It inherits from the torch Dataset class and can be used with a torch DataLoader for fine-tuning. It also contains a method for getting a Pandas DataFrame with the data.
- **feature_extractors**: contains classes for extracting linguistic features and embeddings from the data.
  - **normal_feature_extractor.py**: contains a NormalFeatureExtractor class that is used to extract linguistic features from the data. In particular, it extracts the following features: 
   
    1. Character and word n-grams using the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) implementation.
    2. Spelling mistake feautures using the [symspellpy](https://pypi.org/project/symspellpy/) package.
    3. Grammar mistake features using the [language-tool-python](https://pypi.org/project/language-tool-python/) package.
    4. Function word used using a list of function words from [*Volansky et al. (2015)*](https://www.semanticscholar.org/paper/On-the-features-of-translationese-Volansky-Ordan/766ea82ccfe78dcfcf813fd2f594d03ab06a75a6)
    5. POS (part-of-speech) tags using the [NLTK](https://www.nltk.org/) library.
    6. Average sentence length.
  - **transformer_feature_extractor.py**: contains a TransformerFeatureExtractor class that is used to extract embeddings from the data. In our case, we used it with Big Bird, but it can be used with any model on the [Hugging Face](https://huggingface.co/) library.   
- **fine_tuning_scripts**: contains scripts for fine-tuning Big Bird as well some log files produced during fine-tuning.
- **language_checkers**: contains wrapper classes around the *symspellpy* and *language-tool-python* packages that are used be the NormalFeatureExtractor, as well as some required text files.
- **notebooks**: all notebooks that start with "*experiment*" refer to an experiment described in the paper. *balance_data.ipynb* contains the data balancing code and *figures.ipynb* contains the code for generating the figures in the paper.

### Note
Through out the code you might encounter folders that don't exists in the repository, in particular: *fine_tuned_models* (used for storing the model checkpoints and tokenizers) and *pickles* (used for storing intermediate results). You can create these folders yourself or change the code to store the results in a different location.