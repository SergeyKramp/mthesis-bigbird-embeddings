#%% Imports and constants
from data.dataprocessor import DataProcessor
import os
import logging
import torch
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
import pickle
import datetime
import numpy as np
import time
import random


logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('out_of_domain_bert_large_fine_tuning_clean_chunks.log')
stream_handler = logging.StreamHandler()
file_handler.setLevel(logging.INFO)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

batch_size = 2
seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 3

data_processor = DataProcessor('bert-large-uncased')

with open('pickles/pickled_datasets/seed_42/out_of_domain_fine_tune_dataset_clean_chunks.pkl', 'rb') as f:
    fine_tune_dataset = pickle.load(f)

fine_tune_dataset.max_length = 512

logger.info('Loaded fine-tune dataset')

#%% Initialize datasets and dataloaders

logger.info('Initializing fine-tune dataset')
fine_tune_dataset.extract_encodings_and_labels_from_chunks()
logger.info('Extracted encodings and labels from chunks and initialized dataset')

fine_tune_dataset, fine_tune_validation_dataset = data_processor.split_dataset(fine_tune_dataset, train_size=0.9, shuffle=True, seed=seed)
logger.info('Split fine-tune data to fine-tune and validation datasets')

train_dataloader = DataLoader(fine_tune_dataset, batch_size=batch_size, shuffle=False)
validation_dataloader = DataLoader(fine_tune_validation_dataset, batch_size=batch_size, shuffle=False)

#%% Load model and optimizer
model = AutoModelForSequenceClassification.from_pretrained('bert-large-uncased',
                                                      num_labels=23,
                                                      output_attentions=False,
                                                      output_hidden_states=True)
model.cuda(device)

optimizer = AdamW(model.parameters(),
                    lr=1e-6,
                    eps=1e-8)

warm_up_steps = len(fine_tune_dataset) * 0.1
total_steps = len(fine_tune_dataset) * num_epochs
     
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warm_up_steps,
                                            num_training_steps=total_steps)

#%% Helper functions for training and validation

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

#%% Training loop

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Store the loss value for plotting the learning curve.
training_loss = []
validation_loss = []

logger.info('Training...')

for epoch_i in range(num_epochs):
    training_loss_epoch = []
    validation_loss_epoch = []
    logger.info(f'======== Epoch {epoch_i + 1} / {num_epochs} ========')
    
    t0 = time.time()
    
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        
        optimizer.zero_grad()
         
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids,
                        token_type_ids=None,
                        attention_mask=attention_mask,
                        labels=labels)
        
        loss = outputs[0]
        
        loss.backward()
        
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 and not step == 0:
            training_loss_epoch.append(loss.item())

        if step % 1000 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            avg_loss = np.mean(training_loss_epoch)
            logger.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.      Average Loss: {:}'.format(step, len(train_dataloader), elapsed, round(avg_loss, 4)))
        
    avg_train_loss = np.mean(training_loss_epoch)

    logger.info("  Epoch Average Training Loss: {0:.2f}".format(avg_train_loss))
    logger.info("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    logger.info("Running Validation...")

    t0 = time.time()

    model.eval()

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids,
                            token_type_ids=None,
                            attention_mask=attention_mask,
                            labels=labels)
        
        loss = outputs[0]

        validation_loss_epoch.append(loss.item())

    # Report the final accuracy for this validation run.
    logger.info("  Average Validation Loss: {0:.2f}".format(np.mean(validation_loss_epoch)))
    logger.info("  Validation took: {:}".format(format_time(time.time() - t0)))
    
    training_loss.append(training_loss_epoch)
    validation_loss.append(validation_loss_epoch)

logger.info("")
logger.info("Training complete!") 

#%% Save model and loss values

# pickle loss and accuracy values
with open('fine_tuned_models/out_of_domain_bert_large_training_loss_clean_chunks.pkl', 'wb') as f:
    pickle.dump(training_loss, f)

with open('fine_tuned_models/out_of_domain_bert_large_validation_loss_clean_chunks.pkl', 'wb') as f:
    pickle.dump(validation_loss, f)

logger.info('Pickled loss values')

# save model and tokenizer
model.save_pretrained('fine_tuned_models/out_of_domain_bert_large_clean_chunks')
data_processor.tokenizer.save_pretrained('fine_tuned_models/out_of_domain_bert_large_clean_chunks')

logger.info('Saved model and tokenizer')
logger.info('Done!')