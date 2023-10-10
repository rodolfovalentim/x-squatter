import os
import sys
import argparse
import pathlib
import comet_ml
import logging
import json 
import torch
import glob
import shutil

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner import Tuner

# my classes
from src import dataset
from src import transformer
from src import utils

from pytorch_lightning.callbacks import TQDMProgressBar
from ipatok import tokenise
import panphon
import pandas as pd

# Configuring log
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
terminal_logger = logging.getLogger()
for handler in terminal_logger.handlers:
    handler.setFormatter(formatter)
        
terminal_logger.setLevel(logging.INFO)
terminal_logger.propagate = False

os.environ["COMET_URL_OVERRIDE"] = "https://www.comet.com/clientlib/"
seed_everything(42, workers=True)

# Define the command-line arguments
parser = argparse.ArgumentParser(description='Train a Transformer model for sequence classification')
parser.add_argument('--experiment_name', type=str, default='soundsquatter', help='Name of the experiment for CometML model')
parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--frac', type=float, default=1, help='Fraction of the dataset to be used')
parser.add_argument('--seq_len', type=int, default=50, help='Max length for input')
parser.add_argument('--val_every_epoch', type=int, default=1, help='Verify model every N epochs')
parser.add_argument('--train_data', type=str, default='dataset/pronunciation/workshop_dataset-en-us.csv', help='Path to the training data file')
parser.add_argument('--output_dir', type=str, default='trained_models', help='Directory to save the trained model')
parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
parser.add_argument('--from_checkpoint', type=str, default=None, help='Load trained version from checkpoint')
parser.add_argument('--add_language', type=bool, default=True, help='Add language before.')

# Parse the command-line arguments
args = parser.parse_args()

if args.from_checkpoint is not None:
    base_path = pathlib.Path(args.from_checkpoint).parent.parent
    args.experiment_name = base_path.parent.name
    
    version = int(base_path.name.split('_')[-1])
    output_path = base_path
    base_path = base_path.parent.parent
else:
    # The base output path for the trained model
    base_path = pathlib.Path(args.output_dir) 
    output_path = base_path / args.experiment_name

    # Calcula the version of the model
    version = 0
    version_files = list(output_path.glob('version*'))

    if len(version_files) > 0:
        version = max([int(x.name[len('version') + 1:]) for x in version_files]) + 1

    # New output path with the version
    output_path = output_path / ('version_' + str(version))

terminal_logger.info(f'Running version {version}. Logging to {output_path}.')

# Create the folder for the trained model
output_path.mkdir(parents=True, exist_ok=True)

# Configure CometML and checkpoint
logger = CSVLogger(base_path, name=args.experiment_name, version=version, flush_logs_every_n_steps=100)
comet_logger = CometLogger(api_key="", 
                           workspace="", 
                           project_name=args.experiment_name,
                           experiment_name='version_' + str(version))

# Read configuration file 
config = utils.read_config(args.config)

# Saving command line arguments
terminal_logger.info('Saving command line arguments...')
with open(output_path / 'commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)


terminal_logger.info('Loading dataset.')
train_data = pathlib.Path(args.train_data)
if train_data.is_dir():
    dataset_dir = list(train_data.glob('*.csv'))
elif train_data.is_file():
    dataset_dir = [ train_data ]

ft = panphon.FeatureTable()
df = pd.concat([pd.read_csv(file) for file in dataset_dir]).sample(frac=args.frac)

if args.add_language:
    terminal_logger.info(f'Add language to grapheme column.')
    df['grapheme'] = df.apply(lambda row: f'[{row["language"]}] ' + f' [{row["language"]}] '.join(row["grapheme"].split(' ')), axis=1)

terminal_logger.info(f'Dataset size is {df.shape[0]}')
terminal_logger.info(f'Average length {df["phoneme"].str.len().mean()}')

phoneme_tokenizer = dataset.Tokenizer(seq_len=args.seq_len, phon_vector=True, data_type='phoneme')
phoneme_tokenizer.load_predefined_vocab()

grapheme_tokenizer = dataset.Tokenizer(seq_len=args.seq_len, phon_vector=False, data_type='grapheme')
grapheme_tokenizer.load_predefined_vocab()

inputs_vectors, attentions, inputs_digits = phoneme_tokenizer.encode_sentences(df['phoneme'].to_list(), 
                                                        add_sos_marker=False, 
                                                        add_eos_marker=False, 
                                                        return_attentions=True, 
                                                        return_digits=True)

labels = grapheme_tokenizer.encode_sentences(df['grapheme'].to_list(),
                                             add_sos_marker=True,
                                             add_eos_marker=True, 
                                             return_attentions=False)

pronunciation_dataset = dataset.TextDataset(inputs_vectors, attentions, labels)

dm = dataset.MultiLanguageTextDataModule(dataset=pronunciation_dataset, 
                                         batch_size=args.batch_size,
                                         num_workers=10,
                                         validation_split=0.1)

# Logging
terminal_logger.info('Done')

# Get vocabulary size for model inicialization
vocab_size_src, vocab_size_tgt = phoneme_tokenizer.get_vocab_size(), grapheme_tokenizer.get_vocab_size()
   
terminal_logger.info('Saving tokenizer info...')
phoneme_tokenizer.save(output_path / 'phoneme_vocab.npy')
grapheme_tokenizer.save(output_path / 'grapheme_vocab.npy')

# saving config file
terminal_logger.info('Saving config file...')
shutil.copy2(args.config, output_path)

terminal_logger.info('Initializing the model...')
model = transformer.LitTransformer(vocab_size_src=len(ft.names), 
                                   vocab_size_tgt=vocab_size_tgt,
                                   max_len=args.seq_len)  

checkpoint_callback = ModelCheckpoint(dirpath=output_path / 'checkpoints', save_top_k=-1, monitor="val_loss")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")

trainer = pl.Trainer(max_epochs=args.num_epochs,
                     devices="auto", 
                     accelerator="auto",
                     enable_progress_bar=True, 
                     logger=[
                         comet_logger,
                         logger
                     ], 
                     callbacks=[
                         TQDMProgressBar(),
                         checkpoint_callback,
                         early_stop_callback
                     ],
                     check_val_every_n_epoch=args.val_every_epoch
                    )

if args.from_checkpoint is not None:
    trainer.fit(model, ckpt_path=args.from_checkpoint, datamodule=dm)
else:
    trainer.fit(model=model, datamodule=dm) 
    
# trainer.validate(model=model, datamodule=dm)
# trainer.test(model=model, datamodule=dm)

terminal_logger.info(f'Saving file {version}.')
trainer.save_checkpoint(output_path / 'last_model.ckpt')
