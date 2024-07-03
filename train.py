import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
# from torch.utils.tensorboard import SummaryWriter
import wandb
import torchmetrics

import warnings
from pathlib import Path
from tqdm import tqdm
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

text_table = wandb.Table(columns=['epoch', 'sample', 'loss', 'Source Text', 'Target Text', 'Predicted Text'])

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute encoder output for reuse for each token we get from decoder
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate decoder output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])

        # Select the token with the max probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim = 1)

        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, epoch, loss, num_samples = 2, ):
    model.eval()
    count = 0
    global text_table

    # Console width
    console_width = 80
    
    # Metrics
    source = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1 # B.S is 1

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            # Print to console
            print_msg('-' * console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            # Append for metric
            source.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Send to wandb
            text_table = wandb.Table(columns=text_table.columns, data=text_table.data)
            text_table.add_data(epoch, count, loss, source_text, target_text, model_out_text)
            run.log({'training_samples' : text_table})


            if count == num_samples:
                break
    
    if run:
        # Metrics: CER, WER, BLEU
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        run.log({'Character Error Rate' : cer})
        
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        run.log({'Word Error Rate' : wer})
        
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        run.log({'BLEU Score (Validation)' : bleu})


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]", ], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    # ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train[:02%]')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 90-10 split
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print('Max length of source sentence: ', max_len_src)
    print('Max length of target sentence: ', max_len_tgt)

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config, run):
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    # Create model folder
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filname = get_weights_file_path(config, config['preload'])
        print('Preloading model: ', model_filname)
        state = torch.load(model_filname)
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run them through the model
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (batch, seq_len)

            # (b, seq_len, tgt_vocab_size) -> (b * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f'loss' : f"{loss.item():6.3f}"})

            # Log the loss
            # writer.add_scalar('train_loss', loss.item(), global_step)
            # writer.flush()
            run.log(data = {'loss' : loss.item()})

            # Backpropogate the loss
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), epoch, loss)
        
        # Save the model
        model_filname = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step' : global_step
        }, model_filname)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    run = wandb.init(project='Transformers', job_type='train', name=f"{config['lang_src']}-{config['lang_tgt']}")
    # run = wandb.init(project=f'Test run', job_type='train')
    train_model(config, run)