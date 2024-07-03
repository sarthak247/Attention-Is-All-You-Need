from pathlib import Path
import torch
import torch.nn as nn
from config import get_config, get_weights_file_path
from train import get_model, get_ds, run_validation, greedy_decode


def encode_text(config, src_text, tokenizer_src):
    seq_len = config['seq_len']
    sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
    eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
    pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)
    enc_input_tokens = tokenizer_src.encode(src_text).ids
    enc_padding_tokens = seq_len - len(enc_input_tokens) - 2
    # breakpoint()
    encoder_input = torch.cat(
                [
                    sos_token,
                    torch.tensor(enc_input_tokens, dtype = torch.int64),
                    eos_token,
                    torch.tensor([pad_token] * enc_padding_tokens, dtype = torch.int64)
                ]
            )
    # encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int() # (1, 1, seq_len)
    encoder_mask = (encoder_input != pad_token).int()

    return encoder_input, encoder_mask

def run_validation(model, src_text, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    model.eval()
    with torch.no_grad():
        encoder_input = encoder_input.to(device)
        encoder_mask = encoder_mask.to(device)

        # assert encoder_input.size(0) == 1 # B.S is 1

        model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

        source_text = src_text
        model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

        # Print to console
        print(f'SOURCE: {source_text}')
        print(f'PREDICTED: {model_out_text}')
    return model_out_text

if __name__ == '__main__':
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    config = get_config()
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Load custom checkpoint
    # epoch = 1

    # Load last checkpoint
    p = Path(config['model_folder']).glob('**/*')
    files = [x for x in p if x.is_file()]
    files.sort()
    epoch = str(files[-1]).split('_')[-1].split('.')[0]

    model_filename = get_weights_file_path(config, f"{epoch}")
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    # src_text = input('Enter text to convert: ')
    src_text = 'Hi! My name is Sarthak'
    encoder_input, encoder_mask = encode_text(src_text)
    run_validation(model, src_text, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
