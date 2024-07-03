import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from pathlib import Path
import torch
import torch.nn as nn
from config import get_config, get_weights_file_path
from train import get_model, get_ds, run_validation, greedy_decode
from inference import encode_text, run_validation
from tokenizers import Tokenizer

# Favicon and Title
st.set_page_config(page_title="Attention Is All You Need 📖", page_icon="🐱", layout="centered", initial_sidebar_state="auto", menu_items=None)

# SideBar
with st.sidebar:
    st.title("Attention Is All You Need 📖")
    st.markdown('''
    ## About
    This app is a demo of Transformers and language translations using:
    - [Streamlit](https://streamlit.io)
    - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    - [HuggingFace](https://huggingface.co/)
    ''')
    add_vertical_space(4)
    st.write("Made with :sparkling_heart: by [Sarthak Thakur](https://sarthak247.github.io)")

def get_tokenizers(config, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def main():
    # Main App
    st.header("Attention Is All You Need 📖")

    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_config()

    tokenizer_src = get_tokenizers(config, config['lang_src'])
    tokenizer_tgt = get_tokenizers(config, config['lang_tgt'])
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    p = Path(config['model_folder']).glob('**/*')
    files = [x for x in p if x.is_file()]
    files.sort()
    epoch = str(files[-1]).split('_')[-1].split('.')[0]

    model_filename = get_weights_file_path(config, f"{epoch}")
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    
    

    # Run inference
    src_text = st.text_input('Enter the text to convert: ')
    if src_text:
        encoder_input, encoder_mask = encode_text(config, src_text, tokenizer_src)
        response = run_validation(model, src_text, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
        st.write(response)



    

if __name__ == '__main__':
    main()