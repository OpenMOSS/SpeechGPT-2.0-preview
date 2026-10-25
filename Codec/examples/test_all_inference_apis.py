

import os
import argparse
import sys
import logging
import torchaudio
import numpy as np
import torch

from utils import set_logging, waiting_for_debug, load_audio, save_audio
from models.codec import Generator

@torch.no_grad()
def save_all_audios(generator, wav_list, output_dir):
    logging.info(f"##### Start save wavs ##### ")
    for i, x in enumerate(wav_list):
        x = x.reshape(1, 1, -1) # x.shape = (1, 1, AudioLength)
        save_audio(os.path.join(output_dir, f"gt_{i}.wav"), x.detach().squeeze().unsqueeze(0).cpu(), sample_rate=generator.sampling_rate)
        logging.info("-----------------------------")

@torch.no_grad()
def run_nobatch_inference_tokenize_detokenize(generator, wav_list, output_dir):

    logging.info(f"##### Start inference_tokenize_detokenize ##### ")
    for i, x in enumerate(wav_list):
        x = x.reshape(1, 1, -1) # x.shape = (1, 1, AudioLength)
        non_streaming_tokens = generator.inference_tokenize(x) # tokens.shape = (NumLayers, 1, tokensLength)
        non_streaming_y = generator.inference_detokenize(non_streaming_tokens) # non_streaming_y.shape = (1, 1, AudioLength)
        save_audio(os.path.join(output_dir, f"nobatch_inference_tokenize_detokenize_{i}.wav"),  non_streaming_y.detach().squeeze().unsqueeze(0).cpu(), sample_rate=generator.sampling_rate)
        logging.info("-----------------------------")

def make_batch_data(data, device=torch.device('cuda')):
    batch_lengths = [d.shape[0] for d in data]
    max_len = max(batch_lengths)
    batch_data = torch.zeros(len(data), 1, max_len).to(device)  #[B,1,T]

    for i,d in enumerate(data):
        batch_data[i][:,:d.shape[0]] = d
    return batch_data, batch_lengths

@torch.no_grad()
def run_streaming_inference_tokenize_detokenize_without_input_state(generator, wav_list, output_dir, batch_size, device):

    logging.info(f"##### Start test run_continuous_batch_streaming_inference_tokenize_detokenize ##### ")
    frame_size = 960 
    assert len(wav_list) % batch_size == 0
    for i in range(0, len(wav_list), batch_size):
        logging.info(f"processing wav [{i}, {i + batch_size})")
        x, x_lengths = make_batch_data([wav_list[j].squeeze() for j in range(i, i + batch_size)], device)
        tot_len = 0
        with generator.streaming(batch_size=batch_size):
            streaming_y = []
            streaming_tokens = []
            for idx, offset in enumerate(range(0, x.shape[-1], frame_size)):
                frame_input =  x[:, :, offset: offset + frame_size] # frame_input.shape = (batch_size, 1, frame_size)
                if offset + frame_size > x.shape[-1]: 
                    break
                tot_len += frame_size
                frame_tokens = generator.streaming_inference_tokenize(frame_input) 
                frame_output = generator.streaming_inference_detokenize(frame_tokens)
                streaming_y.append(frame_output)
                streaming_tokens.append(frame_tokens)
            streaming_y = torch.cat(streaming_y, dim=-1) # streaming_y.shape = (batch_size, 1, AudioMaxLength)
            streaming_tokens = torch.cat(streaming_tokens, dim=-1) # streaming_tokens.shape = (batch_size, 1, AudioMaxLength // 960)
        
        for j in range(batch_size):
            save_audio(
                os.path.join(output_dir, f"run_streaming_inference_tokenize_detokenize_without_input_state_{i + j}.wav"), 
                streaming_y[j][:, :x_lengths[j]].detach().squeeze().unsqueeze(0).cpu(), 
                sample_rate=generator.sampling_rate
            )
        
        logging.info("-----------------------------")

@torch.no_grad()
def run_streaming_inference_tokenize_detokenize_with_input_state(generator, wav_list, output_dir):
    logging.info(f"##### Start inference_tokenize_detokenize_with_state ##### ")
    frame_size = 960
    for i, x in enumerate(wav_list):
        x = x.reshape(1, 1, -1)
        tot_len = 0
        streaming_y = []
        streaming_tokens = []
        input_state_of_streaming_inference_tokenize = None 
        input_state_of_streaming_inference_detokenize = None 
        for idx, offset in enumerate(range(0, x.shape[-1], frame_size)):
            frame_input =  x[:, :, offset: offset + frame_size] # frame_input.shape = (batch_size=1, 1, frame_size)
            if offset + frame_size > x.shape[-1]: 
                break
            tot_len += frame_size
            streaming_inference_tokenize_with_state_result = generator.streaming_inference_tokenize_with_state(
                frame_input,
                input_state_of_streaming_inference_tokenize
            )
            
            frame_tokens = streaming_inference_tokenize_with_state_result['tokens']
            input_state_of_streaming_inference_tokenize = streaming_inference_tokenize_with_state_result['updated_state_of_streaming_inference_tokenize']
            
            streaming_inference_detokenize_with_state_result = generator.streaming_inference_detokenize_with_state(
                frame_tokens,
                input_state_of_streaming_inference_detokenize
            ) 
            
            frame_output = streaming_inference_detokenize_with_state_result['y']
            input_state_of_streaming_inference_detokenize = streaming_inference_detokenize_with_state_result['updated_state_of_streaming_inference_detokenize']
            streaming_y.append(frame_output)
            streaming_tokens.append(frame_tokens)
        streaming_y = torch.cat(streaming_y, dim=-1) # streaming_y.shape = (batch_size=1, 1, AudioMaxLength)
        streaming_tokens = torch.cat(streaming_tokens, dim=-1) # streaming_tokens.shape = (batch_size=1, 1, AudioMaxLength // 960)
        save_audio(os.path.join(output_dir, f"run_streaming_inference_tokenize_detokenize_with_input_state_{i}.wav"),  streaming_y.detach().squeeze().unsqueeze(0).cpu(), sample_rate=generator.sampling_rate)

def main():
    set_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/sg2_codec_config.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--audio_paths", nargs='+', required=True)
    parser.add_argument("--audio_output_dir", type=str, default="./examples/audios_output_for_test_all_inference_apis")
    parser.add_argument("--batch_size", type=int, default=2)
    
    parser.add_argument("--debug_ip", type=str)
    parser.add_argument("--debug_port", type=int)
    parser.add_argument("--debug", default=0, type=int, nargs="?",
        help='whether to debug or not',
    )
    args = parser.parse_args()
    if args.debug == 1:
        waiting_for_debug(args.debug_ip, args.debug_port)

    device = torch.device(args.device)

    generator = Generator.load_from_checkpoint(config_path=args.config_path, checkpoint_path=args.checkpoint_path).to(device).eval()
    
    wav_list = [
        load_audio(
            audio_path=audio_path, 
            target_sample_rate=generator.sampling_rate
        ).to(device) 
        for audio_path in args.audio_paths
    ]
    
    output_dir = args.audio_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    save_all_audios(generator, wav_list, output_dir)
    
    # inference_tokenize
    # inference_detokenize

    run_nobatch_inference_tokenize_detokenize(generator, wav_list, output_dir)
    
    # streaming_inference_tokenize
    # streaming_inference_detokenize

    run_streaming_inference_tokenize_detokenize_without_input_state(generator, wav_list, output_dir, batch_size=args.batch_size, device=device)
    
    # streaming_inference_tokenize_with_state
    # streaming_inference_detokenize_with_state

    run_streaming_inference_tokenize_detokenize_with_input_state(generator, wav_list, output_dir)
       
if __name__ == "__main__":
    main()