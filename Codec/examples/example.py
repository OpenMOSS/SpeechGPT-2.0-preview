

import os
import argparse
import sys
import logging
import torchaudio
import numpy as np
import torch

from utils import set_logging, waiting_for_debug, load_audio, save_audio
from models.codec import Generator

def main():
    set_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/sg2_codec_config.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--audio_input", type=str, required=True)
    
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
    
    with torch.no_grad():
        x = load_audio(args.audio_input, generator.sampling_rate).reshape(1, 1, -1).to(device)
        tokens = generator.inference_tokenize(x) 
        y = generator.inference_detokenize(tokens)
        save_audio(f"examples/gen_{os.path.basename (args.audio_input)}", y.detach().cpu().reshape(1, -1), generator.sampling_rate)

if __name__ == "__main__":
    main()