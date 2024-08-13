import os
import torch
import argparse
import numpy as np 
from transformers import AutoFeatureExtractor, WhisperModel

import torchaudio
import torch.nn as nn
import torch.nn.functional as F

import speechbrain
import librosa

from subprocess import CalledProcessError, run

#openai whispers load audio
SAMPLE_RATE=16000
def denorm(input_x):
    input_x = input_x*(5-0) + 0
    return input_x
    
def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

class MosPredictor(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.mean_net_conv = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        self.relu_ = nn.ReLU()
        self.sigmoid_ = nn.Sigmoid()
        
        self.ssl_features = 1280
        self.dim_layer = nn.Linear(self.ssl_features, 512)

        self.mean_net_rnn = nn.LSTM(input_size = 512, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True)
        self.mean_net_dnn = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )        

        self.sinc = speechbrain.nnet.CNN.SincConv(in_channels=1, out_channels=257, kernel_size=251, stride=256, sample_rate=16000)
        self.att_output_layer_quality = nn.MultiheadAttention(128, num_heads=8)                
        self.output_layer_quality = nn.Linear(128, 1)
        self.qualaverage_score = nn.AdaptiveAvgPool1d(1)  
     
        self.att_output_layer_intell = nn.MultiheadAttention(128, num_heads=8)           
        self.output_layer_intell = nn.Linear(128, 1)
        self.intellaverage_score = nn.AdaptiveAvgPool1d(1)  
                       
        self.att_output_layer_stoi= nn.MultiheadAttention(128, num_heads=8)          
        self.output_layer_stoi = nn.Linear(128, 1)        
        self.stoiaverage_score = nn.AdaptiveAvgPool1d(1) 

    def new_method(self):
        self.sin_conv 
                
    def forward(self, wav, lps, whisper):
        #SSL Features
        wav_ = wav.squeeze(1)  ## [batches, audio_len]
        ssl_feat_red = self.dim_layer(whisper.squeeze(1))
        ssl_feat_red = self.relu_(ssl_feat_red)
 
        #PS Features
        sinc_feat=self.sinc(wav.squeeze(1))
        unsq_sinc =  torch.unsqueeze(sinc_feat, axis=1)
        concat_lps_sinc = torch.cat((lps,unsq_sinc), axis=2)
        cnn_out = self.mean_net_conv(concat_lps_sinc)
        batch = concat_lps_sinc.shape[0]
        time = concat_lps_sinc.shape[2]        
        re_cnn = cnn_out.view((batch, time, 512))
        
        concat_feat = torch.cat((re_cnn,ssl_feat_red), axis=1)
        out_lstm, (h, c) = self.mean_net_rnn(concat_feat)
        out_dense = self.mean_net_dnn(out_lstm) # (batch, seq, 1)       
        
        quality_att, _ = self.att_output_layer_quality (out_dense, out_dense, out_dense) 
        frame_quality = self.output_layer_quality(quality_att)
        frame_quality = self.sigmoid_(frame_quality)   
        quality_utt = self.qualaverage_score(frame_quality.permute(0,2,1))

        int_att, _ = self.att_output_layer_intell (out_dense, out_dense, out_dense) 
        frame_int = self.output_layer_intell(int_att)
        frame_int = self.sigmoid_(frame_int)   
        int_utt = self.intellaverage_score(frame_int.permute(0,2,1))

                
        return quality_utt.squeeze(1), int_utt.squeeze(1), frame_quality.squeeze(2), frame_int.squeeze(2)

