#!/bin/bash

pip install -r requirements.txt
mkdir -p ckpt
wget -O ./ckpt/model_tracker_fixed_e20.pt https://hf-mirror.com/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt
