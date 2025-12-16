# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Zhipu AI Inc (authors: CogAudio Group Members)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import torchaudio


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists

def read_json_lists(list_file):
    lists = read_lists(list_file)
    results = {}
    for fn in lists:
        with open(fn, 'r', encoding='utf8') as fin:
            results.update(json.load(fin))
    return results

def load_wav(wav, target_sr, device=None):
    import torch
    # Prioritize CUDA, then MPS (Apple Silicon), then CPU if no device specified
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.to(device)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=target_sr
                ).to(device)
        speech = resampler(speech).to(device)
    return speech

def speed_change(waveform, sample_rate, speed_factor: str):
    effects = [
        ["tempo", speed_factor],  # speed_factor
        ["rate", f"{sample_rate}"]
    ]
    augmented_waveform, new_sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform,
        sample_rate,
        effects
    )
    return augmented_waveform, new_sample_rate
