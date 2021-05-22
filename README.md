---
language: en
datasets:
- common_voice
- librispeech_asr
- timit_asr
metrics:
- wer
- cer
tags:
- audio
- automatic-speech-recognition
- speech
- xlsr-fine-tuning-week
license: apache-2.0
model-index:
- name: XLSR Wav2Vec2 English by Jonatas Grosman
  results:
  - task: 
      name: Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Common Voice en
      type: common_voice
      args: en
    metrics:
       - name: Test WER
         type: wer
         value: 19.76
       - name: Test CER
         type: cer
         value: 8.60
---

# Wav2Vec2-Large-XLSR-53-English

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on English using the [Common Voice](https://huggingface.co/datasets/common_voice), [LibriSpeech](https://huggingface.co/datasets/librispeech_asr) and [TIMIT](https://huggingface.co/datasets/timit_asr),.
When using this model, make sure that your speech input is sampled at 16kHz.

The script used for training can be found here: https://github.com/jonatasgrosman/wav2vec2-sprint

## Usage

The model can be used directly (without a language model) as follows:

```python
import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "en"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
SAMPLES = 10

test_dataset = load_dataset("common_voice", LANG_ID, split=f"test[:{SAMPLES}]")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = batch["sentence"].upper()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)

for i, predicted_sentence in enumerate(predicted_sentences):
    print("-" * 100)
    print("Reference:", test_dataset[i]["sentence"])
    print("Prediction:", predicted_sentence)
```

| Reference  | Prediction |
| ------------- | ------------- |
| "SHE'LL BE ALL RIGHT." | SHE'D BE ALRIGHT |
| SIX | SIX |
| "ALL'S WELL THAT ENDS WELL." | ALL IS WELL THAT ENDS WELL |
| DO YOU MEAN IT? | DO YOU MEAN IT |
| THE NEW PATCH IS LESS INVASIVE THAN THE OLD ONE, BUT STILL CAUSES REGRESSIONS. | THE NEW PATCH IS LESS INVASIVE THAN THE OLD ONE BUT STILL CAUSES REGRESSION |
| HOW IS MOZILLA GOING TO HANDLE AMBIGUITIES LIKE QUEUE AND CUE? | HOW IS MUSILA GOING TO HANDLE ANB HOOTIES LIKE QU AND QU |
| "I GUESS YOU MUST THINK I'M KINDA BATTY." | RISIONAS INCI IN TE BACTY |
| NO ONE NEAR THE REMOTE MACHINE YOU COULD RING? | NO ONE NEAR THE REMOTE MACHINE YOU COULD RING |
| SAUCE FOR THE GOOSE IS SAUCE FOR THE GANDER. | SAUCE FOR THE GUISE IS SAUCE FOR THE GONDER |
| GROVES STARTED WRITING SONGS WHEN SHE WAS FOUR YEARS OLD. | GRAFS STARTED WRITING SOUNDS WHEN SHE WAS FOUR YEARS OLD |

## Evaluation

The model can be evaluated as follows on the English test data of Common Voice.

```python
import torch
import re
import librosa
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "en"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
DEVICE = "cuda"

CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"]

test_dataset = load_dataset("common_voice", LANG_ID, split="test")

# uncomment the following lines to eval using other datasets
# test_dataset = load_dataset("librispeech_asr", "clean", split="test")
# test_dataset = load_dataset("librispeech_asr", "other", split="test")
# test_dataset = load_dataset("timit_asr", split="test")

wer = load_metric("wer.py") # https://github.com/jonatasgrosman/wav2vec2-sprint/blob/main/wer.py
cer = load_metric("cer.py") # https://github.com/jonatasgrosman/wav2vec2-sprint/blob/main/cer.py

chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.to(DEVICE)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speech_array, sampling_rate = librosa.load(batch["file"] if "file" in batch else batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["text"] if "text" in batch else batch["sentence"]).upper()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch

result = test_dataset.map(evaluate, batched=True, batch_size=8)

predictions = [x.upper() for x in result["pred_strings"]]
references = [x.upper() for x in result["sentence"]]

print(f"WER: {wer.compute(predictions=predictions, references=references, chunk_size=1000) * 100}")
print(f"CER: {cer.compute(predictions=predictions, references=references, chunk_size=1000) * 100}")
```

**Test Result**:

In the table below I report the Word Error Rate (WER) and the Character Error Rate (CER) of the model. I ran the evaluation script described above on other models as well (on 2021-05-20). Note that the table below may show different results from those already reported, this may have been caused due to some specificity of the other evaluation scripts used. Initially, I've tested the model only using the Common Voice dataset. Later I've also tested the model using the LibriSpeech and TIMIT datasets, which are better-behaved datasets than the Common Voice, containing only examples in US English extracted from audiobooks.

---

**Common Voice**

| Model | WER | CER |
| ------------- | ------------- | ------------- |
| jonatasgrosman/wav2vec2-large-xlsr-53-english | **19.76%** | **8.60%** |
| jonatasgrosman/wav2vec2-large-english | 21.16% | 9.53% |
| facebook/wav2vec2-large-960h-lv60-self | 22.03% | 10.39% |
| facebook/wav2vec2-large-960h-lv60 | 23.97% | 11.14% |
| facebook/wav2vec2-large-960h | 32.79% | 16.03% |
| boris/xlsr-en-punctuation | 34.81% | 15.51% |
| facebook/wav2vec2-base-960h | 39.86% | 19.89% |
| facebook/wav2vec2-base-100h | 51.06% | 25.06% |
| elgeish/wav2vec2-large-lv60-timit-asr | 59.96% | 34.28% |
| facebook/wav2vec2-base-10k-voxpopuli-ft-en | 66.41% | 36.76% |
| elgeish/wav2vec2-base-timit-asr | 68.78% | 36.81% |

---

**LibriSpeech (clean)**

| Model | WER | CER |
| ------------- | ------------- | ------------- |
| facebook/wav2vec2-large-960h-lv60-self | **1.86%** | **0.54%** |
| facebook/wav2vec2-large-960h-lv60 | 2.15% | 0.61% |
| facebook/wav2vec2-large-960h | 2.82% | 0.84% |
| facebook/wav2vec2-base-960h | 3.44% | 1.06% |
| jonatasgrosman/wav2vec2-large-xlsr-53-english | 4.16% | 1.28% |
| facebook/wav2vec2-base-100h | 6.26% | 2.00% |
| jonatasgrosman/wav2vec2-large-english | 8.00% | 2.55% |
| elgeish/wav2vec2-large-lv60-timit-asr | 15.53% | 4.93% |
| boris/xlsr-en-punctuation | 19.28% | 6.45% |
| elgeish/wav2vec2-base-timit-asr | 29.19% | 8.38% |
| facebook/wav2vec2-base-10k-voxpopuli-ft-en | 31.82% | 12.41% |

---

**LibriSpeech (other)**

| Model | WER | CER |
| ------------- | ------------- | ------------- |
| facebook/wav2vec2-large-960h-lv60-self | **3.89%** | **1.40%** |
| facebook/wav2vec2-large-960h-lv60 | 4.45% | 1.56% |
| facebook/wav2vec2-large-960h | 6.49% | 2.52% |
| jonatasgrosman/wav2vec2-large-xlsr-53-english | 8.82% | 3.42% |
| facebook/wav2vec2-base-960h | 8.90% | 3.55% |
| jonatasgrosman/wav2vec2-large-english | 13.62% | 5.24% |
| facebook/wav2vec2-base-100h | 13.97% | 5.51% |
| boris/xlsr-en-punctuation | 26.40% | 10.11% |
| elgeish/wav2vec2-large-lv60-timit-asr | 28.39% | 12.08% |
| elgeish/wav2vec2-base-timit-asr | 42.04% | 15.57% |
| facebook/wav2vec2-base-10k-voxpopuli-ft-en | 45.19% | 20.32% |

---

**TIMIT**

| Model | WER | CER |
| ------------- | ------------- | ------------- |
| facebook/wav2vec2-large-960h-lv60-self | **5.17%** | **1.33%** |
| facebook/wav2vec2-large-960h-lv60 | 6.24% | 1.54% |
| jonatasgrosman/wav2vec2-large-xlsr-53-english | 6.81% | 2.02% |
| facebook/wav2vec2-large-960h | 9.63% | 2.19% |
| facebook/wav2vec2-base-960h | 11.48% | 2.76% |
| elgeish/wav2vec2-large-lv60-timit-asr | 13.83% | 4.36% |
| jonatasgrosman/wav2vec2-large-english | 13.91% | 4.01% |
| facebook/wav2vec2-base-100h | 16.75% | 4.79% |
| elgeish/wav2vec2-base-timit-asr | 25.40% | 8.16% |
| boris/xlsr-en-punctuation | 25.93% | 9.99% |
| facebook/wav2vec2-base-10k-voxpopuli-ft-en | 51.08% | 19.84% |
