---
language: en
datasets:
- common_voice
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
         value: 18.98
       - name: Test CER
         type: cer
         value: 8.29
---

# Wav2Vec2-Large-XLSR-53-English

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on English using the [Common Voice](https://huggingface.co/datasets/common_voice).
When using this model, make sure that your speech input is sampled at 16kHz.

This model has been fine-tuned thanks to the GPU credits generously given by the [OVHcloud](https://www.ovhcloud.com/en/public-cloud/ai-training/) :)

The script used for training can be found here: https://github.com/jonatasgrosman/wav2vec2-sprint

## Usage

The model can be used directly (without a language model) as follows...

Using the [asrecognition](https://github.com/jonatasgrosman/asrecognition) library:

```python
from asrecognition import ASREngine

asr = ASREngine("en")

audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]
transcriptions = asr.transcribe(audio_paths)
```

Writing your own inference script:

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
| "SHE'LL BE ALL RIGHT." | SHE'LL BE ALL RIGHT |
| SIX | SIX |
| "ALL'S WELL THAT ENDS WELL." | ALL AS WELL THAT ENDS WELL |
| DO YOU MEAN IT? | DO YOU MEAN IT |
| THE NEW PATCH IS LESS INVASIVE THAN THE OLD ONE, BUT STILL CAUSES REGRESSIONS. | THE NEW PATCH IS LESS INVASIVE THAN THE OLD ONE BUT STILL CAUSES REGRESSION |
| HOW IS MOZILLA GOING TO HANDLE AMBIGUITIES LIKE QUEUE AND CUE? | HOW IS MOSLILLAR GOING TO HANDLE ANDBEWOOTH HIS LIKE Q AND Q |
| "I GUESS YOU MUST THINK I'M KINDA BATTY." | RUSTIAN WASTIN PAN ONTE BATTLY |
| NO ONE NEAR THE REMOTE MACHINE YOU COULD RING? | NO ONE NEAR THE REMOTE MACHINE YOU COULD RING |
| SAUCE FOR THE GOOSE IS SAUCE FOR THE GANDER. | SAUCE FOR THE GUICE IS SAUCE FOR THE GONDER |
| GROVES STARTED WRITING SONGS WHEN SHE WAS FOUR YEARS OLD. | GRAFS STARTED WRITING SONGS WHEN SHE WAS FOUR YEARS OLD |

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
        speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()
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

In the table below I report the Word Error Rate (WER) and the Character Error Rate (CER) of the model. I ran the evaluation script described above on other models as well (on 2021-06-17). Note that the table below may show different results from those already reported, this may have been caused due to some specificity of the other evaluation scripts used.

| Model | WER | CER |
| ------------- | ------------- | ------------- |
| jonatasgrosman/wav2vec2-large-xlsr-53-english | **18.98%** | **8.29%** |
| jonatasgrosman/wav2vec2-large-english | 21.53% | 9.66% |
| facebook/wav2vec2-large-960h-lv60-self | 22.03% | 10.39% |
| facebook/wav2vec2-large-960h-lv60 | 23.97% | 11.14% |
| boris/xlsr-en-punctuation | 29.10% | 10.75% |
| facebook/wav2vec2-large-960h | 32.79% | 16.03% |
| facebook/wav2vec2-base-960h | 39.86% | 19.89% |
| facebook/wav2vec2-base-100h | 51.06% | 25.06% |
| elgeish/wav2vec2-large-lv60-timit-asr | 59.96% | 34.28% |
| facebook/wav2vec2-base-10k-voxpopuli-ft-en | 66.41% | 36.76% |
| elgeish/wav2vec2-base-timit-asr | 68.78% | 36.81% |
