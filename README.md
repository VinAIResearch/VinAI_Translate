# A Vietnamese-English Neural Machine Translation System

Our pre-trained VinAI Translate models are state-of-the-art text-to-text translation models for Vietnamese-to-English and English-to-Vietnamese, respectively. The general architecture and experimental results of the pre-trained models can be found in our [VinAI Translate system paper](https://www.isca-speech.org/archive/interspeech_2022/nguyen22e_interspeech.html):

    @inproceedings{vinaitranslate,
    title     = {{A Vietnamese-English Neural Machine Translation System}},
    author    = {Thien Hai Nguyen and 
                 Tuan-Duy H. Nguyen and 
                 Duy Phung and 
                 Duy Tran-Cong Nguyen and 
                 Hieu Minh Tran and 
                 Manh Luong and 
                 Tin Duy Vo and 
                 Hung Hai Bui and 
                 Dinh Phung and 
                 Dat Quoc Nguyen},
    booktitle = {Proceedings of the 23rd Annual Conference of the International Speech Communication Association: Show and Tell (INTERSPEECH)},
    year      = {2022}
    }
    
Please **CITE** our paper whenever the pre-trained models or the system are used to help produce published results or incorporated into other software.

- VinAI Translate system: [https://vinai-translate.vinai.io](https://vinai-translate.vinai.io/)

## Pre-trained models

Model | Max length | License | Note
---|---|---|---
[`vinai/vinai-translate-vi2en-v2`](https://huggingface.co/vinai/vinai-translate-vi2en-v2) | 1024 | GNU AGPL-3.0 | Further fine-tuning `vinai/vinai-translate-vi2en` on a combination with mTet-v2 data (with duplication removal)
[`vinai/vinai-translate-en2vi-v2`](https://huggingface.co/vinai/vinai-translate-en2vi-v2) | 1024 | GNU AGPL-3.0 | Further fine-tuning `vinai/vinai-translate-en2vi` on a combination with mTet-v2 data (with duplication removal)
[`vinai/vinai-translate-vi2en`](https://huggingface.co/vinai/vinai-translate-vi2en) | 1024 | GNU AGPL-3.0 | 448M parameters
[`vinai/vinai-translate-en2vi`](https://huggingface.co/vinai/vinai-translate-en2vi) | 1024 | GNU AGPL-3.0 | 448M parameters

- Users might also play with these models via the HuggingFace space "VinAI Translate" at: [https://huggingface.co/spaces/vinai/VinAI_Translate](https://huggingface.co/spaces/vinai/VinAI_Translate)

### Evaluation


<img width="500" alt="Screen Shot 2023-11-17 at 10 07 05 pm" src="https://github.com/VinAIResearch/VinAI_Translate/assets/2412555/ab968080-0502-467e-966c-38fb340ab327">

*: [Data leakage issue](https://github.com/vietai/mTet/issues/13) where ~10K training pairs in mTet-v2 appear in the PhoMT test set.

See [HERE](https://github.com/VinAIResearch/VinAI_Translate/issues/2#issuecomment-1784567623) for evaluation scripts of VinAI Translate and VietAI `envit5-translation`. 


## Using VinAI Translate in [`transformers`](https://github.com/huggingface/transformers)

### Vietnamese-to-English translation

#### GPU-based batch translation example

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en-v2", src_lang="vi_VN")
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2")
device_vi2en = torch.device("cuda")
model_vi2en.to(device_vi2en)


def translate_vi2en(vi_texts: str) -> str:
    input_ids = tokenizer_vi2en(vi_texts, padding=True, return_tensors="pt").to(device_vi2en)
    output_ids = model_vi2en.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    en_texts = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
    return en_texts

# The input may consist of multiple text sequences, with the number of text sequences in the input ranging from 1 up to 8, 16, 32, or even higher, depending on the GPU memory.
vi_texts = ["Cô cho biết: trước giờ tôi không đến phòng tập công cộng, mà tập cùng giáo viên Yoga riêng hoặc tự tập ở nhà.",
            "Khi tập thể dục trong không gian riêng tư, tôi thoải mái dễ chịu hơn.",
            "cô cho biết trước giờ tôi không đến phòng tập công cộng mà tập cùng giáo viên yoga riêng hoặc tự tập ở nhà khi tập thể dục trong không gian riêng tư tôi thoải mái dễ chịu hơn"]
print(translate_vi2en(vi_texts))
```

#### CPU-based sequence translation example

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en-v2", src_lang="vi_VN")
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2")

def translate_vi2en(vi_text: str) -> str:
    input_ids = tokenizer_vi2en(vi_text, return_tensors="pt").input_ids
    output_ids = model_vi2en.generate(
        input_ids,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    en_text = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
    en_text = " ".join(en_text)
    return en_text

vi_text = "Cô cho biết: trước giờ tôi không đến phòng tập công cộng, mà tập cùng giáo viên Yoga riêng hoặc tự tập ở nhà. Khi tập thể dục trong không gian riêng tư, tôi thoải mái dễ chịu hơn."
print(translate_vi2en(vi_text))

vi_text = "cô cho biết trước giờ tôi không đến phòng tập công cộng mà tập cùng giáo viên yoga riêng hoặc tự tập ở nhà khi tập thể dục trong không gian riêng tư tôi thoải mái dễ chịu hơn"
print(translate_vi2en(vi_text))
```

### English-to-Vietnamese translation

#### GPU-based batch translation example

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")
device_en2vi = torch.device("cuda")
model_en2vi.to(device_en2vi)


def translate_en2vi(en_texts: str) -> str:
    input_ids = tokenizer_en2vi(en_texts, padding=True, return_tensors="pt").to(device_en2vi)
    output_ids = model_en2vi.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    vi_texts = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    return vi_texts

# The input may consist of multiple text sequences, with the number of text sequences in the input ranging from 1 up to 8, 16, 32, or even higher, depending on the GPU memory.
en_texts = ["I haven't been to a public gym before.",
            "When I exercise in a private space, I feel more comfortable.",
            "i haven't been to a public gym before when i exercise in a private space i feel more comfortable"]
print(translate_en2vi(en_texts))
```

#### CPU-based sequence translation example

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")

def translate_en2vi(en_text: str) -> str:
    input_ids = tokenizer_en2vi(en_text, return_tensors="pt").input_ids
    output_ids = model_en2vi.generate(
        input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    vi_text = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    vi_text = " ".join(vi_text)
    return vi_text

en_text = "I haven't been to a public gym before. When I exercise in a private space, I feel more comfortable."
print(translate_en2vi(en_text))

en_text = "i haven't been to a public gym before when i exercise in a private space i feel more comfortable"
print(translate_en2vi(en_text))
```
