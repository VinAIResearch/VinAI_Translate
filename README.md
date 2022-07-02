# A Vietnamese-English Neural Machine Translation System

Pre-trained VinAI Translate models `vinai/vinai-translate-vi2en` and `vinai/vinai-translate-en2vi` are state-of-the-art text translation models for Vietnamese-to-English and English-to-Vietnamese, respectively. Our demonstration system VinAI Translate employing these pre-trained models is available at: [https://vinai-translate.vinai.io](https://vinai-translate.vinai.io). 

The general architecture and experimental results of VinAI Translate can be found in our paper:


    @inproceedings{vinaitranslate,
    title     = {{A Vietnamese-English Neural Machine Translation System}},
    author    = {Thien Hai Nguyen and Tuan-Duy H. Nguyen and Duy Phung and Duy Tran-Cong Nguyen and Hieu Minh Tran and Manh Luong and Tin Duy Vo and Hung Hai Bui and Dinh Phung and Dat Quoc Nguyen},
    booktitle = {Proceedings of the 23rd Annual Conference of the International Speech Communication Association: Show and Tell (INTERSPEECH)},
    year      = {2022}
    }
    
Please **CITE** our paper whenever our pre-trained models are used to help produce published results or incorporated into other software.


## Using VinAI Translate in [`transformers`](https://github.com/huggingface/transformers)

### Installation

    pip install transformers sentencepiece tokenizers
    
### Pre-trained models

Model | #params | Max length  
---|---|---
`vinai/vinai-translate-vi2en` | 448M | 1024  
`vinai/vinai-translate-en2vi` | 448M | 1024  

### Vietnamese-to-English translation

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN")
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en")

def translate_vi2en(vi_text: str, max_output_length: int) -> str:
    if isinstance(vi_text, str):
        input_ids = tokenizer_vi2en(vi_text, return_tensors="pt").input_ids
    else:
        input_ids = tokenizer_vi2en(
            vi_text, return_tensors="pt", padding=True
        ).input_ids
    output_ids = model_vi2en.generate(
        input_ids,
        do_sample=True,
        max_length=max_output_length,
        top_k=100,
        top_p=0.8,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
    )
    en_text = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
    en_text = " ".join(en_text)
    return en_text

vi_text = "Cô cho biết: trước giờ tôi không đến phòng tập công cộng, mà tập cùng giáo viên Yoga riêng hoặc tự tập ở nhà. Khi tập thể dục trong không gian riêng tư, tôi thoải mái dễ chịu hơn."
max_output_length = 108
print(translate_vi2en(vi_text, max_output_length))

vi_text = "cô cho biết trước giờ tôi không đến phòng tập công cộng mà tập cùng giáo viên yoga riêng hoặc tự tập ở nhà khi tập thể dục trong không gian riêng tư tôi thoải mái dễ chịu hơn"
print(translate_vi2en(vi_text, max_output_length))
```

### English-to-Vietnamese translation

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi")

def translate_en2vi(en_text: str, max_output_length: int) -> str:
    if isinstance(en_text, str):
        input_ids = tokenizer_en2vi(en_text, return_tensors="pt").input_ids
    else:
        input_ids = tokenizer_en2vi(
            en_text, return_tensors="pt", padding=True
        ).input_ids
    output_ids = model_en2vi.generate(
        input_ids,
        do_sample=True,
        max_length=max_output_length,
        top_k=100,
        top_p=0.8,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
    )
    vi_text = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    vi_text = " ".join(vi_text)
    return vi_text

en_text = "I haven't been to a public gym before, but with a private yoga teacher or at home. When I exercise in a private space, I feel more comfortable."
max_output_length = 108
print(translate_en2vi(en_text, max_output_length))

en_text = "i haven't been to a public gym before but with a private yoga teacher or at home when i exercise in a private space i feel more comfortable"
print(translate_en2vi(en_text, max_output_length))
```
