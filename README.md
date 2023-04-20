# A Vietnamese-English Neural Machine Translation System

Our pre-trained VinAI Translate models `vinai/vinai-translate-vi2en` and `vinai/vinai-translate-en2vi` are state-of-the-art text-to-text translation models for Vietnamese-to-English and English-to-Vietnamese, respectively. These pre-trained models are currently used in the translation component of our [VinAI Translate system](https://vinai-translate.vinai.io). The general architecture and experimental results of the pre-trained models can be found in our [VinAI Translate system paper](https://openreview.net/forum?id=CRg-RaxKnai):

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


## Using VinAI Translate in [`transformers`](https://github.com/huggingface/transformers)

### Installation

    pip install transformers sentencepiece tokenizers
    
### Pre-trained models

Model | #params | Max length  
---|---|---
`vinai/vinai-translate-vi2en` | 448M | 1024  
`vinai/vinai-translate-en2vi` | 448M | 1024  

- Users might also play with these models via the VinAI Translate text-to-text translation app at: [https://huggingface.co/spaces/vinai/VinAI_Translate](https://huggingface.co/spaces/vinai/VinAI_Translate)

### Vietnamese-to-English translation

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN")
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en")

def translate_vi2en(vi_text: str) -> str:
    input_ids = tokenizer_vi2en(vi_text, return_tensors="pt").input_ids
    output_ids = model_vi2en.generate(
        input_ids,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        # # With sampling
        # do_sample=True,
        # top_k=100,
        # top_p=0.8,
        # With beam search
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

- **NOTE**: Before training, we performed Vietnamese tone normalization on the Vietnamese training data, using [a Python script](https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md). Users should also employ this script to pre-process the Vietnamese input data before feeding the data into our pre-trained model `vinai/vinai-translate-vi2en`. See a simple and complete example code at [HERE](https://huggingface.co/spaces/vinai/VinAI_Translate/blob/main/app.py).

### English-to-Vietnamese translation

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi")

def translate_en2vi(en_text: str) -> str:
    input_ids = tokenizer_en2vi(en_text, return_tensors="pt").input_ids
    output_ids = model_en2vi.generate(
        input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        # # With sampling
        # do_sample=True,
        # top_k=100,
        # top_p=0.8,
        # With beam search
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
