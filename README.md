---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: TurkuNLP/gpt3-finnish-small
tags:
- gguf
- llama.cpp
- text-generation
- finnish
- suomi
- bloom
- 186m
- small-model
- quantization
- q4_k_m
- q8_0
---

# GPT3 Finnish Small GGUF

<div align="center">

<img alt="Model" src="https://img.shields.io/badge/model-gpt3--finnish--small-8A2BE2?style=for-the-badge">
<img alt="GGUF formats" src="https://img.shields.io/badge/GGUF-F16%20%7C%20Q8_0%20%7C%20Q4_K_M-FFD21E?style=for-the-badge">
<img alt="Architecture" src="https://img.shields.io/badge/arch-BLOOM-00A6A6?style=for-the-badge">
<img alt="Parameters" src="https://img.shields.io/badge/params-186M-16A34A?style=for-the-badge">
<img alt="Language" src="https://img.shields.io/badge/language-Finnish-0069B4?style=for-the-badge">
<img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-7C3AED?style=for-the-badge">

</div>

[Source model](https://huggingface.co/TurkuNLP/gpt3-finnish-small) · [HF release](https://huggingface.co/ShayonSarker/gpt3-finnish-small-GGUF) · [Build hub](https://github.com/Dadhichi-Sarker-Shayon/gpt3-finnish-small-GGUF)

Native llama.cpp GGUF release for the 186M-parameter Finnish BLOOM base model. The source config omits `seq_length`; the build restores the tokenizer's 2,048-token limit before conversion.

## Formats

| File | Status | Purpose |
|---|---|---|
| `gpt3-finnish-small-F16.gguf` | Published | Reference quality |
| `gpt3-finnish-small-Q8_0.gguf` | Published | Higher-quality compact format |
| `gpt3-finnish-small-Q4_K_M.gguf` | Published | Smallest release format |

## Verified outputs — read before using

Verbatim `gpt3-finnish-small-Q4_K_M.gguf` completions, `--temp 0`, 24 new tokens, prompt form `Kysymys: ...\nVastaus:`. All eight rows are published, including the wrong ones.

**Correct answers**

| Kysymys | Model answer |
|---|---|
| Mikä on Suomen pääkaupunki? | `Helsinki on Suomen pääkaupunki.` (then repeats the same sentence to fill the token budget) |
| Mikä on Ranskan pääkaupunki? | `Ranskan pääkaupunki on Pariisi.` |
| Onko Helsinki Pohjoismaiden pääkaupunki? | `Helsinki on Pohjoismaiden pääkaupunki.` |

**Wrong or degenerate answers**

| Kysymys | Model answer |
|---|---|
| Kuinka monta päivää viikossa on? | `Jos olet raskaana, sinun on oltava vähintään kaksi viikkoa viikossa.` |
| Mikä planeetta on lähinnä aurinkoa? | `Aurinko on aurinko, mutta se on myös planeetta.` |
| Mikä on veden kemiallinen merkki? | `Veden kemiallinen merkki on vesi.` |
| Mikä on maailman suurin valtameri? | `Maailman suurin valtameri on maailman suurin valtameri.` |
| Mikä on Suomen kieli? | `Suomen kieli on yksi maailman vanhimmista kielistä.` (no answer to the question) |

**3 of 8 factual questions correct, and every completion repeats itself.** This is a 186M base model, so it is useful for Finnish perplexity baselines, tokenizer and BLOOM-architecture GGUF testing, and continued-pretraining experiments. It is not usable as a question-answering model, and quantisation is not the limiting factor here: F16 fails the same way.

## Validation

Finnish Wikipedia evaluation, 8 chunks of 512 tokens. Lower perplexity is better.

| Format | PPL | Ratio to F16 |
|---|---:|---:|
| F16 | 44.4931 | Baseline |
| Q8_0 | 44.5125 | 1.0004 |
| Q4_K_M | 45.2245 | 1.0164 |

This is a conversion release, not an instruction or chat model.

## Build

```bash
python -m pip install -r requirements-build.txt
python build_gguf.py --model-id TurkuNLP/gpt3-finnish-small
```

The builder pins llama.cpp commit `6b790a9c291b5d7af3312bbf9f0c558aa023b13e` and the upstream model revision. It does not upload or overwrite this repository.

## License

Apache-2.0. See the upstream model card and `LICENSE` for source attribution.
