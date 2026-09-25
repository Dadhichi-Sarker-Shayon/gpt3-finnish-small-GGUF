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
- bloom
---

# GPT3 Finnish Small GGUF

[Source model](https://huggingface.co/TurkuNLP/gpt3-finnish-small) · [Build hub](https://github.com/Dadhichi-Sarker-Shayon/gpt3-finnish-small-GGUF)

Native llama.cpp GGUF release for the 186M-parameter Finnish BLOOM base model. The source config omits `seq_length`; the build restores the tokenizer's 2,048-token limit before conversion.

## Formats

| File | Purpose |
|---|---|
| `gpt3-finnish-small-F16.gguf` | Reference quality |
| `gpt3-finnish-small-Q8_0.gguf` | Higher-quality compact format |
| `gpt3-finnish-small-Q4_K_M.gguf` | Smallest release format |

## Validation

Finnish Wikipedia evaluation, 8 chunks of 512 tokens. Lower perplexity is better.

| Format | PPL | Ratio to F16 |
|---|---:|---:|
| F16 | 44.4931 | Baseline |
| Q8_0 | 44.5125 | 1.0004 |
| Q4_K_M | 45.2245 | 1.0164 |

A deterministic Q4_K_M generation smoke test completed successfully. This is a conversion release, not an instruction or chat model.

## Build

```bash
python -m pip install -r requirements-build.txt
python build_gguf.py --model-id TurkuNLP/gpt3-finnish-small
```

The builder pins llama.cpp commit `6b790a9c291b5d7af3312bbf9f0c558aa023b13e` and the upstream model revision. It does not upload or overwrite this repository.

## License

Apache-2.0. See the upstream model card and `LICENSE` for source attribution.
