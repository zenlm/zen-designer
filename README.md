---
license: apache-2.0
language:
- en
tags:
- zen
- zen-lm
- vision-language
- multimodal
- moe
- ocr
- document-understanding
library_name: transformers
pipeline_tag: image-text-to-text
---

<p align="center">
  <img src="https://zenlm.org/logo.png" width="300"/>
</p>

<h1 align="center">Zen Designer</h1>

<p align="center">
  <strong>235B vision-language model by Zen LM — images, video, documents, and spatial reasoning</strong>
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/zenlm/zen-designer-235b-a22b-instruct">HuggingFace</a> &nbsp;|&nbsp;
  📖 <a href="https://zenlm.org">Docs</a> &nbsp;|&nbsp;
  💻 <a href="https://github.com/zenlm">GitHub</a>
</p>

---

## Introduction

**Zen Designer** is Zen LM's flagship vision-language model: 235B total parameters with 22B active via Mixture of Experts (MoE). It delivers comprehensive visual understanding — images, video, documents, charts, and GUIs — combined with state-of-the-art language capabilities.

Two variants are available:

- **instruct**: optimized for direct visual question answering and task completion
- **thinking**: extended chain-of-thought reasoning for complex visual analysis

## Model Family

| Model | Type | Context | Description |
|-------|------|---------|-------------|
| [zen-designer-235b-a22b-instruct](https://huggingface.co/zenlm/zen-designer-235b-a22b-instruct) | Instruct | 256K | Direct VLM tasks |
| [zen-designer-235b-a22b-thinking](https://huggingface.co/zenlm/zen-designer-235b-a22b-thinking) | Thinking | 256K | Extended reasoning |

## Model Specifications

| Attribute | Value |
|-----------|-------|
| Total Parameters | 235B |
| Active Parameters | 22B (MoE) |
| Architecture | Dense vision encoder + MoE language decoder |
| Context Window | 256K tokens (expandable to 1M) |
| Languages | 100+ (OCR in 32 scripts) |
| Video Support | Long-form video understanding |
| License | Apache 2.0 |

## Key Capabilities

### Visual Understanding
- Detailed image analysis and description
- Video understanding with temporal reasoning
- Document parsing: PDFs, invoices, forms, tables, charts
- OCR across 32 languages and writing systems

### Spatial and Structural Reasoning
- 2D and 3D spatial grounding
- Bounding box prediction
- GUI element recognition and navigation
- Code generation from UI screenshots (HTML/CSS/JS)

### Agentic Vision
- Web navigation with visual context
- GUI interaction and automation
- Computer use workflows
- Diagram comprehension (Draw.io, flowcharts, schematics)

## Quick Start

### Install

```bash
pip install transformers torch accelerate
```

### Image Understanding

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image

model_name = "zenlm/zen-designer-235b-a22b-instruct"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

image = Image.open("document.png")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Extract all text and data from this document in structured format."}
        ]
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=2048)

response = processor.decode(generated_ids[0], skip_special_tokens=True)
print(response)
```

### Video Understanding

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

model_name = "zenlm/zen-designer-235b-a22b-instruct"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "path/to/video.mp4", "fps": 2},
            {"type": "text", "text": "Describe what happens in this video, step by step."}
        ]
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=1024)

response = processor.decode(generated_ids[0], skip_special_tokens=True)
print(response)
```

### Extended Thinking (Complex Tasks)

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image

model_name = "zenlm/zen-designer-235b-a22b-thinking"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

image = Image.open("math_problem.png")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Solve this problem step by step, showing your reasoning."}
        ]
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=8192)

response = processor.decode(generated_ids[0], skip_special_tokens=True)
print(response)
```

## Performance Benchmarks

| Benchmark | Score | Category |
|-----------|-------|----------|
| DocVQA | State-of-the-art | Document understanding |
| ChartQA | State-of-the-art | Chart reasoning |
| MMBench | Leading | Multimodal understanding |
| OCRBench | Leading | OCR accuracy |
| Video-MME | Leading | Video reasoning |
| MMMU | Competitive | Multi-discipline QA |

## Hardware Requirements

| Setup | VRAM | Notes |
|-------|------|-------|
| Minimum (INT4) | 4x 24GB | Quantized inference |
| Recommended (BF16) | 4x 80GB | Full precision |
| Optimal | 8x 80GB | Maximum throughput |

## Deployment

```bash
# vLLM (recommended for production)
vllm serve zenlm/zen-designer-235b-a22b-instruct \
    --tensor-parallel-size 4 \
    --max-model-len 65536

# SGLang
python -m sglang.launch_server \
    --model-path zenlm/zen-designer-235b-a22b-instruct \
    --tp-size 4
```

## License

Apache 2.0

## Citation

```bibtex
@misc{zenlm2025zen-designer,
    title={Zen Designer: 235B Vision-Language Model by Zen LM},
    author={Hanzo AI and Zoo Labs Foundation},
    year={2025},
    publisher={HuggingFace},
    howpublished={\url{https://huggingface.co/zenlm/zen-designer-235b-a22b-instruct}}
}
```

---

<p align="center">
  <strong>Zen LM by Hanzo AI</strong> - Clarity Through Intelligence<br>
  <a href="https://zenlm.org">zenlm.org</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/zenlm">HuggingFace</a> &nbsp;|&nbsp;
  <a href="https://github.com/zenlm">GitHub</a>
</p>
