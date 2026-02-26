# Zen Designer

Vision-language model for design tasks from [Zen LM](https://zenlm.org).

Zen Designer is a multimodal model that understands and generates content from images, documents, and visual inputs. It excels at design-related workflows: layout analysis, visual QA, document parsing, GUI understanding, and image-conditioned generation.

## Model Variants

| Model | Parameters | Context | Specialization |
|-------|-----------|---------|---------------|
| **zen-designer-3b** | 3B | 256K | Lightweight, edge |
| **zen-designer-7b** | 7B | 256K | General purpose |
| **zen-designer-30b** | 31B MoE | 256K | Frontier design tasks |

## Key Capabilities

### Document Understanding
- Multi-language OCR (32+ languages)
- Table, chart, and formula extraction
- Handwriting recognition
- Multi-page document parsing

### Visual Grounding
- Precise object detection with absolute coordinates
- Element pointing, counting, and localization
- JSON-formatted bounding box output

### Video Understanding
- Long-form video analysis (hours-length content)
- Temporal event localization
- Dynamic frame rate processing

### Agent / GUI Tasks
- Desktop and mobile screen understanding
- UI element recognition and interaction planning
- ScreenSpot Pro benchmark support

## Quick Start

### Requirements

```
python>=3.9
transformers>=4.37.0
```

### Image Understanding

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests

model_id = "zenlm/zen-designer-7b"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto")

image = Image.open(requests.get("https://example.com/design.png", stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe the layout and design elements in this image."},
        ],
    }
]

inputs = processor.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

### Document Parsing

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": document_image},
            {"type": "text", "text": "Extract all tables from this document and format as JSON."},
        ],
    }
]
```

### Visual Grounding

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": ui_screenshot},
            {"type": "text", "text": "Locate the 'Submit' button. Return bounding box as JSON."},
        ],
    }
]
# Returns: {"bbox": [x1, y1, x2, y2], "label": "Submit button"}
```

### Video Understanding

```python
from zen_designer import load_video

frames = load_video("presentation.mp4", fps=1)
messages = [
    {
        "role": "user",
        "content": frames + [{"type": "text", "text": "Summarize the key design decisions shown."}],
    }
]
```

## Model Architecture

- Dual-path vision encoder: window attention for local features + global attention
- Dynamic resolution: adapts to input image dimensions
- Dynamic FPS sampling: variable-rate video frame processing
- Temporal position encoding for video understanding
- Optimized with SwiGLU activations and RMSNorm

## Performance Benchmarks

| Benchmark | zen-designer-3b | zen-designer-7b | zen-designer-30b |
|-----------|----------------|----------------|-----------------|
| MMMU | 53.1 | 58.6 | 70.2 |
| DocVQA | 93.9 | 95.7 | 96.4 |
| InfoVQA | 77.1 | 82.6 | 87.3 |
| MathVista | 62.3 | 68.2 | 74.8 |
| ScreenSpot | 55.5 | 84.7 | 87.1 |

## Quantized Formats

GGUF, AWQ, and GPTQ quantizations available on [HuggingFace](https://huggingface.co/zenlm).

## Links

- Models: [huggingface.co/zenlm](https://huggingface.co/zenlm)
- Agent framework: [github.com/zenlm/zen-agent](https://github.com/zenlm/zen-agent)
- Docs: [zenlm.org](https://zenlm.org)

## License

Apache 2.0 — Copyright 2024 Zen LM Authors