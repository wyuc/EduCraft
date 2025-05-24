# EduCraft: Automated Lecture Script Generation from Multimodal Presentations

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/xxxx.xxxxx) -->

**EduCraft** is a novel system designed to automate Lecture Script Generation (LSG) from multimodal presentations, addressing key challenges in educational content creation including comprehensive multimodal understanding, long-context coherence, and instructional design efficacy.

## 🎯 Overview

Educators face substantial workload pressures, with significant time invested in preparing teaching materials. EduCraft tackles the demanding task of generating high-quality lecture scripts from slides and presentations, offering a practical solution to reduce educator workload and enhance educational content creation.

### Key Features

- **🎨 Multimodal Processing**: Robust extraction and association of text, images, and visual elements from slides
- **🧠 Dual Workflows**: Support for both VLM (Vision-Language Model) and Caption+LLM approaches
- **📚 RAG Integration**: Optional Retrieval-Augmented Generation for enhanced factual grounding
- **🔧 Flexible Model Support**: Compatible with Claude, GPT, Gemini, Ollama, and vLLM models
- **📊 Comprehensive Evaluation**: Validated through human assessments and automated metrics

### Architecture

EduCraft features a modular architecture comprising:

1. **Multimodal Input Processing Pipeline**: Robust data extraction and association from slides
2. **Lecture Script Generation Engine**: Core generation with VLM and Caption+LLM workflows
3. **Knowledge Augmentation Module**: Optional RAG for enhanced factual grounding  
4. **Model Integration Interface**: Support for diverse AI models with deployable API

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for local models)
- API keys for cloud models (Claude, GPT, Gemini) or local model setup

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/wyuc/EduCraft.git
cd EduCraft
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure API keys:**
```bash
cp config.py.template config.py
```

Edit `config.py` and add your API keys:
```python
MODEL_CONFIGS = {
    "claude": {
        "base_url": "https://api.anthropic.com",
        "api_key": "your-anthropic-api-key",
        "default_model": "claude-3-sonnet-20240229"
    },
    "gpt": {
        "base_url": "https://api.openai.com/v1", 
        "api_key": "your-openai-api-key",
        "default_model": "gpt-4o"
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com",
        "api_key": "your-google-api-key",
        "default_model": "gemini-2.5-pro"
    }
}
```

## 💡 Quick Start

### Basic Usage

Generate lecture scripts from a PowerPoint presentation:

```bash
python -m algo.main \
    --input presentation.pptx \
    --algorithm vlm \
    --model_provider claude \
    --temperature 0.7
```

### Advanced Usage with RAG

```bash
python -m algo.main \
    --input presentation.pptx \
    --algorithm vlm \
    --model_provider gpt \
    --use_rag \
    --kb_path /path/to/knowledge_base \
    --export_excel
```

### Caption+LLM Workflow

```bash
python -m algo.main \
    --input presentation.pptx \
    --algorithm caption_llm \
    --model_provider gpt \
    --caption_model_provider claude \
    --temperature 0.7
```

### Python API

```python
from algo.main import process_ppt

# Basic VLM workflow
result = process_ppt(
    input_path="presentation.pptx",
    algorithm="vlm", 
    model_params={
        "model_provider": "claude",
        "model_name": "claude-3-sonnet-20240229",
        "max_tokens": 32768
    }
)

# With RAG enhancement
result = process_ppt(
    input_path="presentation.pptx",
    algorithm="vlm",
    model_params={
        "model_provider": "gpt", 
        "model_name": "gpt-4o",
        "max_tokens": 32768,
        "use_rag": True,
        "kb_path": "/path/to/knowledge_base"
    }
)
```

## 🏗️ System Architecture

### Workflows

**VLM Workflow**: Direct processing of slide images with associated text using vision-language models for holistic understanding and script generation.

**Caption+LLM Workflow**: Two-stage approach using specialized vision models for captioning followed by LLMs for narrative synthesis.

### Supported Models

| Provider | Models | Type |
|----------|--------|------|
| **Claude** | claude-3-opus, claude-3-sonnet, claude-3-haiku | Cloud API |
| **GPT** | gpt-4o, gpt-4-turbo, gpt-4-vision | Cloud API |  
| **Gemini** | gemini-pro, gemini-2.5-pro | Cloud API |
| **Ollama** | llama2, mistral, vicuna, etc. | Local |
| **vLLM** | Various open-source models | Local |

## 📊 Evaluation Results

### Human Evaluation (20 university presentations)

EduCraft significantly outperforms baseline methods across key quality dimensions:

| Method | Consistency | Readability | Coherence | Overall |
|--------|-------------|-------------|-----------|---------|
| Iterative Baseline | 1.51 | 1.48 | 1.31 | 1.47 |
| Teacher Refined | 2.04 | 2.16 | 2.25 | 2.12 |
| **EduCraft** | **2.44** | **2.37** | **2.44** | **2.41** |

### Automated Evaluation

EduCraft VLM workflow achieves superior performance on comprehensive metrics:

| Model | Content Relevance | Expressive Clarity | Logical Structure | Combined Score |
|-------|------------------|-------------------|------------------|----------------|
| GPT-4o + EduCraft | **4.16** | **4.22** | **4.11** | **3.86** |
| GPT-4o + Direct Prompt | 4.11 | 4.20 | 4.05 | 3.78 |
| GPT-4o + Iterative | 4.13 | 4.16 | 4.06 | 3.70 |

## 🔧 Configuration Options

### Algorithm Parameters

- `--algorithm`: Choose from `vlm`, `caption_llm`, `iterative`, `direct_prompt`
- `--temperature`: Control randomness (0.0-2.0, default: 0.7)
- `--max_tokens`: Maximum tokens to generate
- `--prompt_variant`: Prompt variation (`full`, `no_narrative`, etc.)

### RAG Configuration

- `--use_rag`: Enable RAG integration
- `--kb_path`: Path to knowledge base directory
- `--embedding_model`: Embedding model for retrieval
- `--top_k`: Number of retrieved passages (default: 5)

### Model Selection

- `--model_provider`: Model provider (`claude`, `gpt`, `gemini`, `ollama`, `vllm`)
- `--model_name`: Specific model name (optional)
- `--caption_model_provider`: Caption model provider (for Caption+LLM)

## 📁 Project Structure

```
EduCraft/
├── algo/                 # Core algorithms
│   ├── main.py          # Main entry point
│   ├── vlm.py           # VLM workflow
│   ├── caption_llm.py   # Caption+LLM workflow
│   ├── iterative.py     # Iterative baseline
│   └── prompts/         # Prompt templates
├── models/              # Model interfaces
├── utils/               # Utility functions
├── eval/                # Evaluation scripts
├── config.py.template   # Configuration template
└── requirements.txt     # Dependencies
```

## 📄 Dataset and Evaluation

Our evaluation uses diverse university-level presentations from multiple disciplines:

- **Human Evaluation**: 20 presentations (320 slides) across 4 university courses
- **Automated Evaluation**: 20 presentations (272 slides) in English and Chinese
- **Domains**: Humanities, Social Sciences, STEM, Applied Fields

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and submit pull requests for:

- Bug fixes and improvements
- New model integrations
- Additional evaluation metrics
- Documentation enhancements

## 📖 Citation

If you use EduCraft in your research, please cite:

```bibtex
@article{educraft2024,
  title={EduCraft: Automated Lecture Script Generation from Multimodal Presentations},
  author={[Authors]},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on foundations from MAIC platform
- Evaluation methodology inspired by LecEval
- Special thanks to all annotators and educators who participated in our evaluation

## 📞 Contact

For questions or collaboration opportunities, please contact:
- [Primary Author Email]
- [Project Issues](https://github.com/wyuc/EduCraft/issues)

---

**Note**: This is an open-source implementation of EduCraft. For the latest updates and detailed documentation, please visit our [GitHub repository](https://github.com/wyuc/EduCraft). 