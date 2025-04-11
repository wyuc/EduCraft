# LecGen - 讲稿生成工具

## 配置说明

要使用本项目，您需要配置相应模型的API密钥。请按照以下步骤操作：

1. 复制项目根目录中的 `config.py.template` 文件并重命名为 `config.py`
2. 打开 `config.py` 并填入您的API密钥：
   ```python
   MODEL_CONFIGS = {
       # Cloud API models
       "claude": {
           "base_url": "https://api.anthropic.com",
           "api_key": "your-anthropic-api-key",  # 添加您的Anthropic API密钥
           "default_model": "claude-3-sonnet-20240229"
       },
       
       "gemini": {
           "base_url": "https://generativelanguage.googleapis.com",
           "api_key": "your-google-api-key",  # 添加您的Google API密钥
           "default_model": "gemini-2.5-pro"
       },
       
       "gpt": {
           "base_url": "https://api.openai.com/v1",
           "api_key": "your-openai-api-key",  # 添加您的OpenAI API密钥
           "default_model": "gpt-4o"
       },
   }
   ```
3. 根据需要调整其他配置参数，如base_url和default_model

注意：`config.py` 已添加到 `.gitignore` 中，确保您的API密钥不会被提交到代码仓库。

## 支持的模型

- **Claude (Anthropic)**
  - 需要 Anthropic API 密钥
  - 支持 claude-3-opus, claude-3-sonnet, claude-3-haiku 等系列

- **Gemini (Google)**
  - 需要 Google AI Studio API 密钥
  - 支持 gemini-pro, gemini-2.5-pro 等系列

- **GPT (OpenAI)**
  - 需要 OpenAI API 密钥
  - 支持 gpt-4o, gpt-4-turbo 等系列

- **Ollama (本地模型)**
  - 默认连接到 http://127.0.0.1:11434
  - 支持 llama2, mistral, vicuna 等开源模型
  
- **vLLM (本地模型)**
  - 默认连接到 http://0.0.0.0:8000/v1
  - 支持部署的开源模型

## 安装与使用

[此处为项目的安装和使用说明] 