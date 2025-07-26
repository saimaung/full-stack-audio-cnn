## Environment Setup
```bash
uv venv -p 3.12.3 .venv
source .venv/bin/activate
uv pip install torch
uv pip install modal
uv pip install torchaudio
uv pip install pandas
uv pip install tqdm
uv pip install numpy
uv pip install tensorboard
uv pip install pydantic
uv pip install soundfile
uv pip install librosa fastapi
```

## Training the model on Serverless GPU Provide (Modal)
```bash
modal setup
modal run train.py
```