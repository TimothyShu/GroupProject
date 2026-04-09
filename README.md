# Group project

## Quickstart

From the project root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt (mac)
.\.venv\Scripts\python -m pip install -r requirements.txt (windows)
.venv/bin/python main.py
```

## Notes

- Use exactly: `.venv/bin/python -m pip install -r requirements.txt`
- CUDA/NVIDIA packages are only installed on Linux x86_64. On macOS, pip skips them automatically.

## Requirements

- Tested on Python 3.13
