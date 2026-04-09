# Group project

## Quickstart

From the project root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt (mac)
.\.venv\Scripts\python -m pip install -r requirements.txt (windows)
.venv/bin/python main.py
```

To run TabPFN
1. Go to https://docs.priorlabs.ai/api-reference/getting-started#1-get-your-access-token to get your access token.
2. Create a `.env` file in the project root with:
```dotenv
TABPFN_TOKEN=YOUR_TOKEN_HERE
```
3. (Optional fallback for current PowerShell session only)
```powershell
$env:TABPFN_TOKEN = "YOUR_TOKEN_HERE"
```

## Notes

- Use exactly: `.venv/bin/python -m pip install -r requirements.txt`
- CUDA/NVIDIA packages are only installed on Linux x86_64. On macOS, pip skips them automatically.

## Requirements

- Tested on Python 3.13
