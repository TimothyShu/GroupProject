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
## Running All Examples

Each example dataset has its own folder under `example_*`. To run an example, activate your virtual environment and execute the main script in the desired example folder. For example:

```bash
# Activate your environment first
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate    # Windows

# Run a dataset
python example_adult/main.py
python example_california_housing/main.py
python example_colleges/main.py
python example_credit/main.py
python example_diamonds/main.py
python example_speeddating/main.py
```

There are also more you can find in the repo, just tun the files

You can modify or add your own datasets by following the structure of these example folders.
