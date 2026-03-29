# dwight

A GPT-style causal language model built with PyTorch, served through a FastAPI server that is fully compatible with the OpenAI Python SDK.

## Features

- Transformer decoder (GPT architecture) with multi-head causal self-attention
- tiktoken tokenizer (`cl100k_base`, vocab size 100 277)
- Training on [TinyShakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) with cosine-decay learning rate
- OpenAI-compatible REST API — drop-in replacement for `openai.OpenAI(base_url=...)`
- Streaming (SSE) and non-streaming chat completions
- Click-based CLI with `serve`, `train`, and `predict` subcommands

## Requirements

- Python ≥ 3.12
- A virtual environment

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the package and remaining dependencies
pip install -e ".[dev]"
```

## CLI

The package exposes a `dwight` command (or `python -m dwight`).

### Start the server

```bash
python -m dwight                          # defaults: host=0.0.0.0, port=8000
python -m dwight serve --port 9000
python -m dwight serve --reload           # auto-reload on source changes
```

### Train the model

```bash
python -m dwight train                    # 3 epochs, batch size 8
python -m dwight train --epochs 10 --batch-size 16 --max-lr 1e-4
python -m dwight train --max-steps 50    # quick smoke-test (stops after 50 steps)
python -m dwight train --no-auto-stop    # disable rolling-loss auto-stop safeguard
```

Weights are saved to `checkpoints/model.pt` after each epoch. The server loads them automatically on startup if the file exists.

Training options:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 3 | Number of full passes over the dataset |
| `--batch-size` | 8 | Sequences per gradient step |
| `--max-lr` | 3e-4 | Peak learning rate (Adam) |
| `--warmup-steps` | 100 | Steps of linear LR warmup |
| `--checkpoint-dir` | `checkpoints` | Directory for saved weights |
| `--max-steps` | — | Hard stop after N gradient steps |
| `--auto-stop / --no-auto-stop` | `--auto-stop` | Halt and checkpoint when rolling loss regresses sharply or becomes non-finite |
| `--auto-stop-window` | 50 | Rolling window size for the auto-stop loss heuristic |
| `--auto-stop-ratio` | 1.6 | Relative loss increase over the best rolling window required to trigger auto-stop |
| `--auto-stop-patience` | 5 | Consecutive violating checks required before auto-stop halts training |
| `--auto-stop-min-steps` | 500 | Ignore the auto-stop heuristic until this many optimizer steps have completed |
| `--auto-stop-min-delta` | 0.75 | Minimum absolute rolling-loss increase required before auto-stop can fire |
| `--auto-stop-post-resume-steps` | 50 | Suppress auto-stop violation counting for N steps after restoring state on resume |

### Predict (feed-forward completion)

Run a one-shot text completion from the command line without starting the server:

```bash
python -m dwight predict "To be or not to be"
python -m dwight predict "Henceforth" --max-tokens 200 --temperature 0.8
python -m dwight predict "Once upon a time" --top-p 0.95 --temperature 0
```

The prompt is printed immediately, followed by the generated tokens streamed to stdout.

Predict options:

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/model.pt` | Path to model weights |
| `--max-tokens` | 100 | Number of new tokens to generate |
| `--temperature` | 1.0 | Sampling temperature (0 = greedy) |
| `--top-p` | 0.9 | Nucleus sampling probability mass |

## API

The server exposes an OpenAI-compatible API at `http://localhost:8000`.

### List models

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [{ "id": "dwight", "object": "model", "created": 1700000000, "owned_by": "user" }]
}
```

### Non-streaming chat completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dwight",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

### Streaming chat completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dwight",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true,
    "max_tokens": 64
  }'
```

### Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local",   # any non-empty string is accepted
)

# Non-streaming
response = client.chat.completions.create(
    model="dwight",
    messages=[{"role": "user", "content": "To be or not to be"}],
    max_tokens=80,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="dwight",
    messages=[{"role": "user", "content": "To be or not to be"}],
    max_tokens=80,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Request parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | required | Model identifier (use `"dwight"`) |
| `messages` | array | required | List of `{role, content}` objects |
| `temperature` | float | 1.0 | Sampling temperature (0 = greedy) |
| `top_p` | float | 1.0 | Nucleus sampling probability mass |
| `max_tokens` | int | 256 | Maximum tokens to generate |
| `stream` | bool | false | Enable SSE streaming |

## Architecture

### Model

`GPTModel` is a decoder-only transformer with learned positional embeddings:

```
token_embedding(x) + pos_embedding(positions)
  └─► Dropout
        └─► TransformerBlock × N
              ├─ LayerNorm → MultiHeadCausalAttention → residual
              └─ LayerNorm → FeedForwardNetwork      → residual
  └─► LayerNorm
  └─► Linear(vocab_size)  →  logits
```

**Default hyperparameters** (`ModelConfig`):

| Parameter | Value |
|---|---|
| `num_layers` | 6 |
| `d_model` | 256 |
| `num_heads` | 8 |
| `dff` | 1024 |
| `vocab_size` | 100 277 |
| `max_seq_len` | 512 |
| `dropout` | 0.1 |

### Attention

`MultiHeadCausalAttention` uses scaled dot-product attention with an additive upper-triangular causal mask (−10⁹ for future positions). Q, K, V are projected without bias; the output is projected back to `d_model`.

### Generation

Autoregressive sampling in `GPTModel.generate()` supports:

- **Greedy** (`temperature=0`)
- **Temperature sampling**
- **Nucleus (top-p) sampling** — keeps the smallest set of tokens whose cumulative probability exceeds `top_p`

The same generation backend is used by both the `predict` CLI subcommand and the HTTP server.

### Training

The training loop uses PyTorch with:

- **Loss**: cross-entropy from logits
- **Optimiser**: Adam with gradient clipping (`max_norm=1.0`)
- **LR schedule**: linear warmup for `warmup_steps` steps, then cosine decay to `1e-5`
- **Data**: TinyShakespeare (~1 MB), automatically downloaded and cached to `data/tinyshakespeare.txt`

### Server

The FastAPI application is created via a factory function (`create_app`) loaded by uvicorn with `factory=True`. A lifespan context manager loads model weights from `checkpoints/model.pt` at startup and stores the model and tokenizer in `app.state`. The chat prompt is formatted as:

```
System: <system message>
User: <user message>
Assistant: <assistant message>
...
Assistant:
```

## Training Data

Dwight is trained on a corpus of 3.5 years of data from 4chan's /pol/ board. You can see the dataset used [here](https://www.kaggle.com/datasets/harroopsra/3-5-years-of-4chan-politicallyincorrect-board-data)

## Tests

```bash
pytest                    # run all tests
pytest -q                 # quiet output
pytest tests/test_model.py  # single module
```

The test suite covers config validation, the tokenizer, all model layers, LR schedule, dataset creation, all API endpoints (both streaming and non-streaming), and schema validation. A session-scoped tiny model fixture (`num_layers=1`, `d_model=32`) keeps the suite fast.
