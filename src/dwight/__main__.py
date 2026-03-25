"""Entry point: ``python -m aiml`` starts the server; ``python -m aiml train`` trains."""

from __future__ import annotations

import click
from dotenv import load_dotenv

load_dotenv()  # Load .env from the working directory (no-op if absent); env vars take priority.

from dwight.training.dataset import DEFAULT_ARCHIVE


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """AIML – transformer LLM with an OpenAI-compatible API."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(serve)


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host.")
@click.option("--port", default=8000, show_default=True, type=int, help="Bind port.")
@click.option(
    "--reload", is_flag=True, default=False, help="Auto-reload on code changes."
)
@click.option(
    "--web-ui/--no-web-ui",
    default=False,
    help="Enable the Jinja2 web UI at / and /train.",
)
def serve(host: str, port: int, reload: bool, web_ui: bool) -> None:
    """Start the OpenAI-compatible HTTP server."""
    import os
    import uvicorn

    if web_ui:
        os.environ["DWIGHT_WEB_UI"] = "1"
    else:
        os.environ.pop("DWIGHT_WEB_UI", None)

    uvicorn.run(
        "dwight.server.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


@cli.command()
@click.argument("prompt")
@click.option(
    "--checkpoint",
    default="checkpoints/model.pt",
    show_default=True,
    help="Path to model weights.",
)
@click.option(
    "--max-tokens", default=100, show_default=True, type=int, help="Tokens to generate."
)
@click.option(
    "--temperature",
    default=1.0,
    show_default=True,
    type=float,
    help="Sampling temperature (0 = greedy).",
)
@click.option(
    "--top-p",
    default=0.9,
    show_default=True,
    type=float,
    help="Nucleus sampling probability.",
)
def predict(
    prompt: str, checkpoint: str, max_tokens: int, temperature: float, top_p: float
) -> None:
    """Feed-forward completion of PROMPT."""
    import os
    import torch
    from dwight.config import ModelConfig
    from dwight.model.transformer import GPTModel
    from dwight.tokenizer import TiktokenWrapper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig()
    tokenizer = TiktokenWrapper()
    model = GPTModel(config)

    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, weights_only=False, map_location=device)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        model.load_state_dict(state_dict)
    else:
        raise click.ClickException(f"Checkpoint not found: {checkpoint}")

    model.to(device)
    model.eval()

    click.echo(prompt, nl=False)
    eot = tokenizer.eot_token
    for tid in model.generate(
        tokenizer.encode(prompt),
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    ):
        if tid == eot:
            break
        click.echo(tokenizer.decode([tid]), nl=False)
    click.echo()


@cli.command()
@click.option(
    "--epochs", default=3, show_default=True, type=int, help="Training epochs."
)
@click.option(
    "--batch-size", default=8, show_default=True, type=int, help="Batch size."
)
@click.option(
    "--max-lr", default=3e-4, show_default=True, type=float, help="Peak learning rate."
)
@click.option(
    "--warmup-steps", default=100, show_default=True, type=int, help="LR warmup steps."
)
@click.option(
    "--checkpoint-dir",
    default="checkpoints",
    show_default=True,
    help="Directory to save model weights.",
)
@click.option(
    "--max-steps",
    default=None,
    type=int,
    help="Stop after N gradient steps (useful for quick tests).",
)
@click.option(
    "--data",
    default=DEFAULT_ARCHIVE,
    show_default=True,
    help="Path to the training archive (.tar.zst).",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume training from the last checkpoint in --checkpoint-dir.",
)
@click.option(
    "--grad-accum-steps",
    default=1,
    show_default=True,
    type=int,
    help="Accumulate gradients over N micro-batches before each optimizer step.",
)
def train(
    epochs: int,
    batch_size: int,
    max_lr: float,
    warmup_steps: int,
    checkpoint_dir: str,
    max_steps: int | None,
    data: str,
    resume: bool,
    grad_accum_steps: int,
) -> None:
    """Train the transformer on the 4chan /pol/ archive."""
    from dwight.training.train import train as _train

    _train(
        epochs=epochs,
        batch_size=batch_size,
        max_lr=max_lr,
        warmup_steps=warmup_steps,
        checkpoint_dir=checkpoint_dir,
        max_steps=max_steps,
        data=data,
        resume=resume,
        grad_accum_steps=grad_accum_steps,
    )


if __name__ == "__main__":
    cli()
