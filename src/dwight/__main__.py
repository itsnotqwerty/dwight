"""Entry point: ``python -m aiml`` starts the server; ``python -m aiml train`` trains."""

from __future__ import annotations

import click
from dotenv import load_dotenv
from typing import cast
from dwight.config import ModelConfig

load_dotenv()  # Load .env from the working directory (no-op if absent); env vars take priority.

from dwight.model.registry import MODEL_REGISTRY, load_model
from dwight.model.tiny import TinyModel
from dwight.model.transformer import GPTModel
from dwight.training.dataset import DEFAULT_ARCHIVE, DEFAULT_DPO, DEFAULT_PROMPTS


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
    "--model",
    "model_id",
    type=click.Choice(list(MODEL_REGISTRY)),
    default="dwight",
    show_default=True,
    help="Model architecture to use.",
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
    prompt: str,
    checkpoint: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    """Feed-forward completion of PROMPT."""
    import os
    import torch
    from dwight.tokenizer import TiktokenWrapper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = TiktokenWrapper()
    loaded_model, _, default_checkpoint = load_model(model_id, device)
    model = cast(GPTModel | TinyModel, loaded_model)
    checkpoint = checkpoint or default_checkpoint

    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, weights_only=False, map_location=device)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise click.ClickException(
                f"Checkpoint {checkpoint!r} is incompatible with the current model "
                f"architecture: {exc}. Delete it and retrain from scratch."
            ) from exc
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
    "--batch-size",
    default=None,
    type=int,
    help="Batch size. Defaults to the selected model's training config.",
)
@click.option(
    "--max-lr", default=3e-4, show_default=True, type=float, help="Peak learning rate."
)
@click.option(
    "--warmup-steps", default=1000, show_default=True, type=int, help="LR warmup steps."
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
    default=None,
    type=int,
    help="Accumulate gradients over N micro-batches before each optimizer step. Defaults to the selected model's training config.",
)
@click.option(
    "--auto-stop/--no-auto-stop",
    default=True,
    show_default=True,
    help="Checkpoint and halt if rolling loss regresses sharply or becomes non-finite.",
)
@click.option(
    "--auto-stop-window",
    default=50,
    show_default=True,
    type=int,
    help="Rolling-loss window size for the auto-stop heuristic.",
)
@click.option(
    "--auto-stop-ratio",
    default=1.6,
    show_default=True,
    type=float,
    help="Stop when rolling loss exceeds the best rolling window by at least this ratio.",
)
@click.option(
    "--auto-stop-patience",
    default=5,
    show_default=True,
    type=int,
    help="Number of consecutive violating checks before training halts.",
)
@click.option(
    "--auto-stop-min-steps",
    default=500,
    show_default=True,
    type=int,
    help="Do not evaluate auto-stop until at least this many optimizer steps have completed.",
)
@click.option(
    "--auto-stop-min-delta",
    default=0.75,
    show_default=True,
    type=float,
    help="Minimum absolute rolling-loss increase required before auto-stop can fire.",
)
@click.option(
    "--auto-stop-post-resume-steps",
    default=50,
    show_default=True,
    type=int,
    help="Suppress auto-stop violation counting for N steps after restoring state on resume.",
)
@click.option(
    "--model",
    "model_id",
    type=click.Choice(list(MODEL_REGISTRY)),
    default="dwight",
    show_default=True,
    help="Model architecture to train.",
)
def train(
    epochs: int,
    batch_size: int | None,
    max_lr: float,
    warmup_steps: int,
    checkpoint_dir: str,
    max_steps: int | None,
    data: str,
    resume: bool,
    grad_accum_steps: int | None,
    auto_stop: bool,
    auto_stop_window: int,
    auto_stop_ratio: float,
    auto_stop_patience: int,
    auto_stop_min_steps: int,
    auto_stop_min_delta: float,
    auto_stop_post_resume_steps: int,
    model_id: str,
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
        model_id=model_id,
        auto_stop=auto_stop,
        auto_stop_window=auto_stop_window,
        auto_stop_ratio=auto_stop_ratio,
        auto_stop_patience=auto_stop_patience,
        auto_stop_min_steps=auto_stop_min_steps,
        auto_stop_min_delta=auto_stop_min_delta,
        auto_stop_post_resume_steps=auto_stop_post_resume_steps,
    )


@cli.command("export")
@click.option(
    "--model",
    "model_id",
    type=click.Choice(list(MODEL_REGISTRY)),
    default="tiny",
    show_default=True,
    help="Model architecture to export.",
)
@click.option(
    "--checkpoint",
    default=None,
    help="Path to checkpoint (.pt).  Defaults to the model's registry checkpoint path.",
)
@click.option(
    "--output",
    default=None,
    help="Output artifact path (.lzma).  Defaults to the model's registry artifact_path.",
)
@click.option(
    "--group-size",
    default=512,
    show_default=True,
    type=int,
    help="Quantization group size (larger = smaller artifact, lower quality).",
)
def export(
    model_id: str,
    checkpoint: str | None,
    output: str | None,
    group_size: int,
) -> None:
    """Compress a model checkpoint into a small LZMA artifact for fast loading."""
    import os
    import torch
    from dwight.model.registry import get_model_entry
    from dwight.model.tiny.quantize import save_artifact

    entry = get_model_entry(model_id)

    checkpoint_path = checkpoint or entry.checkpoint_path
    output_path = output or entry.artifact_path
    if output_path is None:
        raise click.ClickException(
            f"Model {model_id!r} has no artifact_path in the registry; "
            "provide --output explicitly."
        )

    if not os.path.exists(checkpoint_path):
        raise click.ClickException(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cpu")
    config = entry.config_cls()
    model = entry.model_cls(config)

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state_dict, strict=False)

    click.echo(f"Exporting {model_id} → {output_path} (group_size={group_size}) …")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_artifact(model, output_path, group_size=group_size)
    size_mb = os.path.getsize(output_path) / 1_048_576
    click.echo(f"Done — {size_mb:.1f} MB")


@cli.command("generate-prompts")
@click.option(
    "--count",
    default=9000,
    show_default=True,
    type=int,
    help="Number of prompt-response examples to generate.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed.",
)
@click.option(
    "--output",
    default=DEFAULT_PROMPTS,
    show_default=True,
    help="Output Markdown file for generated prompts.",
)
def generate_prompts(count: int, seed: int, output: str) -> None:
    """Generate a synthetic prompt corpus for structured SFT."""
    from dwight.training.generate_prompts import (
        generate_prompt_examples,
        write_prompt_examples,
    )

    examples = generate_prompt_examples(count=count, seed=seed)
    path = write_prompt_examples(examples, output)
    click.echo(f"Wrote {len(examples)} prompt examples to {path}")


@cli.command("generate-dpo")
@click.option(
    "--count",
    default=9000,
    show_default=True,
    type=int,
    help="Number of chosen/rejected examples to generate.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed.",
)
@click.option(
    "--output",
    default=DEFAULT_DPO,
    show_default=True,
    help="Output Markdown file for generated DPO prompts.",
)
def generate_dpo(count: int, seed: int, output: str) -> None:
    """Generate a synthetic chosen/rejected corpus for DPO."""
    from dwight.training.generate_dpo_prompts import (
        generate_dpo_examples,
        write_dpo_examples,
    )

    examples = generate_dpo_examples(count=count, seed=seed)
    path = write_dpo_examples(examples, output)
    click.echo(f"Wrote {len(examples)} DPO examples to {path}")


@cli.command()
@click.option(
    "--dpo-path",
    default=DEFAULT_DPO,
    show_default=True,
    help="Path to the chosen/rejected DPO corpus.",
)
@click.option(
    "--epochs", default=1, show_default=True, type=int, help="Training epochs."
)
@click.option(
    "--batch-size",
    default=None,
    type=int,
    help="Batch size. Defaults to the selected model's training config.",
)
@click.option(
    "--lr", default=1e-5, show_default=True, type=float, help="Learning rate."
)
@click.option(
    "--beta",
    default=0.1,
    show_default=True,
    type=float,
    help="DPO beta value.",
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
    "--model",
    "model_id",
    type=click.Choice(list(MODEL_REGISTRY)),
    default="dwight",
    show_default=True,
    help="Model architecture to fine-tune with DPO.",
)
def dpo(
    dpo_path: str,
    epochs: int,
    batch_size: int | None,
    lr: float,
    beta: float,
    checkpoint_dir: str,
    max_steps: int | None,
    model_id: str,
) -> None:
    """Fine-tune a model on a chosen/rejected DPO corpus."""
    import os
    import torch
    from dwight.tokenizer import TiktokenWrapper
    from dwight.training.finetune import dpo_finetune

    if not os.path.exists(dpo_path):
        raise click.ClickException(f"DPO corpus not found: {dpo_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = TiktokenWrapper()
    loaded_model, config, _checkpoint_path = load_model(model_id, device)
    model = cast(GPTModel | TinyModel, loaded_model)

    dpo_finetune(
        model,
        tokenizer,
        cast(ModelConfig, config),
        dpo_path=dpo_path,
        epochs=epochs,
        batch_size=batch_size or getattr(config, "train_batch_size", 1),
        lr=lr,
        beta=beta,
        max_steps=max_steps,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    cli()
