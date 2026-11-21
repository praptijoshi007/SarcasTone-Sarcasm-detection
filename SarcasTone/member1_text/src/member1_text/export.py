"""Export float32 CPU embeddings aligned with utterance ids.

RoBERTa embeddings go to ``embeddings.npz``. DeBERTa uses a different file.
MiniLM embeddings are written only by the baseline module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from member1_text import require_python_311
from member1_text.api import resolve_device
from member1_text.config import PACKAGE_ROOT, model_slug
from member1_text.model import load_span_model


def export_embeddings(
    checkpoint: str | Path,
    examples,
    output_path: str | Path | None = None,
    *,
    batch_size: int = 8,
    device_name: str | None = None,
) -> Path:
    from transformers import AutoTokenizer

    from member1_text.tokenize_encode import assert_tokenizer_matches_model
    from member1_text.train import ExampleDataset, collate_examples

    checkpoint_path = Path(checkpoint)
    device = resolve_device(device_name)
    model = load_span_model(checkpoint_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    assert_tokenizer_matches_model(model.model_name, tokenizer)
    model.eval()
    dataset = ExampleDataset(
        examples,
        tokenizer,
        model.max_length,
        model.model_name,
        label_map=model.label_map,
        task_type=model.task_type,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_examples,
    )
    ids: list[str] = []
    utterances = []
    contexts = []
    fused_rows = []
    probabilities = []
    has_context_rows = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                context_mask=batch["context_mask"].to(device),
                target_mask=batch["target_mask"].to(device),
            )
            ids.extend(batch["example_id"])
            utterances.append(output["utterance"].detach().float().cpu().numpy())
            contexts.append(output["context"].detach().float().cpu().numpy())
            fused_rows.append(output["fused"].detach().float().cpu().numpy())
            has_context_rows.append(
                output["has_context"].detach().view(-1).cpu().numpy().astype(np.bool_)
            )
            if output["logits"].size(-1) > 1:
                probs = torch.softmax(output["logits"].detach().float(), dim=-1)[:, 1]
            else:
                probs = output["logits"].detach().float().view(-1)
            probabilities.append(probs.cpu().numpy())
    utterance = np.concatenate(utterances).astype(np.float32)
    context = np.concatenate(contexts).astype(np.float32)
    fused = np.concatenate(fused_rows).astype(np.float32)
    if utterance.shape[0] != len(ids):
        raise RuntimeError("Embedding rows are not aligned with utterance ids.")
    slug = model_slug(model.model_name)
    if output_path is None:
        filename = "embeddings_deberta.npz" if "deberta" in slug else "embeddings.npz"
        output_path = PACKAGE_ROOT / "outputs" / "eval" / filename
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    id_array = np.asarray(ids, dtype=str)
    has_context = np.concatenate(has_context_rows).astype(np.bool_)
    np.savez(
        destination,
        # Member 3's documented cache contract.
        utterance_ids=id_array,
        embeddings=utterance,
        context_embeddings=context,
        has_context=has_context,
        # Backward-compatible names used by the earlier Member 1 export.
        ids=id_array,
        utterance=utterance,
        context=context,
        fused=fused,
        probability_sarcasm=np.concatenate(probabilities).astype(np.float32),
    )
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export checkpoint embeddings for other members.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    require_python_311()
    args = parse_args(argv)
    from member1_text.train import load_examples

    examples = load_examples(args.dataset, args.task)
    path = export_embeddings(args.checkpoint, examples, args.output, device_name=args.device)
    print(path)


if __name__ == "__main__":
    main()
