from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("This command requires torch in your local model environment.") from exc
    return torch


def _dtype_from_name(torch: Any, name: str):
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(name.lower(), torch.bfloat16)


class Qwen25VL:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        attn_implementation: str | None = "flash_attention_2",
    ) -> None:
        self.torch = _require_torch()
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = self.torch.device(device if device == "cpu" or self.torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "torch_dtype": _dtype_from_name(self.torch, torch_dtype),
            "trust_remote_code": True,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs).eval()
        self.model.to(self.device)

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def build_inputs(self, image_path: str | Path, prompt: str) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def forward_hidden(
        self,
        image_path: str | Path,
        prompt: str,
        pixel_delta=None,
        output_attentions: bool = False,
    ):
        inputs = self.build_inputs(image_path, prompt)
        if pixel_delta is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"] + pixel_delta.to(inputs["pixel_values"].device)
        with self.torch.set_grad_enabled(pixel_delta is not None):
            return self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=output_attentions,
                return_dict=True,
            )

    def pooled_hidden(self, outputs, pooling: str = "last_token") -> list[Any]:
        hidden_states = list(outputs.hidden_states)
        pooled = []
        for h in hidden_states:
            if pooling == "mean":
                pooled.append(h.mean(dim=1).squeeze(0))
            elif pooling == "first_token":
                pooled.append(h[:, 0, :].squeeze(0))
            else:
                pooled.append(h[:, -1, :].squeeze(0))
        return pooled

    def hidden_features(self, image_path: str | Path, prompt: str, pooling: str = "last_token") -> list[Any]:
        with self.torch.no_grad():
            outputs = self.forward_hidden(image_path, prompt)
            return [x.detach().float().cpu() for x in self.pooled_hidden(outputs, pooling)]

    def generate(self, image_path: str | Path, prompt: str, pixel_delta=None, **generate_kwargs: Any) -> str:
        inputs = self.build_inputs(image_path, prompt)
        if pixel_delta is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"] + pixel_delta.to(inputs["pixel_values"].device)
        defaults = {"max_new_tokens": 512, "do_sample": True, "temperature": 0.7, "top_p": 0.95}
        defaults.update(generate_kwargs)
        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, **defaults)
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

