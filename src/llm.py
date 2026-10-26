"""Minimal LLM wrapper around Qwen/Qwen3-0.6B."""

from typing import Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:
    """Generate answers from a query and retrieved contexts."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Load the tokenizer and model.

        Args:
            model_name: Hugging Face model identifier.
        """
        try:
            self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
            self.model: Any = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval()
        except (OSError, RuntimeError, ValueError) as e:
            raise SystemExit(
                f"Could not load language model: {e}"
            )

    def generate(self, query: str, contexts: List[str]) -> str:
        """Answer a query using the given contexts.

        Args:
            query: The question to answer.
            contexts: Retrieved text snippets.

        Returns:
            The generated answer.
        """
        try:
            context = "\n\n".join(contexts)
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Answer the question using only the context below.\n\n"
                        f"Context:\n{context}\n\nQuestion: {query}"
                    ),
                }
            ]
            prompt: str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                output = self.model.generate(
                    **inputs, max_new_tokens=256, do_sample=False
                )
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            answer: str = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )
            return answer.strip()
        except (IndexError, KeyError, RuntimeError, ValueError) as e:
            raise SystemExit(f"Answer generation failed: {e}")
