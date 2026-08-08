# Limitations

This project implements a controlled research pipeline; it has not established
that anti-sycophancy training generalizes to unrestricted conversation or is
safe for production deployment.

- Training pushback is partly synthetic and can contain template or lexical
  bias even though generation is varied and deterministic.
- Evaluation uses strict A/B parsing. It does not measure open-ended
  factuality, calibration, or general conversational quality.
- Tone detection and explanation relevance are lightweight heuristics. They
  can miss paraphrases and context that a human evaluator would recognize.
- The Anthropic benchmark represents a limited family of sycophancy prompts;
  improvement there is not evidence for every domain or pressure style.
- Exact and lexical leakage checks run by default. Optional semantic checking
  depends on the supplied similarity implementation and threshold and cannot
  prove that all contamination is absent.
- Real 7B QLoRA/GRPO execution requires compatible GPU memory, drivers, model
  access, storage, and substantial runtime. Unit tests do not emulate that
  native stack.
- Reducing agreement with invalid pressure can increase unnecessary
  disagreement or resistance to valid corrections. Both behaviors must be
  measured alongside target accuracy and capability regression.
- Training can regress unrelated capabilities. A completed before/after run,
  paired statistics, error review, and multiple seeds are required before a
  broad effectiveness claim.
- The local API and containers isolate project dependencies and filesystem
  access for trusted research code; they are not sandboxes for hostile code or
  internet-facing multi-tenant services.

Implementation status is documented in `industrial_readiness.md`. Model
quality is experimentally demonstrated only when retained run artifacts—not
unit tests, dry runs, or screenshots—support the claim.
