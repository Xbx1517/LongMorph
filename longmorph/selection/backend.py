"""Local greedy LegalJSON inference and source-bound actual-call receipts."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import time

from .protocol import LegalJSON, messages, parse, VERSION

MAX_NEW_TOKENS = 16


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Backend:
    """Load an explicitly supplied local model lazily; never download or substitute."""

    def __init__(self, model_path, device="cuda", input_token_limit=4096):
        self.path = str(Path(model_path).resolve())
        if not Path(self.path).is_dir():
            raise FileNotFoundError("A local instruction model directory is required: " + self.path)
        self.device = device
        self.input_token_limit = int(input_token_limit)
        self.model = None
        self.total_calls = 0

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.torch = torch
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False
        )
        dtype = torch.float16 if torch.device(self.device).type == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.path,
                local_files_only=True,
                trust_remote_code=False,
                dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
        )
        self.load_wall_seconds = time.perf_counter() - started

    def synchronize(self):
        if self.torch.device(self.device).type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def generate(self, role, payload):
        self.load()
        rendered = self.tokenizer.apply_chat_template(
            messages(role, payload), tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        n = int(inputs["input_ids"].shape[-1])
        if n + MAX_NEW_TOKENS > self.input_token_limit:
            raise ValueError("No silent evidence truncation")
        trie = LegalJSON(self.tokenizer, role, n, MAX_NEW_TOKENS)
        ids = inputs["input_ids"].detach().cpu().numpy()
        inputs = inputs.to(self.device)
        self.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                prefix_allowed_tokens_fn=trie,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self.synchronize()
        elapsed = time.perf_counter() - started
        raw = self.tokenizer.decode(out[0, n:], skip_special_tokens=True)
        value, error = parse(raw, role)
        self.total_calls += 1
        return dict(
            raw_response=raw,
            parsed=value,
            parse_error=error,
            parse_mode="constrained_exact_JSON",
            prompt_tokens=n,
            output_tokens=int(out.shape[-1] - n),
            generate_wall_seconds=elapsed,
            actual_input_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
            rendered_prompt=rendered,
            effective_input_text=self.tokenizer.decode(ids[0].tolist(), skip_special_tokens=False),
            API_latency_seconds=None,
            latency_kind="local model.generate; no remote API",
            model_path=self.path,
            device=self.device,
        )


class Journal:
    """One process owns a fresh output directory; failures never trigger retries."""

    def __init__(self, root, run_id, backend, source_paths=()):
        if backend is None:
            raise ValueError("An enabled Agent requires an explicit local backend")
        self.root = Path(root).resolve()
        self.run_id = str(run_id)
        self.backend = backend
        sources = sorted(set(Path(p).resolve() for p in source_paths))
        sources += sorted(Path(__file__).resolve().parent.glob("*.py"))
        self.bindings = [dict(path=str(p), sha256=sha(p)) for p in dict.fromkeys(sources)]
        self.source_stats = {
            str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in dict.fromkeys(sources)
        }
        self.source_sha = hashlib.sha256(
            json.dumps(self.bindings, sort_keys=True).encode()
        ).hexdigest()
        self.executed_calls = 0
        self.referenced_calls = 0

    def call(self, key, role, payload):
        messages(role, payload)
        for item in self.bindings:
            path = Path(item["path"])
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != self.source_stats[str(path)]:
                raise ValueError("Agent source metadata changed during run")
        request = dict(
            run_id=self.run_id,
            key=str(key),
            role=role,
            payload=payload,
            model_path=self.backend.path,
            schema_version=VERSION,
            source_bindings_sha256=self.source_sha,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            attention="eager",
            remote_API=False,
        )
        cid = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        path = self.root / "calls" / (cid + ".json")
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt["status"] != "COMPLETE" or receipt["request"] != request:
                raise RuntimeError("Incomplete or mismatched call preserved; no automatic retry")
            self.referenced_calls += 1
            return dict(
                receipt["result"],
                receipt_path=str(path),
                receipt_sha256=sha(path),
                executed_now=False,
            )
        receipt = dict(
            status="PENDING",
            call_id=cid,
            request=request,
            source_bindings=self.bindings,
            call_start_utc=now(),
        )
        write(path, receipt)
        started = time.perf_counter()
        try:
            result = self.backend.generate(role, payload)
            self.executed_calls += 1
            result.update(call_id=cid, run_id=self.run_id)
            receipt.update(
                status="COMPLETE",
                call_end_utc=now(),
                end_to_end_call_wall_seconds=time.perf_counter() - started,
                result=result,
            )
            write(path, receipt)
            return dict(result, receipt_path=str(path), receipt_sha256=sha(path), executed_now=True)
        except BaseException as error:
            receipt.update(
                status="FAILED",
                call_end_utc=now(),
                end_to_end_call_wall_seconds=time.perf_counter() - started,
                error=repr(error),
                cost_unknown=True,
            )
            write(path, receipt)
            raise
