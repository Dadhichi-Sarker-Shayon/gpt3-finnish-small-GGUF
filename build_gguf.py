from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

LLAMA_COMMIT = "6b790a9c291b5d7af3312bbf9f0c558aa023b13e"
MODELS = {
    "TurkuNLP/gpt3-finnish-small": {
        "revision": "20a19af481bf59f38610a2977b2b513e9df51e3a",
        "name": "gpt3-finnish-small",
        "kind": "causal",
        "minimum_gb": 4,
        "prompt": "Helsinki on Suomen pääkaupunki. Suomen kieli on",
    },
    "HuggingFaceTB/SmolLM-360M": {
        "revision": "59f7ef243ee09a72cbc14cb054393a3e3b771d41",
        "name": "SmolLM-360M",
        "kind": "causal",
        "minimum_gb": 5,
        "prompt": "The capital of France is",
    },
    "HuggingFaceTB/SmolLM2-360M": {
        "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "name": "SmolLM2-360M",
        "kind": "causal",
        "minimum_gb": 5,
        "prompt": "The capital of France is",
    },
    "deepseek-ai/deepseek-coder-1.3b-base": {
        "revision": "c919139c3a9b4070729c8b2cca4847ab29ca8d94",
        "name": "DeepSeek-Coder-1.3B-Base",
        "kind": "causal",
        "minimum_gb": 12,
        "prompt": "def add(a, b):\n    return",
    },
    "bigcode/starcoder2-3b": {
        "revision": "733247c55e3f73af49ce8e9c7949bf14af205928",
        "name": "StarCoder2-3B",
        "kind": "causal",
        "minimum_gb": 18,
        "prompt": "def add(a, b):\n    return",
    },
    "NeuML/bioclinical-modernbert-base-embeddings": {
        "revision": "048ad4491de0fb4e2695bfe4705da67caf4804b8",
        "name": "BioClinical-ModernBERT-Embeddings",
        "kind": "embedding",
        "minimum_gb": 5,
        "prompt": "clinical evidence for a treatment",
    },
}
DOWNLOAD_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.bin",
    "*.txt",
    "*.model",
    "LICENSE*",
]
CALIBRATION = """XGLM is a multilingual language model developed for broad language understanding and generation. It was trained on text from many languages, including Bengali, English, French, German, Arabic, Russian, Chinese, Japanese, Spanish, Portuguese, Hindi, and Turkish.
The capital of France is Paris. The capital of Bangladesh is Dhaka. বাংলাদেশের রাজধানী ঢাকা।
Les systèmes d’apprentissage automatique utilisent des exemples pour apprendre des relations. Machine learning systems should be evaluated with data that was not used for training.
Los sistemas de aprendizaje automático aprenden patrones a partir de ejemplos. La evaluación debe utilizar datos que no se hayan usado durante el entrenamiento.
機械学習系统在测试时应该使用没有出现在训练数据中的样本。Разные языки должны сохранять качество модели.
def add(a, b):
    return a + b

SELECT name, price FROM products WHERE price < 100 ORDER BY price;
"""


def run(command, cwd=None, timeout=7200):
    text = [str(part) for part in command]
    print("+", " ".join(text), flush=True)
    result = subprocess.run(text, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    print(result.stdout, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(text)}")
    return result.stdout


def check_output(command, cwd=None):
    return subprocess.check_output([str(part) for part in command], cwd=cwd, text=True).strip()


def find_binary(build_dir, name):
    for path in (build_dir / "bin" / name, build_dir / "bin" / "Release" / f"{name}.exe", build_dir / f"{name}.exe"):
        if path.is_file():
            return path
    raise FileNotFoundError(name)


def ensure_llama(work_dir):
    llama_dir = work_dir / "llama.cpp"
    if not (llama_dir / ".git").is_dir():
        llama_dir.mkdir(parents=True, exist_ok=True)
        run(["git", "init"], llama_dir)
    remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=llama_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if remote.returncode:
        run(["git", "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp.git"], llama_dir)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=llama_dir, text=True, capture_output=True)
    if head.returncode or head.stdout.strip() != LLAMA_COMMIT:
        if check_output(["git", "status", "--porcelain"], llama_dir):
            raise RuntimeError("llama.cpp work directory has unrelated changes")
        run(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], llama_dir)
        run(["git", "checkout", "--detach", LLAMA_COMMIT], llama_dir)
    return llama_dir


def prepare_base(model_id, base):
    from huggingface_hub import snapshot_download
    snapshot_download(model_id, revision=MODELS[model_id]["revision"], local_dir=str(base), allow_patterns=DOWNLOAD_PATTERNS)
    config_path = base / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if model_id == "TurkuNLP/gpt3-finnish-small":
        config["seq_length"] = 2048
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def convert(model_id, work_dir, threads, dry_run):
    config = MODELS[model_id]
    name = config["name"]
    base = work_dir / model_id.split("/")[-1]
    output = work_dir / "output"
    f16 = output / f"{name}-F16.gguf"
    q8 = output / f"{name}-Q8_0.gguf"
    q4 = output / f"{name}-Q4_K_M.gguf"
    if dry_run:
        print(json.dumps({"model_id": model_id, "revision": config["revision"], "minimum_free_gb": config["minimum_gb"], "outputs": [str(f16), str(q8), str(q4)]}, indent=2))
        return
    if not shutil.which("git") or not shutil.which("cmake"):
        raise RuntimeError("git and cmake are required")
    work_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(work_dir).free / 1_000_000_000
    if free_gb < config["minimum_gb"]:
        raise RuntimeError(f"At least {config['minimum_gb']} GB free space is required; found {free_gb:.1f} GB")
    prepare_base(model_id, base)
    llama_dir = ensure_llama(work_dir)
    build_dir = llama_dir / "build"
    run(["cmake", "-S", llama_dir, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF"])
    targets = ["llama-quantize", "llama-perplexity", "llama-imatrix"]
    if config["kind"] == "causal":
        targets.append("llama-completion")
    else:
        targets.append("llama-embedding")
    run(["cmake", "--build", build_dir, "--config", "Release", "--target", *targets, "--parallel", str(threads)])
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "metadata.json"
    metadata.write_text(json.dumps({"general.name": name, "general.author": "upstream model authors", "general.license.name": "see upstream model license", "general.source.url": f"https://huggingface.co/{model_id}"}, indent=2), encoding="utf-8")
    converter = llama_dir / "convert_hf_to_gguf.py"
    run([sys.executable, converter, base, "--outfile", f16, "--outtype", "f16", "--metadata", metadata])
    quantizer = find_binary(build_dir, "llama-quantize")
    if config["kind"] == "causal":
        calibration = output / "calibration.txt"
        calibration.write_text(CALIBRATION, encoding="utf-8")
        imatrix = output / "calibration.imatrix"
        imatrix_bin = find_binary(build_dir, "llama-imatrix")
        run([imatrix_bin, "-m", f16, "-f", calibration, "-c", "128", "-b", "64", "--chunks", "2", "-ngl", "0", "-t", str(threads), "--process-output", "-o", imatrix])
        run([quantizer, f16, q8, "Q8_0", str(threads)])
        run([quantizer, "--imatrix", imatrix, f16, q4, "Q4_K_M", str(threads)])
        completion = find_binary(build_dir, "llama-completion")
        run([completion, "-m", q4, "-p", config["prompt"], "-n", "16", "--temp", "0", "-ngl", "0", "-t", str(threads), "-c", "256", "--no-display-prompt"])
    else:
        run([quantizer, f16, q8, "Q8_0", str(threads)])
        run([quantizer, f16, q4, "Q4_K_M", str(threads)])
        embedding = find_binary(build_dir, "llama-embedding")
        run([embedding, "-m", f16, "--pooling", "mean", "--embd-output-format", "json", "-p", config["prompt"], "-c", "512", "--no-warmup"])
    for path in (f16, q8, q4):
        if not path.is_file():
            raise RuntimeError(f"Missing GGUF: {path}")
        with path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise RuntimeError(f"Invalid GGUF: {path}")
    print(f"Completed: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=sorted(MODELS), default="HuggingFaceTB/SmolLM-360M")
    parser.add_argument("--work-dir", type=Path, default=Path("build-gguf"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    convert(args.model_id, args.work_dir.resolve(), args.threads, args.dry_run)


if __name__ == "__main__":
    main()
