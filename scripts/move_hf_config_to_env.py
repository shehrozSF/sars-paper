"""One-off: move hardcoded Hugging Face repo ids out of the notebooks into .env.

For every .ipynb in the repo this
  1. inserts an "env loader" cell at the top (reads .env into os.environ), and
  2. replaces quoted repo-id literals in code cells with os.environ["<KEY>"].
Outputs are left untouched. Safe to re-run (idempotent).

    python scripts/move_hf_config_to_env.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPO_KEYS = {
    "SotaSF/sar-to-optical-diffusion": "HF_REPO_ID",
    "SotaSF/pmflow-s2o-work": "HF_WORK_REPO",
    "AliMusaRizvi/sar-to-optical-diffusion": "HF_FALLBACK_REPO",
}

LOADER_MARK = "# [env] load .env"
LOADER_SRC = f'''{LOADER_MARK} (KEY=VALUE lines) into os.environ -- see .env.sample
# real env vars / Colab / Kaggle secrets take precedence; empty values are ignored
import os, pathlib
for _p in [pathlib.Path(".env"), pathlib.Path("../.env"),
           pathlib.Path("/content/drive/MyDrive/.env"),
           *sorted(pathlib.Path("/kaggle/input").glob("*/.env"))]:
    if _p.is_file():
        for _line in _p.read_text().splitlines():
            _k, _sep, _v = _line.partition("=")
            _k, _v = _k.strip(), _v.strip().strip("\\"'")
            if _sep and _k and _v and not _k.startswith("#"):
                os.environ.setdefault(_k, _v)
        print("env loaded from", _p)
        break
else:
    print("no .env found -- copy .env.sample to .env and fill it in")'''

HAS_IMPORT_OS = re.compile(r"^\s*(import\s+([\w.]+\s*,\s*)*os\b|from\s+os\b)", re.M)


def src_text(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def set_src(cell, text):
    if isinstance(cell.get("source"), list):
        cell["source"] = text.splitlines(keepends=True)
    else:
        cell["source"] = text


def ensure_import_os(text):
    if HAS_IMPORT_OS.search(text):
        return text
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith(("%%", "#!", "# -*-")):
        i += 1  # keep cell magics (%%writefile ...) / shebang on top
    lines.insert(i, "import os\n")
    return "".join(lines)


def process(path):
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)
    cells = nb["cells"]
    changed, leftovers = 0, []

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        text = src_text(cell)
        if text.lstrip().startswith(("%%bash", "%%sh", "%%script")):
            if any(r in text for r in REPO_KEYS):
                leftovers.append((idx, "shell cell"))
            continue
        new = text
        for repo, key in REPO_KEYS.items():
            for q in ('"', "'"):
                new = new.replace(f"{q}{repo}{q}", f'os.environ["{key}"]')
        if new != text:
            new = ensure_import_os(new)
            set_src(cell, new)
            changed += 1
        for repo in REPO_KEYS:
            if repo in new:
                leftovers.append((idx, repo))

    if not any(src_text(c).startswith(LOADER_MARK) for c in cells):
        loader = {"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": LOADER_SRC.splitlines(keepends=True)}
        if (nb.get("nbformat", 4), nb.get("nbformat_minor", 0)) >= (4, 5):
            loader = {"id": "env-loader", **loader}
        cells.insert(0, loader)
        changed += 1

    if changed:
        if raw.startswith("{\n"):
            out = json.dumps(nb, indent=1, ensure_ascii=False)
        else:
            out = json.dumps(nb, separators=(",", ":"), ensure_ascii=False)
        if raw.endswith("\n"):
            out += "\n"
        path.write_text(out, encoding="utf-8", newline="\n")
    return changed, leftovers


def main():
    nbs = sorted(p for p in ROOT.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)
    any_left = False
    for p in nbs:
        changed, left = process(p)
        print(f"{p.relative_to(ROOT)}: {changed} cell(s) updated")
        for idx, what in left:
            any_left = True
            print(f"   !! cell {idx}: still contains {what!r} -- fix by hand")
    return 1 if any_left else 0


if __name__ == "__main__":
    sys.exit(main())
