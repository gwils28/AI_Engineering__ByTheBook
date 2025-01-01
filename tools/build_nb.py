"""Construit et exécute les notebooks à partir de leurs sources jupytext (format « percent »).

Usage : python tools/build_nb.py .nbsrc/ch03/ch03_cours_1_xxx.py [autres sources...]
Le notebook est écrit dans notebooks/<dossier du chapitre>/ (indiqué par la ligne
`# target: notebooks/ch03_xxx/` en tête de la source), puis exécuté. Toute erreur
d'exécution fait échouer la construction.
"""
import sys
import time
from pathlib import Path

import jupytext
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]


def build(src: Path, timeout: int = 1800) -> Path:
    text = src.read_text(encoding="utf-8")
    target = None
    for line in text.splitlines()[:5]:
        if line.startswith("# target:"):
            target = line.split(":", 1)[1].strip()
    if target is None:
        raise SystemExit(f"{src}: ligne '# target: notebooks/chXX_.../' manquante")
    text = "\n".join(l for l in text.splitlines() if not l.startswith("# target:"))
    nb = jupytext.reads(text, fmt="py:percent")
    nb.metadata["kernelspec"] = {"name": "aie", "display_name": "Python (aie)", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata.pop("jupytext", None)
    out = ROOT / target / (src.stem + ".ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="aie")
    ep.preprocess(nb, {"metadata": {"path": str(out.parent)}})
    for cell in nb.cells:
        cell.metadata.pop("execution", None)
    nbformat.write(nb, out)
    print(f"OK  {out.relative_to(ROOT)}  ({time.time() - t:.0f} s)")
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        build(Path(arg))
