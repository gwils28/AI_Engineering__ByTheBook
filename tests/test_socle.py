"""Tests du socle commun : softmax, style, diagrammes."""
import matplotlib

matplotlib.use("Agg")

import numpy as np

from aie import llm
from aie.diagrams import Diagram, _Node
from aie.plotting import COLORS, set_style


def test_softmax_somme_a_un_et_stable():
    z = np.array([1000.0, 999.0, -1000.0])
    p = llm.softmax(z)
    assert np.isclose(p.sum(), 1.0)
    assert np.all(np.isfinite(p))
    assert p[0] > p[1] > p[2]


def test_softmax_temperature():
    z = np.array([2.0, 1.0, 0.0])
    froid, chaud = llm.softmax(z, 0.1), llm.softmax(z, 100.0)
    assert froid[0] > 0.99
    assert np.allclose(chaud, 1 / 3, atol=0.01)


def test_ancrage_rectangle():
    n = _Node(0, 0, 2, 1, "box")
    assert np.allclose(n.anchor(10, 0), (1, 0))
    assert np.allclose(n.anchor(0, -10), (0, -0.5))


def test_diagramme_se_construit():
    set_style()
    d = Diagram(6, 3, title="test")
    d.box("a", 1, 1, "A")
    d.box("b", 4, 1, "B", kind="db", color=COLORS[1])
    d.arrow("a", "b", "x")
    assert set(d.nodes) == {"a", "b"}
