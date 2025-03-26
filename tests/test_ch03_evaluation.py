"""Tests rapides (sans GPU) du module aie.ch03_evaluation."""
import math
from math import comb

import numpy as np
import pytest

from aie import ch03_evaluation as ev


# ---------------------------------------------------------------- métriques de langage
def test_entropie_carre_du_livre():
    assert ev.entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert ev.entropy([0.25] * 4) == pytest.approx(2.0)
    assert ev.entropy([1.0, 0.0]) == pytest.approx(0.0)


def test_entropie_croisee_decomposition_kl():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p, q = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
        h, ce, kl = ev.entropy(p), ev.cross_entropy(p, q), ev.kl_divergence(p, q)
        assert ce == pytest.approx(h + kl)
        assert kl >= -1e-12
    assert ev.kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert ev.cross_entropy([0.5, 0.5], [1.0, 0.0]) == math.inf


def test_conversions_bpc_bpb_ppl():
    # exemple du livre : 6 bits/token, 2 caractères/token, 7 bits par caractère
    bpc = ev.bpc_from_bits_per_token(6, 2)
    assert bpc == 3
    assert ev.bpb_from_bpc(bpc, 7 / 8) == pytest.approx(3.4286, abs=1e-3)
    assert ev.perplexity_from_ce(2, "bits") == pytest.approx(4)
    assert ev.perplexity_from_ce(ev.bits_to_nats(2), "nats") == pytest.approx(4)
    assert ev.nats_to_bits(math.log(2)) == pytest.approx(1)


def test_min_k_prob():
    lp = np.array([-0.1, -0.2, -5.0, -0.1, -0.3, -0.05, -0.1, -0.2, -0.1, -4.0])
    assert ev.min_k_prob(lp, 0.2) == pytest.approx(-4.5)
    assert ev.min_k_prob(lp, 1.0) == pytest.approx(lp.mean())


def test_shuffle_conserve_les_mots():
    t = "the quick brown fox jumps over the lazy dog"
    s = ev.shuffle_words(t, seed=1)
    assert sorted(s.split()) == sorted(t.split()) and s != t


# ---------------------------------------------------------------- pass@k et exécution
@pytest.mark.parametrize("n,c,k", [(10, 3, 1), (10, 3, 5), (20, 0, 5), (20, 18, 3), (5, 5, 5), (100, 7, 10)])
def test_pass_at_k_formule_combinatoire(n, c, k):
    attendu = 1 - comb(n - c, k) / comb(n, k)
    assert ev.pass_at_k(n, c, k) == pytest.approx(attendu)


def test_pass_at_k_k1_est_le_taux_de_reussite():
    assert ev.pass_at_k(10, 4, 1) == pytest.approx(0.4)


def test_pass_at_k_sans_biais_par_simulation():
    rng = np.random.default_rng(0)
    p, n, k = 0.15, 20, 5
    est = [ev.pass_at_k(n, int(rng.binomial(n, p)), k) for _ in range(20000)]
    assert np.mean(est) == pytest.approx(ev.pass_at_k_true(p, k), abs=0.01)


def test_extract_code_et_run_tests():
    txt = "Voici :\n```python\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n```\nFin."
    code = ev.extract_code(txt)
    assert code.startswith("def gcd")
    pb = ev.CODE_PROBLEMS[0]
    r = ev.run_tests(code, pb["tests"])
    assert r.passed and r.n_passed == r.n_tests
    faux = ev.run_tests("def gcd(a, b):\n    return 1", pb["tests"])
    assert not faux.passed and 0 < faux.n_passed < faux.n_tests
    assert ev.run_tests("def gcd(a, b) return", pb["tests"]).error == "SyntaxError"
    assert ev.run_tests("def gcd(a, b):\n    return a\nassert gcd(1, 2) == 2", pb["tests"]).error == "LoadError"
    assert ev.run_tests("while True: pass", pb["tests"], timeout=1).error == "Timeout"


def test_sanitize_code():
    brut = ("import math\n\ndef gcd(a, b):\n    return math.gcd(a, b)\n\nprint(gcd(4, 6))\n"
            "assert gcd(1, 2) == 5\ndef incomplete(x):\n    return (x +")
    propre = ev.sanitize_code(brut)
    assert "print" not in propre and "assert" not in propre and "incomplete" not in propre
    assert ev.run_tests(propre, ev.CODE_PROBLEMS[0]["tests"]).passed
    assert ev.run_tests(brut, ev.CODE_PROBLEMS[0]["tests"]).error == "SyntaxError"


def test_solutions_canoniques_passent_tous_les_tests():
    assert set(ev.REFERENCE_SOLUTIONS) == {pb["name"] for pb in ev.CODE_PROBLEMS}
    for pb in ev.CODE_PROBLEMS:
        r = ev.run_tests(ev.REFERENCE_SOLUTIONS[pb["name"]], pb["tests"])
        assert r.passed, (pb["name"], r)


def test_solutions_de_reference_du_banc():
    sols = {
        "pgcd": "import math\ndef gcd(a, b): return math.gcd(a, b)",
        "voyelles": "def count_vowels(s): return sum(c in 'aeiou' for c in s.lower())",
        "rle": ("def run_length_encode(s):\n    import itertools\n"
                "    return ''.join(f'{k}{len(list(g))}' for k, g in itertools.groupby(s))"),
    }
    for pb in ev.CODE_PROBLEMS:
        if pb["name"] in sols:
            assert ev.run_tests(sols[pb["name"]], pb["tests"]).passed, pb["name"]


# ---------------------------------------------------------------- similarité
def test_exact_et_contains_match():
    assert ev.exact_match("The answer is 5.", ["answer is 5"])
    assert not ev.exact_match("The answer is 5.", ["5"])
    assert ev.contains_match("The answer is 5.", ["5"])
    assert ev.contains_match("September 12, 1929", ["1929"])  # faux positif du livre
    assert not ev.contains_match("15", ["5"])


def test_edit_distance_exemples_du_livre():
    assert ev.edit_distance("brad", "bad") == 1
    assert ev.edit_distance("bad", "bard") == 1
    assert ev.edit_distance("bad", "bed") == 1
    assert ev.edit_distance("bad", "cash") == 3
    assert ev.edit_distance("mats", "mast") == 2
    assert ev.edit_distance("mats", "mast", transpositions=True) == 1
    assert ev.edit_similarity("abc", "abc") == 1.0


def test_unigrammes_du_livre():
    ref = "My cats scare the mice"
    assert ev.rouge_n("My cats eat the mice", ref, 1)["rappel"] == pytest.approx(0.8)
    assert ev.rouge_n("Cats and mice fight all the time", ref, 1)["rappel"] == pytest.approx(0.6)
    assert len(ev.ngrams(ev.words(ref), 2)) == 4


def test_bleu_proprietes():
    ref = "the cat is on the mat"
    assert ev.bleu(ref, ref) == pytest.approx(1.0)
    assert ev.bleu("the the the the the the", ref, smooth=False) == 0.0
    # précision unigramme écrêtée : "the" x7 contre 2 dans la référence -> 2/7
    assert ev.bleu("the the the the the the the", ref, max_n=1) == pytest.approx(2 / 7)
    # pénalité de brièveté : candidat plus court que la référence
    assert ev.bleu("the cat", ref, max_n=1) == pytest.approx(math.exp(1 - 6 / 2))
    assert 0 < ev.bleu("a cat is on the mat", ref) < 1


def test_rouge_l():
    r = ev.rouge_l("police killed the gunman", "police kill the gunman")
    assert r["f1"] == pytest.approx(0.75)
    assert ev.lcs_length("ABCBDAB", "BDCABA") == 4


def test_cosinus():
    assert ev.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1)
    assert ev.cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1)
    assert ev.cosine_similarity([1, 0], [0, 3]) == pytest.approx(0)
    m = ev.cosine_similarity(np.eye(3), np.eye(3))
    assert np.allclose(m, np.eye(3))


# ---------------------------------------------------------------- juge et coûts
def test_score_attendu_et_cout():
    assert ev.expected_score({"1": 0.0, "2": 0.0, "3": 0.5, "4": 0.5, "5": 0.0, "_masse": 0.9}) == pytest.approx(3.5)
    c = ev.judge_cost(1000, 500, 200, 10, price_in=1.0, price_out=4.0, n_criteria=3)
    assert c["generation"] == pytest.approx(1000 * (500 + 800) / 1e6)
    assert c["juge"] == pytest.approx(3000 * (1000 * 1.0 + 40) / 1e6)
    assert ev.spot_check_sample_size(0.5, 0.05) == 385


# ---------------------------------------------------------------- Elo et Bradley-Terry
def test_elo():
    assert ev.elo_expected(1000, 1000) == pytest.approx(0.5)
    assert ev.elo_expected(1400, 1000) == pytest.approx(10 / 11)
    ra, rb = ev.elo_update(1000, 1000, 1.0, k=32)
    assert (ra, rb) == (1016, 984)
    r = ev.elo_ratings([(0, 1, 1.0), (0, 1, 0.0)], 2)
    assert r.sum() == pytest.approx(2000)  # Elo conserve la somme des scores


def test_bradley_terry_retrouve_les_forces():
    rng = np.random.default_rng(0)
    vrai = np.array([1200, 1100, 1000, 900, 800.0])
    P = ev.prob_matrix_from_elo(vrai)
    matches = ev.simulate_matches(P, 20000, rng)
    s = ev.bradley_terry(ev.win_matrix(matches, 5))
    elo = ev.bt_to_elo(s)
    assert np.all(np.diff(elo) < 0)
    assert np.allclose(elo - elo.mean(), vrai - vrai.mean(), atol=25)
    assert ev.ranking_accuracy(s, matches) > 0.6


def test_bt_probabilite_coherente_avec_elo():
    s = ev.bradley_terry(np.array([[0, 75.0], [25.0, 0]]))
    assert ev.bt_prob(s[0], s[1]) == pytest.approx(0.75, abs=1e-6)


def test_table_du_livre_contient_un_cycle():
    w = ev.book_win_matrix()
    P = w / (w + w.T + np.eye(5))
    assert (0, 1, 2) in ev.find_cycles3(P)  # 1 bat 2, 2 bat 3, 3 bat 1


def test_matches_needed_decroit_avec_l_ecart():
    assert ev.matches_needed(0.51) > ev.matches_needed(0.55) > ev.matches_needed(0.7)
    assert 15000 < ev.matches_needed(0.51) < 25000
