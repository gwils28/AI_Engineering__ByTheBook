"""Tests rapides du module du chapitre 1 (sans GPU, sans modèle de langage)."""
import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from aie import ch01_introduction as c1


# ---------------------------------------------------------------- statistiques du langage
def test_frequences_somment_a_un_et_e_en_tete():
    f = c1.letter_frequencies(c1.CORPUS_FR)
    assert math.isclose(sum(f.values()), 1.0)
    assert next(iter(f)) == "e"  # la lettre la plus fréquente du français


def test_entropie_uniforme_et_degeneree():
    assert math.isclose(c1.entropy([0.25] * 4), 2.0)
    assert c1.entropy([1.0, 0.0]) == 0.0
    assert math.isclose(c1.entropy({"a": 1, "b": 1}), 1.0)  # normalisation automatique


def test_ngram_distribution_normalisee_et_contexte():
    text = c1.normalize_text(c1.CORPUS_FR)
    m = c1.CharNGram(order=3).fit(text)
    d = m.distribution("qu")
    assert math.isclose(sum(d.values()), 1.0)
    assert max(d, key=d.get) in "eia"  # après « qu », une voyelle
    assert len(m.generate(30, seed=1)) == 30
    assert m.generate(20, seed=3) == m.generate(20, seed=3)


def test_entropie_conditionnelle_decroit_avec_l_ordre():
    text = c1.normalize_text(c1.CORPUS_FR)
    h = [c1.conditional_entropy(text, k) for k in (1, 2, 3, 4)]
    assert all(a > b for a, b in zip(h, h[1:]))
    assert 3.5 < h[0] < 4.6  # ~ 4 bits par caractère sans contexte


# ---------------------------------------------------------------- tokenisation
def test_bpe_fusionne_les_paires_frequentes():
    corpus = ["bas", "basse", "bassin", "base", "basket"] * 3
    merges = c1.train_bpe(corpus, 3)
    # égalité (a, s) / (b, a) : départage lexicographique déterministe
    assert merges[0] == ("a", "s")
    assert merges[1] == ("b", "as")
    toks = c1.bpe_tokenize("basalte", merges)
    assert toks[0].startswith("bas")
    assert "".join(toks) == "basalte_"


def test_bpe_vocabulaire_croit_d_un_par_fusion():
    corpus = c1.words(c1.CORPUS_FR.lower())
    m10, m50 = c1.train_bpe(corpus, 10), c1.train_bpe(corpus, 50)
    assert len(c1.bpe_vocab(corpus, m50)) - len(c1.bpe_vocab(corpus, m10)) <= 40
    assert len(m50) == 50
    assert len(c1.bpe_tokenize("construire", m50)) < len("construire") + 1


def test_tiktoken_compte_et_reconstruit():
    tok = c1.get_tokenizer("GPT-4 (cl100k)")
    text = "J'ai hâte de construire des applications d'IA."
    pieces = tok.pieces(text)
    assert "".join(pieces) == text
    assert tok.count(text) == len(pieces)
    assert tok.vocab_size > 100_000


def test_tokens_par_mot_plus_eleve_en_francais_pour_gpt2():
    tok = c1.get_tokenizer("GPT-2")
    en = " ".join(e for e, _ in c1.PARALLEL_EN_FR)
    fr = " ".join(f for _, f in c1.PARALLEL_EN_FR)
    assert c1.tokens_per_word(fr, tok) > c1.tokens_per_word(en, tok)


# ---------------------------------------------------------------- auto-supervision
def test_paires_autoregressives_exemple_du_livre():
    pairs = c1.autoregressive_pairs(["I", "love", "street", "food", "."])
    assert len(pairs) == 6
    assert pairs[0] == (["<BOS>"], "I")
    assert pairs[-1][1] == "<EOS>"
    assert pairs[-1][0] == ["<BOS>", "I", "love", "street", "food", "."]


def test_paires_fenetre_de_contexte():
    pairs = c1.autoregressive_pairs(list("abcdef"), bos=None, eos=None, max_context=2)
    assert len(pairs) == 5
    assert all(len(ctx) <= 2 for ctx, _ in pairs)
    assert pairs[-1] == (["d", "e"], "f")


def test_paires_masquees():
    toks = list("abcdefghijklmnopqrst")
    masked, targets = c1.masked_pairs(toks, rate=0.15, seed=0)
    assert len(masked) == len(toks)
    assert len(targets) == 3
    for i, t in targets.items():
        assert masked[i] == c1.MASK and toks[i] == t


def test_cout_etiquetage():
    assert c1.labeling_cost(1_000_000, 0.05) == pytest.approx(50_000)
    assert c1.labeling_cost(1_000_000, 0.05, annotators=2) == pytest.approx(100_000)


# ---------------------------------------------------------------- échelle
def test_parametres_gpt2_small():
    n = c1.transformer_params(12, 768, 50257, n_ctx=1024)
    assert abs(n - 124_439_808) / 124_439_808 < 0.01  # valeur exacte : 124 439 808


def test_regle_12_L_d2():
    n = c1.transformer_params(96, 12288, 0)
    assert n == 12 * 96 * 12288 ** 2


def test_chinchilla_et_flops():
    assert c1.chinchilla_tokens(70e9) == pytest.approx(1.4e12)
    assert c1.training_flops(1e9, 1e12) == pytest.approx(6e21)
    assert c1.weights_memory_gb(7e9) == pytest.approx(14.0)


def test_tableau_des_tailles():
    df = c1.model_sizes_frame()
    assert {"modèle", "date", "params_Md"} <= set(df.columns)
    assert df.loc[df["modèle"] == "GPT-3", "params_Md"].item() == 175.0


# ---------------------------------------------------------------- planification
def test_cout_requete_et_mensuel():
    assert c1.request_cost(1_000_000, 0, 0.15, 0.60) == pytest.approx(0.15)
    c = c1.request_cost(1000, 500, 0.15, 0.60)
    assert c == pytest.approx(0.00045)
    assert c1.monthly_cost(10_000, 1000, 500, 0.15, 0.60) == pytest.approx(135.0)


def test_latence():
    assert c1.generation_latency(100, 0.2, 0.01) == pytest.approx(1.19)
    assert c1.generation_latency(1, 0.2, 0.01) == pytest.approx(0.2)


def test_grille_de_decision():
    top = {k: 3 for k in c1.DEFAULT_WEIGHTS}
    bottom = {k: 0 for k in c1.DEFAULT_WEIGHTS}
    assert c1.decision_score(top) == pytest.approx(100)
    assert c1.decision_score(bottom) == pytest.approx(0)
    assert c1.decision_verdict(100, top) == "construire"
    blocked = dict(top, capacite_modele=0)
    assert "bloquant" in c1.decision_verdict(c1.decision_score(blocked), blocked)
    with pytest.raises(KeyError):
        c1.decision_score({"enjeu": 3})


def test_loi_de_puissance_linkedin():
    # 80 % au bout d'un mois, 95 % au bout de cinq
    a, b = c1.fit_power_law_error([1, 5], [80, 95])
    assert a == pytest.approx(20)
    assert b == pytest.approx(math.log(4) / math.log(5))
    assert c1.time_to_quality(95, a, b) == pytest.approx(5)
    assert np.allclose(c1.quality_power_law([1, 5], a, b), [80, 95])


def test_modele_exponentiel():
    q = c1.quality_exponential([0, 1e9], tau=2.0)
    assert q[0] == 0 and q[1] == pytest.approx(100)


def test_wilson():
    lo, hi = c1.wilson_interval(95, 100)
    assert lo < 0.95 < hi
    assert 0.88 < lo < 0.90
    n = c1.samples_to_certify(0.97, 0.95)
    assert n > 100
    lo_n, _ = c1.wilson_interval(round(0.97 * n), n, 1.645)
    assert lo_n > 0.95
    assert c1.samples_to_certify(0.94, 0.95) == -1


def test_construire_ou_acheter():
    api = c1.api_cumulative_cost(24, 100, 10.0)
    assert api[0] == 0 and api[-1] == pytest.approx(24_000)
    inh = c1.inhouse_cumulative_cost(24, 5_000, 500)
    assert inh[0] == 5_000 and inh[-1] == pytest.approx(17_000)
    m = c1.break_even_month(api, inh)
    assert m == 10  # 5000 + 500 m <= 1000 m  <=>  m >= 10
    falling = c1.api_cumulative_cost(24, 100, 10.0, halving_months=6)
    assert falling[-1] < api[-1]
    assert c1.break_even_month(falling, inh) is None or c1.break_even_month(falling, inh) > m


def test_tokens_conversation_multi_tours():
    assert c1.conversation_tokens(1, 800, 600, 60, 180) == (1460, 180)
    n_in, n_out = c1.conversation_tokens(4, 800, 600, 60, 180)
    assert n_in == 4 * 1460 + 240 * 6 and n_out == 720
    # l'historique renvoyé à chaque tour : croissance quadratique
    a = c1.conversation_tokens(10, 0, 0, 10, 10)[0]
    b = c1.conversation_tokens(20, 0, 0, 10, 10)[0]
    assert b / a > 3.5


def test_nombres_lisibles():
    assert c1.human_number(4615) == "4,6 k"
    assert c1.human_number(1.5e9) == "1,5 Md"
    assert c1.human_number(2e12) == "2 000 Md"
    assert c1.human_number(500) == "500"
