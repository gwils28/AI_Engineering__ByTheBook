"""Tests rapides (sans GPU) des outils du chapitre 4."""
import numpy as np
import pandas as pd
import pytest

from aie import ch04_systemes as c4


# ------------------------------------------------------------------ QCM
def test_format_mcq_suit_la_permutation():
    item = c4.MCQ_ITEMS[0]
    p, g = c4.format_mcq(item)
    assert g == item["reponse"] == 0 and "A. Canberra" in p
    p2, g2 = c4.format_mcq(item, order=[3, 2, 1, 0])
    assert g2 == 3 and "D. Canberra" in p2
    p3, _ = c4.format_mcq(item, style="choix")
    assert "Choix :" in p3


def test_baseline_et_binomiale():
    assert c4.random_baseline(4) == 0.25
    assert np.isclose(c4.binom_sf(0, 10, 0.3), 1.0)
    assert np.isclose(c4.binom_sf(10, 10, 0.5), 0.5**10)


def test_items_qcm_coherents_et_equilibres():
    for it in c4.MCQ_ITEMS:
        assert len(it["choix"]) == 4 and 0 <= it["reponse"] < 4
    counts = np.bincount([it["reponse"] for it in c4.MCQ_ITEMS], minlength=4)
    assert counts.max() - counts.min() <= 1


# ------------------------------------------------------------------ NLI / SelfCheck (NLI factice)
def _fake_nli(premises, hypotheses):
    """Implication si l'hypothèse est contenue dans la prémisse, contradiction sinon."""
    out = []
    for p, h in zip(premises, hypotheses):
        out.append([0.9, 0.05, 0.05] if h.lower().strip(".") in p.lower() else [0.05, 0.05, 0.9])
    return np.array(out)


def test_split_sentences():
    assert c4.split_sentences("Bonjour. Ça va ? Oui !") == ["Bonjour.", "Ça va ?", "Oui !"]


def test_factual_consistency_avec_nli_factice():
    df = c4.factual_consistency("Paris est la capitale de la France.",
                                "Paris est la capitale de la France. Lyon est la capitale.", nli=_fake_nli)
    assert list(df["verdict"]) == ["implication", "contradiction"]


def test_selfcheck():
    samples = ["Victor Hugo a écrit Les Misérables.", "Les Misérables est de Victor Hugo. Victor Hugo a écrit Les Misérables."]
    s = c4.selfcheck_scores(["Victor Hugo a écrit Les Misérables.", "Zola a écrit Les Misérables."], samples, nli=_fake_nli)
    assert s[0] < 0.2 and s[1] > 0.8


# ------------------------------------------------------------------ IFEval
@pytest.mark.parametrize("kind,params,ok,ko", [
    ("nombre_mots", dict(n=3, relation="au plus"), "un deux trois", "un deux trois quatre"),
    ("puces", dict(n=2), "* a\n* b", "* a\n* b\n* c"),
    ("titre", {}, "<<Mon titre>>\ntexte", "Mon titre"),
    ("json", {}, '```json\n{"a": 1}\n```', "{a: 1}"),
    ("minuscules", {}, "tout petit", "Pas petit"),
    ("majuscules", {}, "TOUT GRAND !", "Pas GRAND"),
    ("mots_interdits", dict(words=["eau"]), "le ciel est bleu", "de l'eau"),
    ("mots_inclus", dict(keywords=["lait", "farine"]), "Farine et lait.", "Farine seule."),
    ("post_scriptum", dict(marker="P.S."), "Salut.\nP.S. à demain", "Salut."),
    ("emplacements", dict(n=2), "[nom] habite [ville]", "[nom] habite ici"),
    ("fin_phrase", dict(phrase="Voilà."), "C'est tout. Voilà.", "Voilà. C'est tout."),
    ("choix_parmi", dict(options=["POSITIF", "NEGATIF"]), "POSITIF", "Plutôt positif"),
    ("surlignage", dict(n=2), "*un* et *deux*", "*un* seul"),
    ("sans_virgule", {}, "pas de virgule", "une, virgule"),
    ("guillemets", {}, '"Citation."', "Citation."),
    ("paragraphes", dict(n=2), "un\n***\ndeux", "un\n\ndeux"),
    ("frequence_mot", dict(word="chat", n=2, relation="au moins"), "chat chat", "chat chaton"),
    ("frequence_lettre", dict(letter="z", n=2, relation="au moins"), "zéro zèbre", "zéro"),
    ("nombre_phrases", dict(n=1, relation="au plus"), "Une phrase.", "Une. Deux."),
])
def test_verificateurs(kind, params, ok, ko):
    inst = c4.Instruction(kind, params)
    assert inst.check(ok) and not inst.check(ko)
    assert isinstance(inst.text, str) and len(inst.text) > 5


def test_variantes_souples_et_resume():
    tasks = [c4.IFTask("x", [c4.Instruction("titre"), c4.Instruction("minuscules")]),
             c4.IFTask("y", [c4.Instruction("choix_parmi", dict(options=["OUI", "NON"]))])]
    # la 1re réponse rate « minuscules » en strict (titre en majuscule) ; la 2e passe en souple
    resp = ["<<titre>>\nbonjour", "Voici ma réponse :\nOUI"]
    df = c4.evaluate_ifeval(tasks, resp)
    assert df["strict"].tolist() == [True, True, False]
    assert df["souple"].tolist() == [True, True, True]
    s = c4.ifeval_summary(df)
    assert s["prompt strict"] == 0.5 and s["prompt souple"] == 1.0
    assert np.isclose(s["instruction strict"], 2 / 3)


def test_toutes_les_taches_ont_un_texte():
    for t in c4.IF_TASKS:
        assert t.prompt.startswith(t.base) and len(t.instructions) >= 1
        for i in t.instructions:
            assert i.kind in c4.INSTRUCTION_TYPES


# ------------------------------------------------------------------ coût, latence
def test_latency_model():
    assert np.isclose(c4.latency_model(101, ttft=0.2, tpot=0.01), 1.2)


def test_couts_et_point_mort():
    api = lambda n: c4.api_cost(n, 1000, 500, price_in=1.0, price_out=4.0)    # 0,003 $ / requête
    assert np.isclose(api(1000), 3.0)
    self_ = lambda n: c4.selfhost_cost(n, 1500, gpu_hour=2.0, gpu_tokens_per_s=1000,
                                       fixed_monthly=0.0, utilization=1.0, hours=730)
    assert np.isclose(self_(1), 2.0 * 730)          # un GPU minimum
    be = c4.breakeven(api, self_)
    assert np.isclose(be, 1460 / 0.003, rtol=0.01)  # 486 667 requêtes / mois


def test_selfhost_nombre_entier_de_gpu():
    cap = 1000 * 3600 * 730  # tokens par GPU et par mois à utilisation 1
    c = c4.selfhost_cost(np.array([cap / 1500, cap / 1500 + 1]), 1500, 1.0, 1000, utilization=1.0)
    assert c[1] == 2 * c[0]


# ------------------------------------------------------------------ sélection
def test_pareto():
    keep = c4.pareto_front([1, 2, 3, 2.5], [50, 70, 80, 60])
    assert keep.tolist() == [True, True, True, False]


def test_hard_filter():
    out = c4.hard_filter(c4.CATALOGUE, {"poids ouverts": lambda d: d["poids_ouverts"]})
    assert set(out.loc[out["elimine_par"] == "", "modele"]) == set(c4.CATALOGUE.loc[c4.CATALOGUE.poids_ouverts, "modele"])


def test_agregations():
    s = pd.DataFrame({"b1": [90, 50, 10], "b2": [10, 50, 90]}, index=["A", "B", "C"])
    mwr = c4.mean_win_rate(s)
    assert np.allclose(mwr.values, [0.5, 0.5, 0.5])
    z = c4.zscore_mean(s, weights=[3, 1])
    assert z.idxmax() == "A"


def test_correlation_symetrique_et_simulation():
    assert np.allclose(c4.HF_CORR.values, c4.HF_CORR.values.T)
    lb = c4.synthetic_leaderboard(400, seed=1)
    r = lb.corr().loc["MMLU", "WinoGrande"]
    assert r > 0.7


# ------------------------------------------------------------------ contamination
def test_ngram_overlap():
    train = "le renard brun rapide saute par dessus le chien paresseux tous les matins"
    grams = c4.ngrams(c4.simple_tokenize(train), 5)
    assert c4.ngram_overlap("Le renard brun rapide saute par dessus", grams, n=5) == 1.0
    assert c4.ngram_overlap("un tout autre texte sans aucun rapport ici", grams, n=5) == 0.0
    assert c4.ngram_overlap("trop court", grams, n=5) == 0.0


def test_saturation():
    y = c4.logistic_saturation([-100, 0, 100], t0=0, rate=1)
    assert np.allclose(y, [25, 60, 95])


# ------------------------------------------------------------------ statistiques
def test_bootstrap_ci_couvre_la_moyenne():
    x = np.r_[np.ones(70), np.zeros(30)]
    m, lo, hi = c4.bootstrap_ci(x, seed=0)
    assert m == 0.7 and lo < 0.7 < hi and 0.05 < hi - lo < 0.25


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    a = rng.random(500) < 0.6
    b = a | (rng.random(500) < 0.2)       # b corrige une partie des erreurs de a
    r = c4.paired_bootstrap(a, b)
    assert r["diff"] > 0 and r["lo"] > 0 and r["p"] < 0.01


def test_taille_echantillon_regle_3x_10x():
    n10 = c4.sample_size_two_proportions(0.5, 0.10)
    n3 = c4.sample_size_two_proportions(0.5, 0.10 / 3)
    assert 8.5 < n3 / n10 < 9.5          # (3)² = 9 : la « règle 3x -> 10x » arrondit √10 ≈ 3,16
    assert 350 < n10 < 420
    np_ = c4.sample_size_two_proportions(0.5, 0.10, paired=True, discordance=0.2)
    assert np_ < n10


def test_simpson():
    counts = {"A": {"g1": (81, 87), "g2": (192, 263)}, "B": {"g1": (234, 270), "g2": (55, 80)}}
    t = c4.simpson_table(counts)
    assert t.loc["A", "g1"] > t.loc["B", "g1"] and t.loc["A", "global"] < t.loc["B", "global"]
    assert c4.is_simpson(counts)
    assert not c4.is_simpson({"A": {"g1": (9, 10), "g2": (9, 10)}, "B": {"g1": (1, 10), "g2": (1, 10)}})


def test_kappa():
    assert c4.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert np.isclose(c4.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]), 0.0)
