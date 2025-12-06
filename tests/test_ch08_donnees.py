"""Tests du module aie.ch08_donnees (ingénierie des données)."""
import numpy as np
import pytest

from aie import ch08_donnees as d


# ----------------------------------------------------------------------- curation
def test_subpop_data_formes_et_classes():
    X, y, s, c = d.make_subpop_data(500, n_classes=3, n_subpops=2, dim=5)
    assert X.shape == (500, 5) and c.shape == (6, 5)
    assert np.array_equal(y, s // 2)
    X2, _, _, c2 = d.make_subpop_data(100, n_classes=3, n_subpops=2, dim=5, seed=1, centers=c)
    assert np.array_equal(c, c2)


def test_flip_labels_taux_exact():
    y = np.repeat(np.arange(4), 250)
    yb = d.flip_labels(y, 0.2, 4)
    assert np.mean(yb != y) == pytest.approx(0.2)
    yt = d.flip_labels(y, 0.3, 4, target=0)
    assert np.all(yt[yt != y] == 0)


def test_loi_de_puissance_retrouvee():
    n = np.array([20, 50, 100, 200, 500, 1000, 2000])
    err = d.power_law(n, 2.0, 0.6, 0.05)
    a, b, c = d.fit_power_law(n, err)
    assert b == pytest.approx(0.6, abs=0.05) and c == pytest.approx(0.05, abs=0.01)


def test_courbe_apprentissage_croissante():
    X, y, _, c = d.make_subpop_data(3000, seed=0)
    Xt, yt, _, _ = d.make_subpop_data(1000, seed=1, centers=c)
    m, _ = d.learning_curve(X, y, Xt, yt, [20, 2000], n_repeats=2)
    assert m[1] > m[0]


def test_diversite():
    E = np.eye(4)
    assert d.vendi_score(E) == pytest.approx(4)
    assert d.vendi_score(np.ones((4, 3)) / np.sqrt(3)) == pytest.approx(1)
    assert d.mean_pairwise_distance(E) == pytest.approx(1)
    assert d.distinct_n(["a b c", "a b c"], 2) == pytest.approx(0.5)


# --------------------------------------------------------------------- annotation
def test_cohen_kappa_comme_sklearn():
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(0)
    a = rng.integers(1, 6, 300)
    b = np.where(rng.random(300) < 0.6, a, rng.integers(1, 6, 300))
    for w in (None, "linear", "quadratic"):
        assert d.cohen_kappa(a, b, w) == pytest.approx(cohen_kappa_score(a, b, weights=w))
    assert d.cohen_kappa(a, a) == pytest.approx(1)


def test_fleiss_exemple_wikipedia():
    # exemple classique : 10 items, 14 annotateurs, 5 catégories, kappa = 0,210
    counts = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
                       [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
                       [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
    assert d.fleiss_kappa(counts) == pytest.approx(0.210, abs=1e-3)


def test_fleiss_deux_annotateurs_proche_de_scott():
    r = d.simulate_annotators(np.random.default_rng(0).integers(0, 3, 2000), [1.0, 1.0], 3)
    assert d.fleiss_kappa(d.ratings_to_counts(r, 3)) == pytest.approx(1)
    r = d.simulate_annotators(np.random.default_rng(0).integers(0, 3, 4000), [0.8, 0.8, 0.8], 3)
    k = d.fleiss_kappa(d.ratings_to_counts(r, 3))
    # accord attendu théorique : p_o = 0.8^2 + 0.2^2 / 2 = 0.66, p_e = 1/3
    assert k == pytest.approx((0.66 - 1 / 3) / (1 - 1 / 3), abs=0.03)


# ---------------------------------------------------------- augmentation, synthèse
def test_gender_swap_casse():
    assert d.gender_swap("She's a fantastic nurse.") == "He's a fantastic nurse."
    assert d.gender_swap("Mr. Wang and his DAD") == "Ms. Wang and her MOM"


def test_typo_noise_deterministe():
    t = "Comment réinitialiser mon mot de passe ?"
    assert d.typo_noise(t, 0.0) == t
    assert d.typo_noise(t, 0.2, seed=3) == d.typo_noise(t, 0.2, seed=3) != t


def test_problemes_verifiables_et_extraction():
    for p in d.make_word_problems(200):
        assert isinstance(p["answer"], int) and p["answer"] >= 0
    assert d.extract_number("Il reste donc 46 chaises.") == 46
    assert d.extract_number("Réponse : 1 250 euros") == 1250
    assert d.extract_number("aucun") is None


def test_filtre_self_instruct():
    assert d.self_instruct_filter("Aide", "ok")[0] is False
    assert d.self_instruct_filter("Explique la photosynthèse simplement.", "")[1] == "réponse vide"
    ok, _ = d.self_instruct_filter("Explique la photosynthèse simplement.", "Les plantes captent la lumière.")
    assert ok
    assert not d.self_instruct_filter("Explique la photosynthèse simplement.", "x",
                                      ["Explique la photosynthèse très simplement."])[0]


# ------------------------------------------------------------ effondrement, distillation
def test_effondrement_gaussien_et_melange():
    _, s_synth = d.gaussian_collapse(50, 300, 0.0, seed=0)
    _, s_mix = d.gaussian_collapse(50, 300, 0.5, seed=0)
    assert s_synth[-1] < 0.3 < s_mix[-1]


def test_effondrement_categoriel_perd_la_queue():
    p = 1 / np.arange(1, 101)
    h = d.categorical_collapse(p, 500, 40)
    assert (h[-1] > 0).sum() < 100
    assert np.allclose(h.sum(1), 1)
    hm = d.categorical_collapse(p, 500, 40, real_frac=0.3)
    assert (hm[-1] > 0).sum() > (h[-1] > 0).sum()


def test_kd_loss():
    z = np.array([[2.0, 1.0, 0.0]])
    assert d.kd_loss(z, z) == pytest.approx(0, abs=1e-9)
    assert d.kd_loss(z, z[:, ::-1]) > 0


def test_train_mlp_cibles_douces():
    X, y, _, _ = d.make_subpop_data(400, n_classes=2, n_subpops=1, dim=4, sep=4)
    soft = np.eye(2)[y] * 0.9 + 0.05
    f = d.train_mlp(X, soft, 2, hidden=8, epochs=30)
    assert np.mean(f(X).argmax(1) == y) > 0.9


# ---------------------------------------------------------------------- traitement
def test_jeu_sale_et_langue():
    df = d.messy_instruction_dataset()
    assert {"propre", "doublon_exact", "quasi_doublon", "html", "pii"} <= set(df.kind)
    for lang, _, ins, rep in d.BASE_PAIRS:
        assert d.detect_language(ins + " " + rep) == lang


def test_nettoyage():
    assert d.strip_html("<p>Bonjour&nbsp;&amp; merci</p>") == "Bonjour & merci"
    t, n = d.mask_pii("Écrire à jean.dupont@example.com ou au 06 12 34 56 78.")
    assert n == 2 and "[COURRIEL]" in t and "[TÉLÉPHONE]" in t and t.endswith(".")
    assert d.exact_hash("Bonjour  le monde ?") == d.exact_hash("bonjour le monde?")
    assert d.repetition_ratio("a b c a b c a b c") > 0.5


def test_minhash_estime_jaccard():
    a = d.shingles("le chat noir dort sur le canapé du salon", 3)
    b = d.shingles("le chat noir dort sur le fauteuil du salon", 3)
    mh = d.MinHasher(512, seed=0)
    est = d.minhash_jaccard(mh.signature(a), mh.signature(b))
    assert est == pytest.approx(d.jaccard(a, b), abs=0.06)


def test_lsh_retrouve_les_quasi_doublons():
    docs, pairs = d.make_near_duplicate_corpus(400, dup_frac=0.2, edit_rate=0.05)
    mh = d.MinHasher(128)
    S = mh.signatures([d.shingles(x, 3, "word") for x in docs])
    b, r = d.lsh_optimal_params(128, 0.5)
    assert b * r <= 128
    cands = d.lsh_candidates(S, b, r)
    assert len(pairs & cands) / len(pairs) > 0.9
    assert len(cands) < 400 * 399 / 2 * 0.01
    assert d.lsh_probability(1.0, b, r) == pytest.approx(1)
    groups = d.connected_groups(5, [(0, 1), (1, 2)])
    assert sorted(map(sorted, groups)) == [[0, 1, 2], [3], [4]]


def test_sft_masque_la_perte():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    ex = d.build_sft_example(tok, d.to_messages("burger -->", "edible"))
    lab = [t for t in ex["labels"] if t != -100]
    assert len(lab) > 0 and len(lab) < len(ex["input_ids"])
    assert "edible" in tok.decode(lab) and "burger" not in tok.decode(lab)
    pairs = d.fewshot_to_examples("Item: burger\nLabel: edible\n\nItem: {INPUT}\nLabel:")
    assert pairs == [("burger -->", "edible")]
