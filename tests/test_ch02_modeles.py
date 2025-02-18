"""Tests rapides (sans GPU ni téléchargement) des fonctions du chapitre 2."""
import math

import numpy as np
import pytest

from aie import ch02_modeles as m
from aie import llm

rng = np.random.default_rng(0)


# ------------------------------------------------------------------------- données
def test_corpus_parallele_complet():
    assert all(len(s) == 6 for s in m.PARALLEL_CORPUS.values())
    assert set(m.PARALLEL_CORPUS) <= set(m.LANG_NAMES)


def test_tokens_par_langue_ratio_anglais_vaut_un():
    df = m.tokens_per_language({"car": lambda s: list(s)})
    assert np.isclose(df.loc[df.code == "en", "ratio_vs_en"].item(), 1.0)
    assert (df["tokens"] == df["caracteres"]).all()


def test_table_common_crawl():
    df = m.common_crawl_table()
    pa = df.loc[df.code == "pa"].iloc[0]
    assert math.isclose(pa.ratio_pop_cc, 231.5, rel_tol=0.01)  # valeur du tableau 2-2
    assert df.loc[df.code == "en", "ratio_pop_cc"].item() < 1


# ------------------------------------------------------------------------ attention
def test_attention_poids_somment_a_un_et_masque_causal():
    T, d = 5, 8
    Q, K, V = rng.normal(size=(3, T, d))
    out, w = m.scaled_dot_product_attention(Q, K, V, m.causal_mask(T))
    assert out.shape == (T, d)
    assert np.allclose(w.sum(-1), 1)
    assert np.allclose(np.triu(w, 1), 0)
    assert np.allclose(out[0], V[0])  # le premier token ne voit que lui-même


def test_attention_requete_alignee_choisit_la_bonne_valeur():
    K = np.eye(4) * 20
    V = np.arange(16.0).reshape(4, 4)
    out, w = m.scaled_dot_product_attention(K[2:3], K, V)
    assert np.argmax(w) == 2 and np.allclose(out[0], V[2], atol=1e-6)


def test_multi_tetes_une_tete_egale_attention_simple():
    T, d = 6, 4
    x = rng.normal(size=(T, d))
    W = [rng.normal(size=(d, d)) for _ in range(4)]
    out, w = m.multi_head_attention(x, *W, n_heads=1)
    ref, _ = m.scaled_dot_product_attention(x @ W[0], x @ W[1], x @ W[2], m.causal_mask(T))
    assert np.allclose(out, ref @ W[3])
    out2, w2 = m.multi_head_attention(x, *W, n_heads=2)
    assert w2.shape == (2, T, T) and out2.shape == (T, d)


def test_rope_conserve_la_norme_et_depend_des_ecarts():
    q, k = rng.normal(size=(2, 1, 16))
    pos = np.array([3])
    assert np.isclose(np.linalg.norm(m.rope(q, pos)), np.linalg.norm(q))
    s1 = (m.rope(q, [5]) @ m.rope(k, [2]).T).item()
    s2 = (m.rope(q, [13]) @ m.rope(k, [10]).T).item()
    assert np.isclose(s1, s2)  # seul l'écart de positions compte


# ---------------------------------------------------------------------- paramètres
@pytest.mark.parametrize("name,expected", [
    ("GPT-2 (124M)", 124_439_808), ("Qwen2.5-0.5B", 494_032_768), ("Llama 2-7B", 6_738_415_616),
    ("Llama 3-8B", 8_030_261_248),
])
def test_compte_parametres_exact(name, expected):
    assert m.count_params_config(m.KNOWN_CONFIGS[name]) == expected


def test_moe_parametres_actifs():
    c = m.KNOWN_CONFIGS["Mixtral 8x7B"]
    assert abs(m.count_params_config(c) / 1e9 - 46.7) < 0.1
    assert abs(m.count_params_config(c, active=True) / 1e9 - 12.9) < 0.1


def test_kv_cache_llama2():
    # Llama 2-7B, 4096 tokens, fp16 : 2 x 32 x 32 x 128 x 4096 x 2 octets = 2 Gio
    assert m.kv_cache_bytes(m.KNOWN_CONFIGS["Llama 2-7B"], 4096) == 2 * 32 * 4096 * 4096 * 2
    assert m.weights_memory_gb(7e9, 2) == 14.0


# -------------------------------------------------------------------------- échelle
def test_flops_gpt3_et_duree_du_livre():
    C = m.training_flops(175e9, 300e9)
    assert math.isclose(C, 3.15e23, rel_tol=0.01)
    days = m.training_days(3.14e23, 256, 5.2e18 / 86400)
    assert 230 < days < 240


def test_chinchilla_regle_20x_et_parametrique():
    N, D = m.chinchilla_optimal(6 * 70e9 * 1.4e12)
    assert math.isclose(N, 70e9, rel_tol=1e-6) and math.isclose(D, 1.4e12, rel_tol=1e-6)
    C = 1e22
    Np, Dp = m.chinchilla_optimal(C, "parametric")
    assert math.isclose(6 * Np * Dp, C, rel_tol=1e-9)
    grid = np.logspace(8, 11, 400)
    assert abs(np.log10(grid[np.argmin(m.chinchilla_loss(grid, C / (6 * grid)))]) - np.log10(Np)) < 0.02


# ----------------------------------------------------------------- post-entraînement
def test_bradley_terry_retrouve_l_ordre():
    true = np.array([1.5, 0.5, 0.0, -2.0])
    W, L = [], []
    for _ in range(3000):
        i, j = rng.choice(4, 2, replace=False)
        if rng.random() < m.sigmoid(true[i] - true[j]):
            W.append(i); L.append(j)
        else:
            W.append(j); L.append(i)
    s, hist = m.fit_bradley_terry(W, L, 4)
    assert list(np.argsort(-s)) == [0, 1, 2, 3]
    assert hist[-1] < hist[0]
    assert np.allclose(s - s.mean(), true - true.mean(), atol=0.25)


def test_recompense_lineaire():
    w_true = np.array([2.0, -1.0, 0.0])
    X1, X2 = rng.normal(size=(2, 2000, 3))
    win = rng.random(2000) < m.sigmoid((X1 - X2) @ w_true)
    Xw = np.where(win[:, None], X1, X2)
    Xl = np.where(win[:, None], X2, X1)
    w, _ = m.fit_linear_reward(Xw, Xl)
    assert np.allclose(w, w_true, atol=0.3)


def test_politique_regularisee_kl():
    p = np.array([0.5, 0.3, 0.2])
    r = np.array([0.0, 1.0, 2.0])
    assert np.allclose(m.kl_regularized_policy(p, r, 1e6), p, atol=1e-5)
    assert m.kl_regularized_policy(p, r, 1e-3)[2] > 0.999


# --------------------------------------------------------------------- échantillonnage
def test_entropie():
    assert math.isclose(m.entropy(np.ones(4) / 4, base=2), 2.0)
    assert m.entropy([1.0, 0.0]) == 0.0


def test_temperature_exemple_du_livre():
    assert np.allclose(llm.softmax(np.array([1.0, 2.0])), [0.27, 0.73], atol=0.01)
    assert np.allclose(llm.softmax(np.array([1.0, 2.0]), 0.5), [0.12, 0.88], atol=0.01)


def test_top_k_top_p_min_p():
    z = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    assert np.isfinite(m.top_k_filter(z, 2)).sum() == 2
    assert np.isfinite(m.top_p_filter(z, 0.8)).sum() == 2  # 0.5 + 0.3 = 0.8 atteint
    assert np.isfinite(m.top_p_filter(z, 0.81)).sum() == 3
    assert np.isfinite(m.top_p_filter(z, 0.01)).sum() == 1
    assert np.isfinite(m.min_p_filter(z, 0.5)).sum() == 2  # garde p >= 0.25
    p = m.sampling_distribution(z, top_p=0.8)
    assert np.allclose(p[:2], [0.625, 0.375]) and p[2:].sum() == 0
    assert m.sampling_distribution(z, temperature=0)[0] == 1.0


def test_echantillonnage_respecte_les_frequences():
    z = np.log(np.array([0.7, 0.3]))
    r = np.random.default_rng(1)
    draws = [m.sample_token(z, r) for _ in range(4000)]
    assert abs(np.mean(draws) - 0.3) < 0.03


def test_vote_majoritaire():
    assert m.majority_vote([3, 3, None, 5]) == (3, 2 / 3)
    assert m.majority_vote([None]) == (None, 0.0)
    assert math.isclose(m.majority_vote_accuracy(0.6, 1), 0.6)
    assert m.majority_vote_accuracy(0.6, 25) > 0.8
    assert m.majority_vote_accuracy(0.4, 25) < 0.2
    assert math.isclose(m.pass_at_n(0.5, 3), 0.875)


def test_extraction_de_nombre_et_logprob_sequence():
    assert m.extract_number("Il reste 42 pommes, soit 1,234 au total : 1234") == 1234
    assert m.extract_number("The answer is 17.") == 17
    assert m.extract_number("aucun") is None
    lp = np.log([0.2, 0.1, 0.3])
    assert math.isclose(math.exp(m.sequence_logprob(lp)), 0.006)


# ------------------------------------------------------------------ sorties structurées
def test_regex_partielle():
    pat = r'\{"label": "(positif|négatif)"\}'
    assert m.regex_allows(pat, '{"lab')
    assert m.regex_allows(pat, '{"label": "pos')
    assert not m.regex_allows(pat, '{"label": "x')
    assert not m.regex_allows(pat, '{"label": "pos', partial=False)
    assert m.regex_allows(pat, '{"label": "positif"}', partial=False)


def test_reparation_json():
    assert m.repair_json('Voici : ```json\n{"a": 1, "b": [1, 2],}\n```') == {"a": 1, "b": [1, 2]}
    assert m.repair_json("{'a': 'x'") == {"a": "x"}
    assert m.repair_json("pas de json") is None
