"""Tests rapides (sans GPU) des fonctions du chapitre 9."""
import math

import numpy as np
import pandas as pd
import pytest

from aie import ch09_inference as inf

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- roofline et matériel
def test_intensite_produit_matrice_vecteur_vaut_environ_un():
    assert inf.matmul_intensity(8192, 8192, 1) == pytest.approx(1.0, rel=1e-3)
    # matrices carrées : n / 3 en bf16
    assert inf.matmul_intensity(3000, 3000, 3000) == pytest.approx(1000, rel=1e-9)


def test_roofline_et_point_equilibre():
    pi, beta = 1000e12, 4e12
    r = inf.ridge_point(pi, beta)
    assert r == 250
    assert inf.roofline(r / 10, pi, beta) == pytest.approx(pi / 10)
    assert inf.roofline(r * 10, pi, beta) == pi
    assert inf.bound_kind(10, pi, beta) == "mémoire" and inf.bound_kind(1000, pi, beta) == "calcul"


def test_intensite_decode_proportionnelle_au_lot():
    N = 7e9
    assert inf.decode_intensity(N, 1) == pytest.approx(1.0)
    assert inf.decode_intensity(N, 32) == pytest.approx(32.0)
    # la lecture du cache KV réduit l'intensité
    assert inf.decode_intensity(N, 32, kv_bytes_per_seq=1e8) < 32


def test_borne_de_debit_decode():
    # 7B en fp16 sur 2 To/s : un pas lit 14 Go -> ~143 tokens/s
    assert inf.decode_tokens_per_s_bound(7e9, 2e12, 2) == pytest.approx(2e12 / 14e9)
    assert inf.decode_tokens_per_s_bound(7e9, 2e12, 2, batch=8) == pytest.approx(8 * 2e12 / 14e9)


def test_table_accelerateurs():
    tab = inf.accelerators_table()
    h = tab.set_index("accélérateur").loc["H100 SXM"]
    assert h["point d'équilibre (FLOP/octet)"] == pytest.approx(989e12 / 3.35e12)


# ------------------------------------------------------------------------- métriques
def test_exemples_du_livre_cout_et_mbu():
    assert inf.cost_per_million_tokens(2.0, 100) == pytest.approx(5.5556, rel=1e-4)
    assert inf.mbu(7e9, 2, 100, 2e12) == pytest.approx(0.7)
    assert inf.mfu(2 * 7e9, 100, 312e12) == pytest.approx(1.4e12 / 312e12)


def test_latence_totale_et_percentiles():
    assert inf.total_latency(0.2, 0.05, 101) == pytest.approx(5.2)
    p = inf.percentiles([100, 102, 100, 100, 99, 104, 110, 90, 3000, 95])
    assert p["p50"] == 100 and p["moyenne"] == 390


def test_goodput():
    thr, good = inf.goodput([0.1, 0.3, 0.1, 0.1], [0.05, 0.05, 0.2, 0.08], 2.0, 0.2, 0.1)
    assert thr == 2.0 and good == 1.0


# --------------------------------------------------------------------- quantification
def test_quantification_erreur_bornee_par_demi_pas():
    x = rng.normal(size=(64, 128))
    for bits in [8, 4]:
        q, n = inf.quantize_symmetric(x, bits)
        step = np.abs(x).max() / (2 ** (bits - 1) - 1)
        assert np.abs(q - x).max() <= step / 2 + 1e-12 and n == 1


def test_granularite_plus_fine_erreur_plus_faible():
    x = rng.normal(size=(64, 256))
    x[:, 3] *= 50  # canal aberrant
    e_t = np.abs(inf.quantize_symmetric(x, 4)[0] - x).mean()
    e_g = np.abs(inf.quantize_symmetric(x, 4, group_size=32)[0] - x).mean()
    assert e_g < e_t
    assert inf.quantize_symmetric(x, 4, axis=0)[1] == 256


def test_bits_effectifs():
    assert inf.quant_bits_per_weight(4, 128) == pytest.approx(4.125)
    assert inf.quant_bits_per_weight(8, None) == 8


def test_masque_2_4():
    w = rng.normal(size=(8, 16))
    m = inf.two_four_mask(w)
    assert m.reshape(-1, 4).sum(1).tolist() == [2] * 32
    g = np.abs(w).reshape(-1, 4)
    kept_min = np.where(m.reshape(-1, 4), g, np.inf).min(1)
    dropped_max = np.where(~m.reshape(-1, 4), g, -np.inf).max(1)
    assert (kept_min >= dropped_max).all()


# ------------------------------------------------------------ décodage spéculatif
def test_tokens_par_tour_et_acceleration():
    assert inf.expected_tokens_per_round(0.0, 4) == pytest.approx(1.0)
    assert inf.expected_tokens_per_round(1.0, 4) == pytest.approx(5.0)
    assert inf.expected_tokens_per_round(0.5, 2) == pytest.approx(1.75)
    # brouillon gratuit et parfait : accélération k + 1
    assert inf.speculative_speedup(1.0, 4, 0.0) == pytest.approx(5.0)
    # brouillon aussi cher que la cible : jamais d'accélération
    assert all(inf.speculative_speedup(a, k, 1.0) <= 1 + 1e-12 for a in [0.5, 0.9] for k in [1, 4])


def test_k_optimal_croit_avec_alpha():
    assert inf.optimal_k(0.95, 0.05) > inf.optimal_k(0.6, 0.05)


def test_estimateur_alpha_geometrique():
    a, k = 0.8, 5
    j = np.minimum(rng.geometric(1 - a, 20000) - 1, k)  # acceptations tronquées à k
    assert inf.estimate_alpha(j, np.full_like(j, k)) == pytest.approx(a, abs=0.01)


def test_prompt_lookup_propose_la_suite():
    prop = inf.prompt_lookup_proposer(ngram=2)
    seq = [1, 2, 3, 4, 5, 9, 2, 3]
    assert prop(seq, 3) == [4, 5, 9]
    assert prop([7, 8, 9], 3) == []


# ------------------------------------------------------------------- cache KV
def test_cache_kv_exemple_du_livre():
    b = inf.kv_cache_bytes_book(32, 2048, 40, 5120, 2)
    assert b / 1e9 == pytest.approx(53.7, rel=0.01)


def test_cache_kv_gqa_fenetre_partage():
    mha = inf.kv_cache_bytes_gqa(1, 8192, 32, 32, 128)
    assert inf.kv_cache_bytes_gqa(1, 8192, 32, 8, 128) == mha / 4
    assert inf.kv_cache_bytes_gqa(1, 8192, 32, 32, 128, window=1024) == mha / 8
    assert inf.kv_cache_bytes_gqa(1, 8192, 32, 32, 128, layer_share=2) == mha / 2
    assert mha == inf.kv_cache_bytes_book(1, 8192, 32, 32 * 128, 2)


def test_allocation_paginee_gaspille_moins():
    L = rng.integers(50, 2000, 100)
    s = inf.kv_allocation_stats(L, 2048, 16)
    assert s["gaspillage paginé"] < 16 * 100 / L.sum() + 1e-9  # au plus un bloc par requête
    assert s["gaspillage contigu"] > s["gaspillage paginé"]
    assert inf.max_concurrent_requests(L, 20_000, block_size=16) > inf.max_concurrent_requests(L, 20_000, max_len=2048)


def test_disposition_paginee_coherente():
    owner, fill, tables = inf.paged_layout([33, 16, 5], 16, 10, seed=0)
    assert [len(tables[r]) for r in range(3)] == [3, 1, 1]
    assert (owner >= 0).sum() == 5
    assert fill[tables[0][-1]] == pytest.approx(1 / 16)


# ----------------------------------------------------------------- batching
@pytest.fixture
def work():
    return inf.make_workload(200, rate=20, seed=0)


def test_simulation_coherente(work):
    for pol in ["none", "static", "dynamic", "continuous"]:
        df = inf.simulate_batching(work, pol, max_batch=8, window=0.05)
        assert df[["debut", "premier_token", "fin"]].notna().all().all()
        assert (df["debut"] >= df["arrivee"] - 1e-12).all()
        assert (df["premier_token"] >= df["debut"]).all() and (df["fin"] >= df["premier_token"]).all()


def test_sans_batching_une_requete_isolee():
    w = pd.DataFrame({"arrivee": [0.0], "prompt": [100], "sortie": [11]})
    c = inf.StepCost(0.01, 0.001, 0.005, 0.0001)
    df = inf.simulate_batching(w, "none", cost=c)
    assert df["latence"].iloc[0] == pytest.approx(c.prefill(100) + 10 * c.decode(1))
    assert df["tpot"].iloc[0] == pytest.approx(c.decode(1))


def test_continu_meilleur_que_statique(work):
    lat = {p: inf.simulate_batching(work, p, max_batch=8)["latence"].median() for p in ["static", "continuous"]}
    assert lat["continuous"] < lat["static"]


def test_dissociation_supprime_les_a_coups():
    w = inf.make_workload(150, rate=6, seed=1, prompt_mean=1500)
    c = inf.StepCost(0.006, 0.0001, 0.005, 0.00003)
    sim = inf.simulate_colocated_vs_disaggregated(w, c, max_batch=16)
    itl_co = sim["colocalisé (2 répliques)"][1]
    itl_dis = sim["dissocié (1 prefill + 1 decode)"][1]
    assert np.percentile(itl_dis, 90) < np.percentile(itl_co, 90)


# ---------------------------------------------------- cache de prompt et parallélisme
def test_cache_de_prompt_rentable_des_deux_requetes():
    one = inf.prompt_cache_savings(1, 10_000, 50, 100, 3, 15)
    two = inf.prompt_cache_savings(2, 10_000, 50, 100, 3, 15)
    assert one["économie"] < 0 < two["économie"]


def test_parallelisme_de_tenseur_exact():
    X, W = rng.normal(size=(5, 12)), rng.normal(size=(12, 9))
    for mode in ["column", "row"]:
        for p in [2, 3]:
            Y, parts = inf.tensor_parallel_matmul(X, W, p, mode)
            assert np.allclose(Y, X @ W) and len(parts) == p


def test_bulle_de_pipeline():
    assert inf.pipeline_bubble_fraction(4, 1) == pytest.approx(0.75)
    assert inf.pipeline_bubble_fraction(1, 8) == 0
    g = inf.pipeline_schedule(4, 6)
    assert g.shape == (4, 9) and (g >= 0).sum() == 24
    assert (g < 0).mean() == pytest.approx(inf.pipeline_bubble_fraction(4, 6))


def test_rangement_glouton():
    placement, free = inf.first_fit_decreasing({"a": 50, "b": 30, "c": 30, "d": 100}, [80, 40])
    assert placement["d"] is None and placement["a"] == 0
    assert sorted(free) == [0, 10]


def test_memoire_de_service():
    gb = inf.serving_memory_gb(7e9, 2, 0.5e6, 4096, 10, overhead=0.0)
    assert gb == pytest.approx(14 + 0.5e6 * 4096 * 10 / 1e9)
