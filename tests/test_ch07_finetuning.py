"""Tests rapides (CPU) du module du chapitre 7."""
import math

import numpy as np
import pytest

from aie import ch07_finetuning as c7


# --- données ---------------------------------------------------------------------------

def test_tickets_format_roundtrip():
    tr = c7.make_tickets(40, seed=1)
    te = c7.make_tickets(12, seed=2, split="test")
    assert len(tr) == 40 and len(te) == 12
    for t in tr + te:
        p = c7.parse_ticket(t.cible)
        assert p == {"categorie": t.categorie, "urgence": t.urgence, "resume": t.resume}
    # les formulations de test ne sont jamais vues à l'entraînement
    assert not {t.resume for t in tr} & {t.resume for t in te}


def test_parse_ticket_rejects():
    assert c7.parse_ticket("La catégorie est livraison.") is None
    assert c7.parse_ticket("Catégorie : cuisine | Urgence : haute | Résumé : x") is None
    assert c7.parse_ticket("") is None
    s = c7.score_tickets(["Catégorie : compte | Urgence : basse | Résumé : x", "bla"],
                         [c7.Ticket("m", "compte", "haute", "x")] * 2)
    assert s["format valide"] == 0.5 and s["catégorie exacte"] == 0.5 and s["urgence exacte"] == 0


def test_recommend_adaptation():
    assert "prompt" in c7.recommend_adaptation(False, "information")
    assert "BM25" in c7.recommend_adaptation(True, "information")
    assert "LoRA" in c7.recommend_adaptation(True, "comportement")
    assert "RAG + finetuning" == c7.recommend_adaptation(True, "les deux", rag_tried=True)


def test_fewshot_cost():
    d = c7.fewshot_vs_finetune_cost(1e6, 1000, 200, 100, 1.0, 2.0)
    assert d["few-shot"] == pytest.approx(1e6 * (1200 * 1 + 100 * 2) / 1e6)
    assert d["finetuné"] < d["few-shot"]


# --- mémoire ---------------------------------------------------------------------------

def test_memory_book_examples():
    # 13 B en 16 bits : 26 Go de poids, 31,2 Go pour l'inférence
    assert c7.inference_memory(13e9, 2) == pytest.approx(31.2e9)
    m = c7.training_memory(13e9, recipe="livre", optimizer="adam")
    assert m["gradients"] + m["états de l'optimiseur"] == pytest.approx(78e9)
    m1 = c7.training_memory(13e9, 1e9)
    assert m1["gradients"] + m1["états de l'optimiseur"] == pytest.approx(6e9)
    mix = c7.training_memory(7e9, recipe="mixte")
    assert mix["total"] == pytest.approx(16 * 7e9)


def test_activation_checkpointing_smaller():
    spec = c7.MODEL_SPECS["Llama 2-7B"]
    full = c7.activation_memory(spec, 2048, 1)
    ck = c7.activation_memory(spec, 2048, 1, checkpointing="complet")
    assert ck < full / 10
    # linéaire en batch
    assert c7.activation_memory(spec, 2048, 4) == pytest.approx(4 * full)


# --- formats -------------------------------------------------------------------------

def test_float_info():
    fp16 = c7.float_info("FP16")
    assert fp16.max == 65504.0 and fp16.eps == 2 ** -10
    bf16 = c7.float_info("BF16")
    assert bf16.max == pytest.approx(3.3895e38, rel=1e-3)
    assert c7.float_info("FP32").bits == 32


def test_round_to_float_matches_torch():
    torch = pytest.importorskip("torch")
    x = np.array([0.0123456789, 0.123456789, 1.23456789, 12.3456789, 1234.56789, 123456.789, -3.3e-6])
    for name, dt in [("FP16", torch.float16), ("BF16", torch.bfloat16)]:
        ref = torch.tensor(x, dtype=torch.float32).to(dt).double().numpy()
        np.testing.assert_allclose(c7.round_to_float(x, name), ref, rtol=0, atol=0)
    assert np.isinf(c7.round_to_float([1e6], "FP16")[0])
    assert c7.round_to_float([1000.0], "FP8 E4M3")[0] == 448.0


def test_float_bits():
    pytest.importorskip("torch")
    s, e, m = c7.float_bits(1.0, "FP32")
    assert s == "0" and e == "01111111" and set(m) == {"0"}
    s, e, m = c7.float_bits(-2.0, "BF16")
    assert s == "1" and e == "10000000" and len(m) == 7


# --- quantification ------------------------------------------------------------------

def test_absmax_and_zeropoint():
    x = np.array([-1.0, -0.5, 0.0, 0.25, 2.0])
    q, s = c7.quantize_absmax(x, 8)
    assert q.max() == 127 and q[2] == 0
    assert np.abs(c7.dequantize_absmax(q, s) - x).max() <= s / 2 + 1e-12
    q, s, z = c7.quantize_zeropoint(x, 8)
    assert q.min() == 0 and q.max() == 255
    assert np.abs(c7.dequantize_zeropoint(q, s, z) - x).max() <= s / 2 + 1e-12


def test_groupwise_improves_with_outlier():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.02, 4096)
    w[7] = 1.0  # valeur aberrante
    e_tensor = c7.quant_error(w, c7.quantize_groupwise(w, 4, 4096))["rmse relative"]
    e_group = c7.quant_error(w, c7.quantize_groupwise(w, 4, 64))["rmse relative"]
    assert e_group < e_tensor / 3
    assert c7.effective_bits(4, 64) == pytest.approx(4.25)


def test_nf4_levels():
    assert len(c7.NF4_LEVELS) == 16 and c7.NF4_LEVELS[7] == 0
    approx = c7.nf4_levels_from_normal()
    np.testing.assert_allclose(approx, c7.NF4_LEVELS, atol=2e-3)
    rng = np.random.default_rng(1)
    w = rng.normal(size=8192)
    e_nf4 = c7.quant_error(w, c7.quantize_groupwise(w, 4, 64, "nf4"))["rmse"]
    e_int4 = c7.quant_error(w, c7.quantize_groupwise(w, 4, 64, "absmax"))["rmse"]
    assert e_nf4 < e_int4


def test_fake_quantize_weight_torch():
    torch = pytest.importorskip("torch")
    w = torch.randn(16, 128)
    for gs in (None, -1, 32):
        r = c7.fake_quantize_weight(w, 8, gs)
        assert r.shape == w.shape and (r - w).abs().max() < 0.05
    r4 = c7.fake_quantize_weight(w, 4, 32, "nf4")
    assert len(torch.unique(r4[0, :32])) <= 16


# --- LoRA ----------------------------------------------------------------------------

def test_lora_counts():
    assert c7.lora_param_count(4096, 4096, 8) == 65536
    gpt3 = {"hidden_size": 12288, "num_attention_heads": 96, "intermediate_size": 4 * 12288,
            "num_hidden_layers": 96}
    assert c7.lora_params_for_model(gpt3, ["q_proj", "k_proj", "v_proj", "o_proj"], 2) == 18_874_368


def test_lora_linear():
    torch = pytest.importorskip("torch")
    LoRALinear = c7.make_lora_linear_class()
    base = torch.nn.Linear(10, 6)
    lay = LoRALinear(base, r=2, alpha=4)
    x = torch.randn(3, 10)
    assert torch.allclose(lay(x), base(x))  # B = 0 au départ
    with torch.no_grad():
        lay.B.normal_()
    merged = lay.merged()
    assert torch.allclose(merged(x), lay(x), atol=1e-5)
    assert sum(p.numel() for p in lay.parameters() if p.requires_grad) == 2 * (10 + 6)


def test_low_rank_and_toy():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(20, 4)) @ rng.normal(size=(4, 20))
    assert np.allclose(c7.low_rank_approx(W, 4), W)
    pytest.importorskip("torch")
    W0 = rng.normal(size=(16, 16))
    D = rng.normal(size=(16, 2)) @ rng.normal(size=(2, 16)) * 0.3
    h = c7.train_lora_toy(W0, W0 + D, r=2, steps=300, lr=3e-2)
    assert h[-1] < 0.05 * h[0]


def test_weighted_loss():
    torch = pytest.importorskip("torch")
    logits = torch.randn(1, 5, 7)
    ids = torch.randint(0, 7, (1, 5))
    w1 = torch.ones(1, 5)
    full = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
    assert torch.allclose(c7.weighted_lm_loss(logits, ids, w1), full)
    w = torch.tensor([[0, 0, 0, 1.0, 1.0]])
    part = torch.nn.functional.cross_entropy(logits[0, 2:4], ids[0, 3:5])
    assert torch.allclose(c7.weighted_lm_loss(logits, ids, w), part)


# --- fusion --------------------------------------------------------------------------

def test_merges():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert np.allclose(c7.linear_merge([a, b]), [0.5, 0.5])
    s = c7.slerp(a, b, 0.5)
    assert np.isclose(np.linalg.norm(s), 1.0) and np.allclose(s, [math.sqrt(0.5)] * 2)
    assert np.allclose(c7.slerp(a, b, 0.0), a) and np.allclose(c7.slerp(a, b, 1.0), b)
    base = np.zeros(2)
    assert np.allclose(c7.task_arithmetic(base, [a, b], [1, -1]), [1, -1])


def test_ties_and_dare():
    t1 = np.array([1.0, -0.1, 0.5, 0.0])
    t2 = np.array([0.8, 0.2, -0.6, 0.05])
    assert np.count_nonzero(c7.trim(t1, 0.5)) == 2
    m = c7.ties_merge([t1, t2], density=1.0)
    # composante 0 : signes d'accord -> moyenne ; composante 2 : signe élu + (0.5 - 0.6 < 0 -> -)
    assert m[0] == pytest.approx(0.9) and m[2] == pytest.approx(-0.6)
    rng = np.random.default_rng(0)
    tv = rng.normal(size=200_000)
    d = c7.dare(tv, 0.9)
    assert abs(d.mean() - tv.mean()) < 0.05 and np.mean(d == 0) == pytest.approx(0.9, abs=0.01)


def test_digits_and_flat():
    pytest.importorskip("torch")
    tasks = c7.digits_tasks()
    Xtr, ytr, Xte, yte = tasks["bruité"]
    assert Xtr.shape[1] == 64 and Xtr.min() >= 0 and Xtr.max() <= 1
    net = c7.make_mlp()
    v = c7.get_flat(net)
    c7.set_flat(net, v * 0)
    assert np.allclose(c7.get_flat(net), 0)


def test_digits_merge_setup():
    pytest.importorskip("torch")
    s = c7.digits_merge_setup(hidden=64, ft_epochs=3)
    assert set(s.task_vectors) == {"bruité", "occulté", "décalé"}
    acc_base = s.evaluate(s.base)
    acc_ft = s.evaluate(s.finetuned["décalé"])
    assert acc_ft["décalé"] > acc_base["décalé"] - 0.02
    assert 0 <= acc_base["original"] <= 1
