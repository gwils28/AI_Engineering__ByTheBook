"""Chapitre 2 : comprendre les modèles de fondation.

Fonctions réutilisables des notebooks du chapitre 2, regroupées par thème :

1. **Données** : statistiques de Common Crawl, corpus parallèle multilingue et coût en
   tokens par langue ;
2. **Attention** : attention par produit scalaire, masque causal, multi-têtes, RoPE,
   implémentées en numpy ;
3. **Paramètres et mémoire** : nombre de paramètres d'un transformer à partir de sa
   configuration, mémoire des poids et du cache KV ;
4. **Échelle** : FLOPs d'entraînement (≈ 6ND), durée et coût, loi de Chinchilla ;
5. **Post-entraînement** : perte de Bradley-Terry, modèles de récompense jouets ;
6. **Échantillonnage** : température, top-k, top-p, min-p, entropie, vote majoritaire ;
7. **Sorties structurées** : décodage contraint par masque de logits (expression régulière).
"""

from __future__ import annotations

import functools
import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import llm

# =====================================================================================
# 1. Données d'entraînement
# =====================================================================================

#: Langues ayant au moins 1 % de Common Crawl (Lai et al., 2023, repris dans le tableau 2-1)
#: et quelques langues sous-représentées (tableau 2-2). Colonnes : code, nom français,
#: locuteurs (millions), part dans Common Crawl (%).
COMMON_CRAWL_LANGS = [
    ("en", "anglais", 1452, 45.8786),
    ("ru", "russe", 258, 5.9692),
    ("de", "allemand", 134, 5.8811),
    ("zh", "chinois", 1118, 4.8747),
    ("ja", "japonais", 125, 4.7884),
    ("fr", "français", 274, 4.7254),
    ("es", "espagnol", 548, 4.4690),
    ("it", "italien", 68, 2.5712),
    ("nl", "néerlandais", 30, 2.0585),
    ("pl", "polonais", 45, 1.6636),
    ("pt", "portugais", 257, 1.1505),
    ("vi", "vietnamien", 85, 1.0299),
    ("bn", "bengali", 272, 0.0930),
    ("ur", "ourdou", 231, 0.0274),
    ("mr", "marathi", 99, 0.0213),
    ("te", "télougou", 95, 0.0183),
    ("gu", "gujarati", 62, 0.0126),
    ("kn", "kannada", 64, 0.0122),
    ("sw", "swahili", 71, 0.0077),
    ("pa", "pendjabi", 113, 0.0061),
]

WORLD_POPULATION_M = 8000


def common_crawl_table():
    """Tableau pandas des langues : part de la population, part de Common Crawl et ratio."""
    import pandas as pd

    df = pd.DataFrame(COMMON_CRAWL_LANGS, columns=["code", "langue", "locuteurs_M", "cc_pct"])
    df["pop_pct"] = 100 * df["locuteurs_M"] / WORLD_POPULATION_M
    df["ratio_pop_cc"] = df["pop_pct"] / df["cc_pct"]
    return df


#: Petit corpus parallèle rédigé à la main : les mêmes six phrases dans 18 langues.
PARALLEL_CORPUS = {
    "en": ["The weather is very nice today.", "Where is the nearest train station?",
           "I would like to order a coffee, please.", "My brother works in a hospital in the city.",
           "Can you help me understand this document?", "We will meet tomorrow morning at nine o'clock."],
    "fr": ["Il fait très beau aujourd'hui.", "Où est la gare la plus proche ?",
           "Je voudrais commander un café, s'il vous plaît.", "Mon frère travaille dans un hôpital en ville.",
           "Peux-tu m'aider à comprendre ce document ?", "Nous nous retrouverons demain matin à neuf heures."],
    "es": ["Hoy hace muy buen tiempo.", "¿Dónde está la estación de tren más cercana?",
           "Quisiera pedir un café, por favor.", "Mi hermano trabaja en un hospital de la ciudad.",
           "¿Puedes ayudarme a entender este documento?", "Nos veremos mañana por la mañana a las nueve."],
    "de": ["Das Wetter ist heute sehr schön.", "Wo ist der nächste Bahnhof?",
           "Ich möchte bitte einen Kaffee bestellen.", "Mein Bruder arbeitet in einem Krankenhaus in der Stadt.",
           "Kannst du mir helfen, dieses Dokument zu verstehen?", "Wir treffen uns morgen früh um neun Uhr."],
    "it": ["Oggi il tempo è molto bello.", "Dov'è la stazione ferroviaria più vicina?",
           "Vorrei ordinare un caffè, per favore.", "Mio fratello lavora in un ospedale in città.",
           "Puoi aiutarmi a capire questo documento?", "Ci vediamo domani mattina alle nove."],
    "pt": ["O tempo está muito bom hoje.", "Onde fica a estação de trem mais próxima?",
           "Eu gostaria de pedir um café, por favor.", "Meu irmão trabalha em um hospital na cidade.",
           "Você pode me ajudar a entender este documento?", "Vamos nos encontrar amanhã de manhã às nove horas."],
    "tr": ["Bugün hava çok güzel.", "En yakın tren istasyonu nerede?",
           "Bir kahve sipariş etmek istiyorum, lütfen.", "Erkek kardeşim şehirdeki bir hastanede çalışıyor.",
           "Bu belgeyi anlamama yardım edebilir misin?", "Yarın sabah saat dokuzda buluşacağız."],
    "vi": ["Hôm nay thời tiết rất đẹp.", "Ga tàu gần nhất ở đâu?",
           "Tôi muốn gọi một ly cà phê, làm ơn.", "Anh trai tôi làm việc ở một bệnh viện trong thành phố.",
           "Bạn có thể giúp tôi hiểu tài liệu này không?", "Chúng ta sẽ gặp nhau vào chín giờ sáng mai."],
    "sw": ["Hali ya hewa ni nzuri sana leo.", "Kituo cha treni kilicho karibu zaidi kiko wapi?",
           "Ningependa kuagiza kahawa, tafadhali.", "Kaka yangu anafanya kazi katika hospitali mjini.",
           "Je, unaweza kunisaidia kuelewa hati hii?", "Tutakutana kesho asubuhi saa tatu."],
    "ru": ["Сегодня очень хорошая погода.", "Где находится ближайший вокзал?",
           "Я хотел бы заказать кофе, пожалуйста.", "Мой брат работает в больнице в городе.",
           "Ты можешь помочь мне понять этот документ?", "Мы встретимся завтра утром в девять часов."],
    "el": ["Ο καιρός είναι πολύ ωραίος σήμερα.", "Πού είναι ο πλησιέστερος σιδηροδρομικός σταθμός;",
           "Θα ήθελα να παραγγείλω έναν καφέ, παρακαλώ.", "Ο αδελφός μου δουλεύει σε ένα νοσοκομείο στην πόλη.",
           "Μπορείς να με βοηθήσεις να καταλάβω αυτό το έγγραφο;", "Θα συναντηθούμε αύριο το πρωί στις εννέα."],
    "ar": ["الطقس جميل جدا اليوم.", "أين أقرب محطة قطار؟", "أود أن أطلب قهوة من فضلك.",
           "أخي يعمل في مستشفى في المدينة.", "هل يمكنك مساعدتي في فهم هذه الوثيقة؟",
           "سنلتقي غدا صباحا في الساعة التاسعة."],
    "hi": ["आज मौसम बहुत अच्छा है।", "सबसे नज़दीकी रेलवे स्टेशन कहाँ है?",
           "मैं एक कॉफ़ी ऑर्डर करना चाहूँगा, कृपया।", "मेरा भाई शहर के एक अस्पताल में काम करता है।",
           "क्या आप इस दस्तावेज़ को समझने में मेरी मदद कर सकते हैं?", "हम कल सुबह नौ बजे मिलेंगे।"],
    "bn": ["আজ আবহাওয়া খুব সুন্দর।", "নিকটতম রেলওয়ে স্টেশন কোথায়?",
           "আমি একটি কফি অর্ডার করতে চাই, দয়া করে।", "আমার ভাই শহরের একটি হাসপাতালে কাজ করে।",
           "আপনি কি আমাকে এই নথিটি বুঝতে সাহায্য করতে পারবেন?", "আমরা আগামীকাল সকাল নয়টায় দেখা করব।"],
    "th": ["วันนี้อากาศดีมาก", "สถานีรถไฟที่ใกล้ที่สุดอยู่ที่ไหน", "ฉันอยากสั่งกาแฟหนึ่งแก้วค่ะ",
           "พี่ชายของฉันทำงานที่โรงพยาบาลในเมือง", "คุณช่วยให้ฉันเข้าใจเอกสารนี้ได้ไหม",
           "เราจะพบกันพรุ่งนี้เช้าเวลาเก้าโมง"],
    "zh": ["今天天气很好。", "最近的火车站在哪里？", "我想点一杯咖啡，谢谢。",
           "我哥哥在城里的一家医院工作。", "你能帮我理解这份文件吗？", "我们明天早上九点见面。"],
    "ja": ["今日はとても良い天気です。", "一番近い駅はどこですか？", "コーヒーを一杯注文したいです、お願いします。",
           "兄は町の病院で働いています。", "この書類を理解するのを手伝ってくれますか？", "明日の朝九時に会いましょう。"],
    "ko": ["오늘 날씨가 아주 좋습니다.", "가장 가까운 기차역이 어디에 있나요?", "커피 한 잔 주문하고 싶어요.",
           "제 형은 시내에 있는 병원에서 일해요.", "이 문서를 이해하도록 도와줄 수 있나요?",
           "우리는 내일 아침 아홉 시에 만날 거예요."],
}

LANG_NAMES = {
    "en": "anglais", "fr": "français", "es": "espagnol", "de": "allemand", "it": "italien",
    "pt": "portugais", "tr": "turc", "vi": "vietnamien", "sw": "swahili", "ru": "russe",
    "el": "grec", "ar": "arabe", "hi": "hindi", "bn": "bengali", "th": "thaï", "zh": "chinois",
    "ja": "japonais", "ko": "coréen",
}


def tokens_per_language(encoders: dict, corpus: dict | None = None):
    """Nombre de tokens nécessaires pour le même contenu dans chaque langue.

    encoders : {nom: fonction texte -> liste d'identifiants}.
    Renvoie un DataFrame (langue, tokenizer, tokens, caractères, ratio_vs_en) où
    ratio_vs_en est le nombre de tokens rapporté à celui de l'anglais (même tokenizer).
    """
    import pandas as pd

    corpus = corpus or PARALLEL_CORPUS
    rows = []
    for name, enc in encoders.items():
        n_en = sum(len(enc(s)) for s in corpus["en"])
        for lang, sents in corpus.items():
            n = sum(len(enc(s)) for s in sents)
            rows.append(dict(code=lang, langue=LANG_NAMES.get(lang, lang), tokenizer=name, tokens=n,
                             caracteres=sum(len(s) for s in sents), ratio_vs_en=n / n_en))
    return pd.DataFrame(rows)


# =====================================================================================
# 2. Attention
# =====================================================================================


def causal_mask(T: int) -> np.ndarray:
    """Masque booléen (T, T) : True là où la position i peut regarder la position j (j <= i)."""
    return np.tril(np.ones((T, T), dtype=bool))


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V.

    Q : (..., Tq, d), K : (..., Tk, d), V : (..., Tk, dv). mask : booléen diffusable vers
    (..., Tq, Tk), True = autorisé. Renvoie (sortie, poids d'attention).
    """
    Q, K, V = (np.asarray(a, dtype=np.float64) for a in (Q, K, V))
    d = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = llm.softmax(scores)
    return weights @ V, weights


def split_heads(x, n_heads: int):
    """(T, d) -> (n_heads, T, d / n_heads)."""
    T, d = x.shape
    return x.reshape(T, n_heads, d // n_heads).transpose(1, 0, 2)


def merge_heads(x):
    """(n_heads, T, dh) -> (T, n_heads * dh)."""
    h, T, dh = x.shape
    return x.transpose(1, 0, 2).reshape(T, h * dh)


def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads: int, bq=None, bk=None, bv=None, bo=None,
                         causal: bool = True):
    """Attention multi-têtes (convention « x @ W ») sur une séquence x de forme (T, d).

    Renvoie (sortie (T, d), poids d'attention (n_heads, T, T)).
    """
    x = np.asarray(x, dtype=np.float64)

    def proj(W, b):
        y = x @ W
        return y + b if b is not None else y

    Q, K, V = (split_heads(proj(W, b), n_heads) for W, b in ((Wq, bq), (Wk, bk), (Wv, bv)))
    mask = causal_mask(x.shape[0]) if causal else None
    out, w = scaled_dot_product_attention(Q, K, V, mask)
    out = merge_heads(out) @ Wo
    if bo is not None:
        out = out + bo
    return out, w


def rope(x, positions, theta: float = 10000.0):
    """Applique les plongements de position rotatifs (RoPE, convention « rotate_half »).

    x : (..., T, d) avec d pair ; positions : (T,). La paire (x_i, x_{i + d/2}) de chaque
    vecteur est tournée d'un angle position * theta^(-2i/d).
    """
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    inv_freq = 1.0 / theta ** (np.arange(0, d, 2) / d)
    ang = np.outer(np.asarray(positions, dtype=np.float64), inv_freq)  # (T, d/2)
    cos, sin = np.cos(np.concatenate([ang, ang], -1)), np.sin(np.concatenate([ang, ang], -1))
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rotated = np.concatenate([-x2, x1], -1)
    return x * cos + rotated * sin


@functools.lru_cache(maxsize=2)
def load_eager(model_id: str = llm.BASE, device: str = "cpu"):
    """Charge un modèle en float32 avec l'attention « eager » (pour récupérer les poids d'attention).

    Par défaut sur CPU : les textes analysés sont courts, et cela laisse le GPU libre.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                                 attn_implementation="eager").to(device)
    model.eval()
    return tok, model


def attention_maps(text: str, model_id: str = llm.BASE):
    """Poids d'attention réels d'un modèle : (tokens, tableau (couches, têtes, T, T))."""
    import torch

    tok, model = load_eager(model_id)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    att = torch.stack(out.attentions)[:, 0].float().cpu().numpy()
    return [tok.decode([i]) for i in inputs["input_ids"][0].tolist()], att


def attention_matrix_bytes(seq_len: int, n_heads: int, n_layers: int = 1, batch: int = 1,
                           bytes_per: int = 2) -> float:
    """Mémoire des matrices de scores d'attention matérialisées (T x T par tête et par couche)."""
    return float(batch) * n_layers * n_heads * seq_len ** 2 * bytes_per


# =====================================================================================
# 3. Paramètres et mémoire
# =====================================================================================


@dataclass
class TransformerConfig:
    """Hyperparamètres qui déterminent la taille d'un transformer décodeur.

    gated_mlp=True : MLP de type SwiGLU (3 matrices d x d_ff : gate, up, down), comme Llama
    et Qwen ; False : MLP classique à 2 matrices (GPT-2). n_kv_heads < n_heads : attention à
    requêtes groupées (GQA). learned_pos > 0 : nombre de positions apprises (GPT-2) ; 0 pour
    RoPE (aucun paramètre). n_experts > 1 : mélange d'experts (MoE) dont top_k sont actifs.
    """

    name: str
    d_model: int
    n_layers: int
    d_ff: int
    vocab: int
    n_heads: int
    n_kv_heads: int | None = None
    gated_mlp: bool = True
    tied: bool = False
    qkv_bias: bool = False
    attn_out_bias: bool = False
    mlp_bias: bool = False
    norm_bias: bool = False
    learned_pos: int = 0
    n_experts: int = 1
    top_k: int = 1
    context: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads


def param_breakdown(cfg: TransformerConfig, active: bool = False) -> dict:
    """Nombre de paramètres par composant. active=True : ne compte que les experts actifs (MoE)."""
    d, L, f, V = cfg.d_model, cfg.n_layers, cfg.d_ff, cfg.vocab
    dh, hkv = cfg.head_dim, cfg.kv_heads
    norm = (2 * d) if cfg.norm_bias else d  # gain (+ biais pour LayerNorm)
    emb = V * d
    pos = cfg.learned_pos * d
    q = d * d + (d if cfg.qkv_bias else 0)
    kv = 2 * (d * hkv * dh + (hkv * dh if cfg.qkv_bias else 0))
    o = d * d + (d if cfg.attn_out_bias else 0)
    attn = L * (q + kv + o)
    n_mat = 3 if cfg.gated_mlp else 2
    one_expert = n_mat * d * f + ((n_mat - 1) * f + d if cfg.mlp_bias else 0)
    n_exp = cfg.top_k if (active and cfg.n_experts > 1) else cfg.n_experts
    router = d * cfg.n_experts if cfg.n_experts > 1 else 0
    mlp = L * (n_exp * one_expert + router)
    norms = L * 2 * norm + norm  # deux normalisations par bloc + une finale
    head = 0 if cfg.tied else V * d
    return {"embeddings": emb, "positions": pos, "attention": attn, "mlp": mlp,
            "normalisations": norms, "tête de sortie": head}


def count_params_config(cfg: TransformerConfig, active: bool = False) -> int:
    """Nombre total de paramètres d'un transformer décodeur, calculé à partir de sa configuration."""
    return int(sum(param_breakdown(cfg, active).values()))


def config_from_hf(hf_config, name: str | None = None) -> TransformerConfig:
    """Convertit une configuration Hugging Face (GPT-2, Llama, Qwen2, Mistral, Mixtral) en TransformerConfig."""
    c = hf_config
    mt = c.model_type
    if mt == "gpt2":
        return TransformerConfig(name or "gpt2", c.n_embd, c.n_layer, c.n_inner or 4 * c.n_embd, c.vocab_size,
                                 c.n_head, gated_mlp=False, tied=True, qkv_bias=True, attn_out_bias=True,
                                 mlp_bias=True, norm_bias=True, learned_pos=c.n_positions, context=c.n_positions)
    return TransformerConfig(
        name or mt, c.hidden_size, c.num_hidden_layers, c.intermediate_size, c.vocab_size,
        c.num_attention_heads, getattr(c, "num_key_value_heads", None), gated_mlp=True,
        tied=bool(getattr(c, "tie_word_embeddings", False)), qkv_bias=(mt == "qwen2"),
        n_experts=getattr(c, "num_local_experts", 1) or 1, top_k=getattr(c, "num_experts_per_tok", 1) or 1,
        context=getattr(c, "max_position_embeddings", None))


#: Configurations publiques (fichiers config.json officiels) de quelques modèles connus.
KNOWN_CONFIGS = {
    "GPT-2 (124M)": TransformerConfig("GPT-2 (124M)", 768, 12, 3072, 50257, 12, gated_mlp=False, tied=True,
                                      qkv_bias=True, attn_out_bias=True, mlp_bias=True, norm_bias=True,
                                      learned_pos=1024, context=1024),
    "GPT-2 XL (1.5B)": TransformerConfig("GPT-2 XL (1.5B)", 1600, 48, 6400, 50257, 25, gated_mlp=False,
                                         tied=True, qkv_bias=True, attn_out_bias=True, mlp_bias=True,
                                         norm_bias=True, learned_pos=1024, context=1024),
    "Qwen2.5-0.5B": TransformerConfig("Qwen2.5-0.5B", 896, 24, 4864, 151936, 14, 2, tied=True, qkv_bias=True,
                                      context=32768),
    "Qwen2.5-1.5B": TransformerConfig("Qwen2.5-1.5B", 1536, 28, 8960, 151936, 12, 2, tied=True, qkv_bias=True,
                                      context=32768),
    "Llama 2-7B": TransformerConfig("Llama 2-7B", 4096, 32, 11008, 32000, 32, 32, context=4096),
    "Llama 2-13B": TransformerConfig("Llama 2-13B", 5120, 40, 13824, 32000, 40, 40, context=4096),
    "Llama 2-70B": TransformerConfig("Llama 2-70B", 8192, 80, 28672, 32000, 64, 8, context=4096),
    "Llama 3-8B": TransformerConfig("Llama 3-8B", 4096, 32, 14336, 128256, 32, 8, context=8192),
    "Llama 3-70B": TransformerConfig("Llama 3-70B", 8192, 80, 28672, 128256, 64, 8, context=8192),
    "Llama 3.1-405B": TransformerConfig("Llama 3.1-405B", 16384, 126, 53248, 128256, 128, 8, context=131072),
    "Mixtral 8x7B": TransformerConfig("Mixtral 8x7B", 4096, 32, 14336, 32000, 32, 8, n_experts=8, top_k=2,
                                      context=32768),
}


def count_params_safetensors(model_id: str) -> int:
    """Compte les paramètres d'un modèle du cache HF en lisant seulement les en-têtes safetensors."""
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"], local_files_only=True))
    total = 0
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), framework="numpy") as h:
            for k in h.keys():
                total += int(np.prod(h.get_slice(k).get_shape()))
    return total


def weights_memory_gb(n_params: float, bytes_per: float = 2) -> float:
    """Mémoire des poids (Go, 1 Go = 1e9 octets) pour un nombre de paramètres et une précision donnés."""
    return n_params * bytes_per / 1e9


def kv_cache_bytes(cfg: TransformerConfig, seq_len: int, batch: int = 1, bytes_per: int = 2) -> float:
    """Taille du cache KV : 2 (K et V) x couches x têtes KV x dim. de tête x tokens x batch x octets."""
    return 2.0 * cfg.n_layers * cfg.kv_heads * cfg.head_dim * seq_len * batch * bytes_per


# =====================================================================================
# 4. Échelle : FLOPs, coût, Chinchilla
# =====================================================================================


def training_flops(n_params: float, n_tokens: float) -> float:
    """FLOPs d'entraînement ≈ 6 N D (2 N D pour la passe avant, 4 N D pour la rétropropagation)."""
    return 6.0 * n_params * n_tokens


def training_days(flops: float, n_gpus: int, peak_flops_per_s: float, utilization: float = 1.0) -> float:
    """Durée d'entraînement (jours) sur n_gpus accélérateurs utilisés à `utilization` de leur pic."""
    return flops / (n_gpus * peak_flops_per_s * utilization) / 86400.0


def training_cost(flops: float, peak_flops_per_s: float, price_per_gpu_hour: float,
                  utilization: float = 1.0) -> float:
    """Coût (dollars) : heures-GPU nécessaires x prix horaire."""
    gpu_hours = flops / (peak_flops_per_s * utilization) / 3600.0
    return gpu_hours * price_per_gpu_hour


#: Ajustement paramétrique de Hoffmann et al. (2022, « approche 3 ») :
#: L(N, D) = E + A / N^alpha + B / D^beta (perte en nats par token).
CHINCHILLA = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)


def chinchilla_loss(N, D, E=CHINCHILLA["E"], A=CHINCHILLA["A"], B=CHINCHILLA["B"],
                    alpha=CHINCHILLA["alpha"], beta=CHINCHILLA["beta"]):
    """Perte prédite pour N paramètres entraînés sur D tokens."""
    N, D = np.asarray(N, dtype=float), np.asarray(D, dtype=float)
    return E + A / N ** alpha + B / D ** beta


def chinchilla_optimal(C: float, method: str = "20x") -> tuple[float, float]:
    """Taille de modèle N et nombre de tokens D optimaux pour un budget de C FLOPs.

    method="20x" : règle empirique D = 20 N avec C = 6 N D, soit N = sqrt(C / 120).
    method="parametric" : minimise exactement chinchilla_loss(N, C / (6N)).
    """
    if method == "20x":
        N = math.sqrt(C / 120.0)
        return N, 20.0 * N
    a, b = CHINCHILLA["alpha"], CHINCHILLA["beta"]
    A, B = CHINCHILLA["A"], CHINCHILLA["B"]
    G = (a * A / (b * B)) ** (1.0 / (a + b))
    N = G * (C / 6.0) ** (b / (a + b))
    return N, C / (6.0 * N)


#: Quelques modèles : (nom, année, paramètres, tokens d'entraînement). Sources : articles
#: et fiches techniques publiques (tableau 2-5 du livre pour les cinq premiers).
MODELS_SCALE = [
    ("LaMDA", 2022, 137e9, 168e9),
    ("GPT-3", 2020, 175e9, 300e9),
    ("Jurassic", 2021, 178e9, 300e9),
    ("Gopher", 2021, 280e9, 300e9),
    ("MT-NLG", 2022, 530e9, 270e9),
    ("Chinchilla", 2022, 70e9, 1.4e12),
    ("Llama 1-65B", 2023, 65e9, 1.4e12),
    ("Llama 2-70B", 2023, 70e9, 2e12),
    ("Llama 3-8B", 2024, 8e9, 15e12),
    ("Llama 3-70B", 2024, 70e9, 15e12),
    ("Qwen2.5-0.5B", 2024, 0.49e9, 18e12),
]


# =====================================================================================
# 5. Post-entraînement : préférences et modèle de récompense
# =====================================================================================


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1 / (1 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1 + np.exp(-np.abs(x))))


def log_sigmoid(x):
    """log(sigmoid(x)) numériquement stable."""
    x = np.asarray(x, dtype=float)
    return -np.logaddexp(0.0, -x)


def bradley_terry_loss(s_w, s_l) -> float:
    """Perte du modèle de récompense d'InstructGPT : - moyenne de log sigma(s_w - s_l)."""
    return float(-np.mean(log_sigmoid(np.asarray(s_w) - np.asarray(s_l))))


def fit_linear_reward(X_w, X_l, lr: float = 0.5, epochs: int = 300, l2: float = 1e-3):
    """Ajuste un modèle de récompense linéaire r(x) = w . x sur des paires (gagnant, perdant).

    Descente de gradient sur la perte de Bradley-Terry. Renvoie (w, historique des pertes).
    """
    X_w, X_l = np.asarray(X_w, float), np.asarray(X_l, float)
    diff = X_w - X_l
    w = np.zeros(diff.shape[1])
    hist = []
    for _ in range(epochs):
        m = diff @ w
        hist.append(bradley_terry_loss(m, 0.0) + 0.5 * l2 * w @ w)
        grad = -(diff * (1 - sigmoid(m))[:, None]).mean(0) + l2 * w
        w -= lr * grad
    return w, np.array(hist)


def fit_bradley_terry(winners, losers, n_items: int, lr: float = 0.5, epochs: int = 500, l2: float = 1e-3):
    """Forces latentes s_i de n_items concurrents à partir de duels (gagnant, perdant).

    P(i bat j) = sigma(s_i - s_j). Les forces sont centrées (moyenne nulle), car seules
    les différences sont identifiables. Renvoie (forces, historique des pertes).
    """
    winners, losers = np.asarray(winners), np.asarray(losers)
    s = np.zeros(n_items)
    hist = []
    for _ in range(epochs):
        m = s[winners] - s[losers]
        hist.append(bradley_terry_loss(m, 0.0))
        g = 1 - sigmoid(m)
        grad = np.zeros(n_items)
        np.add.at(grad, winners, -g)
        np.add.at(grad, losers, g)
        grad = grad / len(winners) + l2 * s
        s -= lr * grad * n_items
        s -= s.mean()
    return s, np.array(hist)


def kl_regularized_policy(p_ref, rewards, beta: float) -> np.ndarray:
    """Solution exacte de max_pi E_pi[r] - beta KL(pi || p_ref) : pi* ∝ p_ref exp(r / beta)."""
    p_ref = np.asarray(p_ref, float)
    return llm.softmax(np.log(p_ref) + np.asarray(rewards, float) / beta)


# =====================================================================================
# 6. Échantillonnage
# =====================================================================================


def entropy(p, base: float = math.e) -> float:
    """Entropie H(p) = - sum p log p (en nats par défaut ; base=2 pour des bits)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(base))


def top_k_filter(logits, k: int) -> np.ndarray:
    """Garde les k plus grands logits, met les autres à -inf."""
    z = np.asarray(logits, dtype=float).copy()
    if k >= z.size:
        return z
    kth = np.partition(z, -k)[-k]
    z[z < kth] = -np.inf
    return z


def top_p_filter(logits, p: float, temperature: float = 1.0) -> np.ndarray:
    """Échantillonnage du noyau : garde le plus petit ensemble de tokens de probabilité cumulée >= p.

    Le token qui fait franchir le seuil est inclus (le plus probable est toujours gardé).
    """
    z = np.asarray(logits, dtype=float)
    probs = llm.softmax(z, temperature)
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    n_keep = int(np.searchsorted(cum, p - 1e-9) + 1)  # premier rang où la somme atteint p
    n_keep = min(max(n_keep, 1), z.size)
    out = np.full_like(z, -np.inf)
    keep = order[:n_keep]
    out[keep] = z[keep]
    return out


def min_p_filter(logits, min_p: float, temperature: float = 1.0) -> np.ndarray:
    """Garde les tokens dont la probabilité vaut au moins min_p x (probabilité du meilleur)."""
    z = np.asarray(logits, dtype=float)
    probs = llm.softmax(z, temperature)
    out = z.copy()
    out[probs < min_p * probs.max()] = -np.inf
    return out


def sampling_distribution(logits, temperature: float = 1.0, top_k: int | None = None,
                          top_p: float | None = None, min_p: float | None = None) -> np.ndarray:
    """Distribution effectivement échantillonnée après température, top-k, top-p et min-p.

    temperature = 0 : distribution de Dirac sur l'argmax (décodage glouton).
    """
    z = np.asarray(logits, dtype=float)
    if temperature == 0:
        p = np.zeros_like(z)
        p[np.argmax(z)] = 1.0
        return p
    z = z / temperature
    if top_k is not None:
        z = top_k_filter(z, top_k)
    if top_p is not None:
        z = top_p_filter(z, top_p)
    if min_p is not None:
        z = min_p_filter(z, min_p)
    return llm.softmax(z)


def sample_token(logits, rng=None, **kw) -> int:
    """Tire un identifiant de token selon sampling_distribution(logits, **kw)."""
    rng = rng if rng is not None else np.random.default_rng()
    p = sampling_distribution(logits, **kw)
    return int(rng.choice(p.size, p=p))


def generate_custom(prompt: str, model_id: str = llm.SMALL, max_new_tokens: int = 30, chat_format: bool = True,
                    seed: int = 0, stop_at_eos: bool = True, **sampling_kw):
    """Boucle de génération écrite à la main (avec cache KV), échantillonnant via sample_token.

    Renvoie (texte généré, liste des logprobs des tokens choisis sous la distribution du modèle à T=1).
    """
    import torch

    tok, model = llm.load(model_id)
    if chat_format:
        ids = llm._chat_inputs(tok, prompt)["input_ids"].to(model.device)
    else:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
    rng = np.random.default_rng(seed)
    out_ids, logps, past = [], [], None
    cur = ids
    eos = {tok.eos_token_id, getattr(tok, "pad_token_id", None)} | set(_eos_ids(tok))
    with torch.no_grad():
        for _ in range(max_new_tokens):
            o = model(cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            z = o.logits[0, -1].float().cpu().numpy()
            t = sample_token(z, rng, **sampling_kw)
            logps.append(float(np.log(llm.softmax(z)[t] + 1e-300)))
            if stop_at_eos and t in eos:
                break
            out_ids.append(t)
            cur = torch.tensor([[t]], device=model.device)
    return tok.decode(out_ids, skip_special_tokens=True), np.array(logps)


def _eos_ids(tok) -> list[int]:
    ids = []
    for s in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.append(i)
    return ids


def sequence_logprob(token_logprobs, normalize: bool = False) -> float:
    """Logprob d'une séquence : somme des logprobs des tokens (moyenne si normalize=True)."""
    lp = np.asarray(token_logprobs, dtype=float)
    return float(lp.mean() if normalize else lp.sum())


_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def extract_number(text: str):
    """Dernier nombre apparaissant dans un texte (None s'il n'y en a pas)."""
    nums = _NUM_RE.findall(text.replace(" ", "").replace(",", "") if text else "")
    if not nums:
        return None
    v = float(nums[-1])
    return int(v) if v.is_integer() else v


def majority_vote(answers):
    """Réponse la plus fréquente (en ignorant None) et sa fréquence relative parmi les réponses valides."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0.0
    ans, n = Counter(valid).most_common(1)[0]
    return ans, n / len(valid)


def majority_vote_accuracy(p: float, n: int) -> float:
    """Probabilité qu'un vote majoritaire sur n tirages indépendants soit correct (cas binaire).

    Chaque tirage est correct avec probabilité p ; égalité (n pair) tranchée à pile ou face.
    """
    from math import comb

    acc = 0.0
    for k in range(n + 1):
        pk = comb(n, k) * p ** k * (1 - p) ** (n - k)
        if 2 * k > n:
            acc += pk
        elif 2 * k == n:
            acc += 0.5 * pk
    return acc


def pass_at_n(p: float, n: int) -> float:
    """Probabilité qu'au moins un des n tirages soit correct (vérificateur parfait)."""
    return 1 - (1 - p) ** n


# =====================================================================================
# 7. Sorties structurées : décodage contraint
# =====================================================================================


@functools.lru_cache(maxsize=4)
def vocab_strings(model_id: str = llm.SMALL) -> tuple[str, ...]:
    """Texte (décodé) de chaque token du vocabulaire, mis en cache."""
    tok, model = llm.load(model_id)
    n = model.get_output_embeddings().weight.shape[0]
    return tuple(tok.decode([i]) if i < len(tok) else "" for i in range(n))


def regex_allows(pattern: str, text: str, partial: bool = True) -> bool:
    """True si `text` est un préfixe valide (partial=True) ou une correspondance complète du motif."""
    import regex

    return regex.fullmatch(pattern, text, partial=partial) is not None


@dataclass
class ConstrainedStep:
    token: str
    n_valid: int
    n_checked: int
    mass_valid: float
    top_unconstrained: str


def constrained_generate(prompt, pattern: str, model_id: str = llm.SMALL, max_new_tokens: int = 60,
                         top_n: int = 400, chat_format: bool = True, system: str | None = None,
                         temperature: float = 0.0, seed: int = 0):
    """Génère un texte qui respecte une expression régulière, par masquage des logits.

    À chaque pas : on examine les `top_n` tokens les plus probables, on interdit (logit = -inf)
    ceux qui feraient sortir le texte du langage décrit par `pattern` (test de préfixe partiel
    avec le module `regex`), puis on choisit parmi les tokens restants (glouton si temperature=0).
    Le token de fin n'est autorisé que lorsque le texte correspond entièrement au motif.
    Renvoie (texte, liste de ConstrainedStep).
    """
    import torch

    tok, model = llm.load(model_id)
    vs = vocab_strings(model_id)
    if chat_format:
        msgs = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        ids = llm._chat_inputs(tok, msgs)["input_ids"].to(model.device)
    else:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
    eos = set(_eos_ids(tok)) | {tok.eos_token_id}
    rng = np.random.default_rng(seed)
    text, steps, past, cur = "", [], None, ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            o = model(cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            z = o.logits[0, -1].float().cpu().numpy()
            probs = llm.softmax(z)
            order = np.argsort(-z)[:top_n]
            done = regex_allows(pattern, text, partial=False)
            valid = []
            for t in order:
                t = int(t)
                if t in eos:
                    ok = done
                else:
                    s = vs[t]
                    ok = bool(s) and regex_allows(pattern, text + s)
                if ok:
                    valid.append(t)
            if not valid:  # aucun candidat valide parmi les top_n : on s'arrête si le texte est complet
                break
            zv = z[valid]
            if temperature > 0:
                pv = llm.softmax(zv, temperature)
                t = valid[int(rng.choice(len(valid), p=pv))]
            else:
                t = valid[int(np.argmax(zv))]
            steps.append(ConstrainedStep(vs[t] if t not in eos else "<fin>", len(valid), len(order),
                                         float(probs[valid].sum()), vs[int(order[0])]))
            if t in eos:
                break
            text += vs[t]
            cur = torch.tensor([[t]], device=model.device)
    return text, steps


def label_logprobs(prompt, labels, model_id: str = llm.SMALL, chat_format: bool = True,
                   system: str | None = None) -> dict:
    """Logprob totale de chaque étiquette candidate comme réponse au prompt (classification contrainte)."""
    import torch

    tok, model = llm.load(model_id)
    if chat_format:
        msgs = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        prefix = llm._chat_inputs(tok, msgs)["input_ids"][0].tolist()
    else:
        prefix = tok(prompt)["input_ids"]
    out = {}
    for lab in labels:
        lab_ids = tok(lab, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prefix + lab_ids], device=model.device)
        with torch.no_grad():
            lp = torch.log_softmax(model(ids).logits[0].float(), -1)
        pos = torch.arange(len(prefix) - 1, len(prefix) - 1 + len(lab_ids), device=model.device)
        out[lab] = float(lp[pos, torch.tensor(lab_ids, device=model.device)].sum())
    return out


def repair_json(text: str):
    """Post-traitement « défensif » : extrait le premier objet JSON d'un texte et tente de le réparer.

    Corrige quelques fautes courantes (balises de code, guillemets simples, virgule finale,
    accolades non fermées). Renvoie l'objet Python ou None.
    """
    import json

    if text is None:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    start = s.find("{")
    if start < 0:
        return None
    s = s[start:]
    end = s.rfind("}")
    candidates = [s[: end + 1]] if end >= 0 else []
    candidates.append(s)
    for c in candidates:
        for fix in (lambda x: x, lambda x: x.replace("'", '"'), lambda x: re.sub(r",\s*([}\]])", r"\1", x)):
            t = fix(c)
            t = t + "}" * max(0, t.count("{") - t.count("}"))
            try:
                return json.loads(t)
            except Exception:
                continue
    return None


# =====================================================================================
# Utilitaire graphique
# =====================================================================================


def plain_log_axis(ax, axis: str = "x", fmt=None) -> None:
    """Étiquettes lisibles sur un axe logarithmique (évite la notation 10^-k en mathtext).

    fmt : fonction valeur -> texte (par défaut, notation courte « 0.01 », « 10 », « 1e+21 »).
    """
    import matplotlib.ticker as mticker

    fmt = fmt or (lambda v: f"{v:g}")
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt(v)))
    target.set_minor_formatter(mticker.NullFormatter())
