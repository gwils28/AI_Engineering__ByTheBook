"""Tests rapides (sans GPU) du module du chapitre 6 : RAG et agents."""
import json
import math
import sqlite3

import numpy as np
import pytest

from aie import ch06_rag_agents as c6


# ----------------------------------------------------------------------------- corpus
def test_corpus_et_annotations_coherents():
    P = c6.passages()
    assert 40 <= len(P) <= 80
    pids = {p.pid: p for p in P}
    assert len(pids) == len(P)
    for q in c6.QUESTIONS:
        assert q["relevant"] and all(r in pids for r in q["relevant"])
        best = max(q["relevant"], key=q["relevant"].get)
        assert q["key"] in pids[best].text, q["qid"]
        assert q["type"] in {"lexical", "semantique", "ambigu"}
    for conv in c6.CONVERSATIONS:
        assert all(r in pids for r in conv["relevant"])


def test_tokenisation():
    toks = c6.tokenize("Le code E-42 du Thermo-X200 et les tomates")
    assert "e-42" in toks and "thermo-x200" in toks
    assert "le" not in toks and "tomate" in toks
    assert c6.tokenize("Télétravail", stem=False) == ["teletravail"]


# ------------------------------------------------------------------- recherche par termes
def test_index_inverse():
    idx = c6.build_inverted_index([["a", "b", "a"], ["b", "c"]])
    assert idx["a"] == [(0, 2)]
    assert sorted(idx["b"]) == [(0, 1), (1, 1)]


def test_tfidf_formule_du_livre():
    docs = [["chat", "chat", "noir"], ["chien", "noir"], ["chat"]]
    m = c6.TfIdf(docs)
    s = m.scores(["chat"])
    assert math.isclose(s[0], 2 * math.log(3 / 2))
    assert s[1] == 0


def test_bm25_identique_a_rank_bm25():
    from rank_bm25 import BM25Okapi

    toks = [c6.tokenize(p.text) for p in c6.passages()]
    q = c6.tokenize(c6.QUESTIONS[0]["question"] + " " + c6.QUESTIONS[30]["question"])
    for k1, b in [(1.5, 0.75), (1.2, 0.3), (2.0, 1.0)]:
        np.testing.assert_allclose(c6.BM25(toks, k1=k1, b=b).scores(q), BM25Okapi(toks, k1=k1, b=b).get_scores(q), atol=1e-10)


def test_bm25_saturation_et_longueur():
    # même tf, document plus long -> score plus faible ; tf croissant -> score borné
    docs = [["x"] + ["y"] * 2, ["x"] + ["y"] * 20] + [["z"]] * 5
    s = c6.BM25(docs, variant="lucene").scores(["x"])
    assert s[0] > s[1] > 0
    docs = [["x"] * n + ["w"] * (50 - n) for n in (1, 5, 50)] + [["z"] * 50] * 5
    s = c6.BM25(docs, b=0.0, variant="lucene").scores(["x"])
    idf = math.log(1 + (8 - 3 + 0.5) / (3 + 0.5))
    assert s[0] < s[1] < s[2] < idf * (1.5 + 1)


def test_bm25_trouve_les_questions_lexicales():
    P = c6.passages()
    r = c6.TermRetriever([p.text for p in P], [p.pid for p in P])
    assert r.search("Que signifie l'erreur EADDRNOTAVAIL ?", 1) == ["reseau-2"]
    assert r.search("code E-42", 1) == ["thermo-4"]


# ------------------------------------------------------------------------------- fusion
def test_rrf():
    fused = c6.rrf([["a", "b", "c"], ["b", "a", "d"]], k=0)
    scores = dict(fused)
    assert math.isclose(scores["a"], 1 + 0.5) and math.isclose(scores["b"], 1.5)
    assert fused[-1][0] in {"c", "d"}
    fused = c6.rrf([["a", "b"], ["b", "c"]], k=60)
    assert fused[0][0] == "b"


# ----------------------------------------------------------------------------- métriques
def test_metriques():
    ret = ["x", "a", "y", "b", "z"]
    rel = {"a": 1, "b": 1, "c": 1}
    assert c6.precision_at_k(ret, rel, 5) == pytest.approx(2 / 5)
    assert c6.recall_at_k(ret, rel, 5) == pytest.approx(2 / 3)
    assert c6.reciprocal_rank(ret, rel) == pytest.approx(1 / 2)
    assert c6.average_precision(ret, rel) == pytest.approx((1 / 2 + 2 / 4) / 3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert c6.ndcg_at_k(ret, rel, 5) == pytest.approx(dcg / idcg)
    assert c6.ndcg_at_k(["a", "b", "c"], rel, 3) == pytest.approx(1.0)
    # notes graduées : mettre le document noté 2 en tête est meilleur
    g = {"a": 2, "b": 1}
    assert c6.ndcg_at_k(["a", "b"], g, 2) == pytest.approx(1.0)
    assert c6.ndcg_at_k(["b", "a"], g, 2) < 1.0


def test_evaluate_run_parfait():
    run = {q["qid"]: sorted(q["relevant"], key=lambda d: -q["relevant"][d]) + ["zzz"] * 5 for q in c6.QUESTIONS}
    m = c6.evaluate(run, k=5)
    assert m["MRR"] == 1.0 and m["R@5"] == 1.0 and m["NDCG@5"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- découpage
def test_decoupages():
    t = "abcdefghij"
    assert c6.chunk_chars(t, 4, 0) == ["abcd", "efgh", "ij"]
    assert c6.chunk_chars(t, 4, 2) == ["abcd", "cdef", "efgh", "ghij"]
    assert c6.chunk_words("a b c d e", 2, 1) == ["a b", "b c", "c d", "d e"]
    s = c6.split_sentences("Une phrase. Une autre ! Et 3 encore ?\n\nNouveau paragraphe.")
    assert len(s) == 4
    doc = c6.document_text(c6.DOCUMENTS[0])
    rec = c6.chunk_recursive(doc, 300)
    assert all(len(c) <= 300 for c in rec)
    assert c6._norm_ws(" ".join(rec)).replace(" ", "") == c6._norm_ws(doc).replace(" ", "")
    with pytest.raises(ValueError):
        c6.chunk_chars(t, 3, 3)


def test_pertinence_des_morceaux():
    texts, ids, _ = c6.make_chunks(c6.chunk_paragraphs)
    assert len(texts) == len(c6.passages())
    rel = c6.chunk_relevance(texts, ids, c6.QUESTIONS[0]["key"])
    assert rel == {"thermo#3"}


# ---------------------------------------------------------------------------- génération
def test_answer_ok_et_prompt():
    assert c6.answer_ok("Il coûte 18 €.", ["18"])
    assert not c6.answer_ok("Il coûte 180 €.", ["18"])
    assert c6.answer_ok("C'est Vénus", ["Venus"])
    assert c6.answer_ok("Prenez la taille M.", ["M"]) and not c6.answer_ok("Mesurez", ["M"])
    p = c6.build_rag_prompt("Q ?", ["doc a", "doc b"])
    assert "[Document 2]" in p and p.endswith("Réponse :")
    assert c6.abstains("Je ne sais pas.")


# -------------------------------------------------------------------------------- agents
def test_calculatrice_sure():
    assert c6.safe_eval("3 * 18") == 54
    assert c6.safe_eval("round(199999 / 292, 2)") == 684.93
    assert c6.safe_eval("sqrt(1764)") == 42
    assert c6.safe_eval("17,5 / 100 * 2 480") == pytest.approx(434)
    assert c6.safe_eval("2^10") == 1024
    for bad in ["__import__('os')", "open('x')", "9**9**9", "a + 1", "(1).real"]:
        with pytest.raises(Exception):
            c6.safe_eval(bad)


def test_parse_et_validation_des_appels():
    tools = c6.make_tools(lambda q: ["passage"])
    txt = 'Je calcule.\n<tool_call>\n{"name": "calculatrice", "arguments": {"expression": "2+2"}}\n</tool_call>'
    thought, calls = c6.parse_tool_calls(txt)
    assert thought == "Je calcule." and calls[0]["name"] == "calculatrice"
    assert c6.validate_call(calls[0], tools) == "ok"
    assert c6.validate_call({"name": "bing_search", "arguments": {}}, tools) == "outil_invalide"
    assert c6.validate_call({"name": "calculatrice", "arguments": {"x": "1"}}, tools) == "parametres_invalides"
    assert c6.validate_call({"name": "calculatrice", "arguments": {}}, tools) == "parametres_invalides"
    _, calls = c6.parse_tool_calls("<tool_call>{pas du json}</tool_call>")
    assert c6.validate_call(calls[0], tools) == "json_invalide"
    _, calls = c6.parse_tool_calls('{"name": "date_du_jour", "arguments": "{}"}')
    assert calls[0]["arguments"] == {} and c6.validate_call(calls[0], tools) == "ok"
    assert tools["jours_entre"].fn("2026-11-25", "2026-12-31") == 36
    assert tools["date_du_jour"].fn() == "2026-11-25"


def test_boucle_agent_avec_modele_scripte():
    tools = c6.make_tools(lambda q: ["Le Fedora Fruité est vendu 18 euros."])
    script = iter([
        '<tool_call>{"name": "recherche_corpus", "arguments": {"requete": "prix Fedora"}}</tool_call>',
        '<tool_call>{"name": "calculatrice", "arguments": {"expression": "3*"}}</tool_call>',  # erreur
        '<tool_call>{"name": "calculatrice", "arguments": {"expression": "3*18"}}</tool_call>',
        "Trois Fedoras coûtent 54 euros.",
    ])
    seen = []

    def step(messages, schemas):
        seen.append(len(messages))
        assert {s["function"]["name"] for s in schemas} == set(tools)
        return next(script)

    res = c6.run_agent("Combien coûtent 3 Fedoras ?", tools, step)
    assert res.finished and "54" in res.answer
    assert res.tools_used == ["recherche_corpus", "calculatrice", "calculatrice"]
    assert [s.status for s in res.steps[:3]] == ["ok", "erreur_execution", "ok"]
    assert res.steps[2].observation == "54"
    assert seen == [1, 3, 5, 7]  # chaque observation est réinjectée dans l'historique
    assert "Observation" in res.trace()


def test_budget_d_etapes():
    tools = c6.make_tools()
    res = c6.run_agent("?", tools, lambda m, s: '<tool_call>{"name": "date_du_jour", "arguments": {}}</tool_call>', max_steps=3)
    assert not res.finished and res.answer is None and res.n_tool_calls == 3


def test_erreurs_composees_et_plans():
    assert c6.compound_success(0.95, 10) == pytest.approx(0.5987, abs=1e-4)
    assert c6.compound_success(0.95, 100) == pytest.approx(0.0059, abs=1e-4)
    tools = {"a", "b"}
    assert c6.validate_plan(["a", "b"], tools) == (True, "ok")
    assert not c6.validate_plan(["a", "google"], tools)[0]
    assert not c6.validate_plan([], tools)[0]
    assert c6.parse_plan("Plan: [fetch_info, generate_response]") == ["fetch_info", "generate_response"]
    assert c6.parse_plan("1. a()\n2. b") == ["a", "b"]


# ------------------------------------------------------------------------------- mémoire
def test_conversation_et_strategies():
    msgs = c6.make_conversation(30, seed=1)
    assert len(msgs) == 60
    assert sum(m["fact"] is not None for m in msgs) == len(c6.MEMORY_FACTS)
    assert msgs[0]["fact"] == 0
    full = c6.fact_retention(msgs)
    assert full.sum() == len(c6.MEMORY_FACTS)
    budget = 200
    fifo = c6.memory_fifo(msgs, budget)
    assert sum(c6.approx_tokens(m["content"]) for m in fifo) <= budget
    assert fifo[-1] is msgs[-1]
    ht = c6.memory_head_tail(msgs, budget)
    assert ht[0] is msgs[0] and c6.fact_retention(ht)[0] == 1
    assert c6.fact_retention(fifo)[0] == 0  # FIFO oublie le premier fait

    def fake_embed(texts):  # embedding « sac de mots » sur un petit vocabulaire
        vocab = ["chien", "pixel", "allergique", "arachides", "budget", "train", "camille", "lyon"]
        E = np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=float) + 1e-3
        return E / np.linalg.norm(E, axis=1, keepdims=True)

    ret = c6.memory_retrieval(msgs, budget, "Comment s'appelle mon chien ?", fake_embed)
    assert sum(c6.approx_tokens(m["content"]) for m in ret) <= budget
    assert c6.fact_retention(ret)[4] == 1
    summ = c6.memory_summary(msgs, budget, lambda old: "L'utilisatrice s'appelle Léa.")
    assert summ[0]["role"] == "system" and c6.fact_retention(summ)[0] == 1


# ------------------------------------------------------------------------- RAG tabulaire
def test_base_sql_et_garde_fous():
    conn = c6.make_sales_db()
    for item in c6.SQL_QUESTIONS:
        res = c6.run_sql(conn, item["gold"])
        assert not isinstance(res, str) and len(res) >= 1
    assert c6.run_sql(conn, "DROP TABLE ventes").startswith("ERREUR")
    assert c6.run_sql(conn, "SELECT 1; DELETE FROM ventes").startswith("ERREUR")
    assert c6.extract_sql("Voici :\n```sql\nSELECT COUNT(*) FROM ventes;\n```") == "SELECT COUNT(*) FROM ventes"
    assert c6.same_result([(1.0,)], [(1.0000000001,)]) and not c6.same_result([(1,)], [(2,)])
    assert isinstance(conn, sqlite3.Connection)


# ------------------------------------------------------------------------- graphe ANN
def test_recherche_gloutonne_dans_un_graphe():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    g = c6.knn_graph(X, m=6)
    assert all(len(n) >= 6 for n in g)
    q = np.array([0.3, -0.2])
    path = c6.greedy_graph_search(X, g, q, entry=0)
    d = ((X - q) ** 2).sum(1)
    assert d[path[-1]] <= d[path[0]]
    assert np.argsort(d).tolist().index(path[-1]) < 10  # proche du vrai plus proche voisin


def test_classification_des_executions():
    tools = c6.make_tools(lambda q: ["Le Fedora Fruité est vendu 18 euros."])

    def run(outputs, task):
        it = iter(outputs)
        return c6.classify_run(task, c6.run_agent(task["task"], tools, lambda m, s: next(it), max_steps=3))

    t1 = c6.AGENT_TASKS[0]
    call = lambda n, a: f'<tool_call>{{"name": "{n}", "arguments": {a}}}</tool_call>'  # noqa: E731
    ok = [call("recherche_corpus", '{"requete": "Fedora"}'), call("calculatrice", '{"expression": "3*18"}'), "Cela fait 54 euros."]
    assert run(ok, t1) == "succès"
    assert run([call("calculatrice", '{"expression": "3*35"}'), "Cela fait 105 euros."], t1) == "outil nécessaire non utilisé"
    assert run([call("bing", "{}"), "?"], t1) == "outil inventé"
    assert run([call("calculatrice", '{"x": 1}'), "?"], t1) == "paramètres invalides"
    assert run([call("date_du_jour", "{}")] * 3, t1) == "budget d'étapes épuisé"
    bad = [call("recherche_corpus", '{"requete": "Fedora"}'), call("calculatrice", '{"expression": "3*19"}'), "Cela fait 57 euros."]
    assert run(bad, t1) == "valeurs ou réponse incorrectes"
    salut, meteo = c6.AGENT_TASKS[8], c6.AGENT_TASKS[9]
    assert run(["Bonjour, je suis un assistant."], salut) == "succès"
    assert run([call("date_du_jour", "{}"), "Bonjour."], salut) == "appel inutile"
    assert run(["Il fera 20 degrés."], meteo) == "hallucination (outil manquant)"
    assert run(["Je n'ai pas accès à la météo."], meteo) == "succès"


def test_reflexion_par_critique_d_ancrage():
    tools = c6.make_tools(lambda q: ["Le Fedora Fruité est vendu 18 euros."])
    call = lambda n, a: f'<tool_call>{{"name": "{n}", "arguments": {a}}}</tool_call>'  # noqa: E731
    script = iter(["Trois Fedoras coûtent 105 euros.",  # nombre inventé -> critique
                   call("recherche_corpus", '{"requete": "prix Fedora"}'),
                   call("calculatrice", '{"expression": "3*18"}'),
                   "Trois Fedoras coûtent 54 euros."])
    res = c6.run_agent("Combien coûtent 3 Fedoras ?", tools, lambda m, s: next(script), max_steps=6,
                       critic=c6.grounding_critic)
    assert res.finished and "54" in res.answer
    assert [s.status for s in res.steps].count("réflexion") == 1
    fake = c6.AgentResult("Combien font 2 + 2 ?", "Cela fait 4.", [c6.Step("", {"name": "calculatrice"}, "ok", "4")], True)
    assert c6.grounding_critic(fake.question, fake) is None
    fake.answer = "Je vais chercher cette information."
    assert "annoncé" in c6.grounding_critic(fake.question, fake)


def test_critique_accepte_les_arrondis():
    r = c6.AgentResult("Combien font 199999 / 292 ?", "Environ 684,93.", [c6.Step("", {"name": "calculatrice"}, "ok", "684.928082")], True)
    assert c6.grounding_critic(r.question, r) is None
    r.answer = "Environ 690."
    assert c6.grounding_critic(r.question, r) is not None
