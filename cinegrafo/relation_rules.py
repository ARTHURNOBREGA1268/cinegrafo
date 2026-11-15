"""Regras conservadoras de complemento Text-to-Graph.

As regras só emitem arestas quando ambas as entidades já existem no trecho,
ambas aparecem literalmente na mesma frase e um padrão de relação explícito
é encontrado. Não inferem fatos por proximidade semântica.
"""
from __future__ import annotations

import re

WORK_TYPES = {"Movie", "Series"}


def _name_pattern(name: str) -> str:
    # Os nomes são escapados; aspas tipográficas opcionais são aceitas porque
    # algumas fontes formatam títulos e personagens com aspas e outras não.
    return r"[“\"'‘]?" + re.escape(name) + r"[”\"'’]?"


def _find_sentence_evidence(sentence: str, left: str, cue: str, right: str) -> str | None:
    pattern = (
        r"(?P<evidence>" + _name_pattern(left) + r"\s+" + cue + r"\s*"
        + _name_pattern(right) + r")"
    )
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    return match.group("evidence").strip() if match else None


def infer_explicit_relations(text: str, entities: list[dict]) -> list[dict]:
    """Return only relation triples supported by exact, contiguous text spans."""
    people = [e for e in entities if e.get("type") == "Person" and e.get("name")]
    works = [e for e in entities if e.get("type") in WORK_TYPES and e.get("name")]
    characters = [e for e in entities if e.get("type") == "Character" and e.get("name")]
    genres = [e for e in entities if e.get("type") == "Genre" and e.get("name")]
    franchises = [e for e in entities if e.get("type") == "Franchise" and e.get("name")]
    platforms = [e for e in entities if e.get("type") == "Platform" and e.get("name")]

    rules: list[tuple[str, str, str, str]] = []
    for person in people:
        for work in works:
            rules.extend([
                (person["name"], r"criou\s+(?:a\s+)?(?:s[ée]rie|filme|longa(?:-metragem)?)", work["name"], "CREATED"),
                (person["name"], r"dirigiu\s+(?:o\s+)?(?:filme|longa(?:-metragem)?|s[ée]rie)", work["name"], "DIRECTED"),
                (person["name"], r"atuou\s+(?:n[oa]\s+)?(?:filme|longa(?:-metragem)?|s[ée]rie)?\s*(?:“|\"|'|‘)?", work["name"], "ACTED_IN"),
            ])
        for character in characters:
            # Padrão admite texto intermediário do crédito, mas exige a
            # formulação literal "atuou ... como ... personagem ...".
            rules.append((person["name"], r"atuou\b.*?\bcomo\s+(?:a\s+)?personagem(?:\s+fict[ií]cia)?\s+", character["name"], "PORTRAYS"))

    for work in works:
        for genre in genres:
            rules.append((work["name"], r".*?\btem\s+(?:o\s+)?g[êe]nero\s+", genre["name"], "HAS_GENRE"))
        for franchise in franchises:
            rules.append((work["name"], r".*?\bintegra\s+(?:a\s+)?franquia\s+(?:fict[ií]cia\s+)?", franchise["name"], "PART_OF"))
        for platform in platforms:
            rules.append((work["name"], r".*?\best[aá]\s+dispon[ií]vel\s+(?:na|em)\s+(?:plataforma\s+)?", platform["name"], "AVAILABLE_ON"))

    output, seen = [], set()
    # Isola frases para impedir que entidades de registros vizinhos se
    # conectem por estarem no mesmo chunk.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
    for sentence in sentences:
        for left, cue, right, relation_type in rules:
            if not left or not right:
                continue
            evidence = _find_sentence_evidence(sentence, left, cue, right)
            if not evidence:
                continue
            key = (left.casefold(), right.casefold(), relation_type)
            if key in seen:
                continue
            seen.add(key)
            output.append({"source": left, "target": right, "type": relation_type, "evidence": evidence})
    return output
