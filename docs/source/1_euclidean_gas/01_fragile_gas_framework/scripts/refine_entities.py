#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple


CATEGORY_MAP = {
    "axioms": "axiom",
    "theorems": "theorem",
    "lemmas": "lemma",
    "propositions": "proposition",
    "corollaries": "corollary",
    "definitions": "definition",
    "objects": "object",
    "parameters": "parameter",
    "proofs": "proof",
}


STATEMENT_TYPE_MAP = {
    "axiom": "AxiomBox",
    "theorem": "TheoremBox",
    "lemma": "LemmaBox",
    "proposition": "PropositionBox",
    "corollary": "CorollaryBox",
    "definition": "DefinitionBox",
    "object": "ObjectBox",
    "parameter": "ParameterBox",
    "proof": "ProofBox",
}


def parse_lines_range(value: str) -> tuple[int, int]:
    try:
        a, b = value.split("-")
        return int(a), int(b)
    except Exception:
        return (None, None)  # type: ignore


def normalize_document_id(source_name: str) -> str:
    # Remove extension and any directory components
    base = os.path.basename(source_name)
    if base.lower().endswith(".md"):
        base = base[:-3]
    return base


def derive_output_dir(raw_path: str) -> tuple[str, str]:
    # raw_path like raw_data/definitions/foo.json -> refined_data/definitions/
    parts = raw_path.strip().split(os.sep)
    try:
        category = parts[1]
    except Exception:
        category = "definitions"
    entity_type = CATEGORY_MAP.get(category, "definition")
    out_dir = os.path.join("refined_data", category)
    return out_dir, entity_type


def ensure_label(raw: dict[str, Any], fallback_filename: str) -> str:
    label = raw.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    # filename without .json
    return os.path.splitext(os.path.basename(fallback_filename))[0]


def pick_statement(raw: dict[str, Any]) -> str:
    for key in ("statement", "content", "full_text", "natural_language_statement", "text"):
        v = raw.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None  # type: ignore


def pick_name(raw: dict[str, Any], label: str) -> str:
    for key in ("name", "title", "term", "term_being_defined"):
        v = raw.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # fallback: transform label
    return re.sub(r"[-_]+", " ", label).strip().title()


def collect_dependencies(
    raw: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[dict[str, str]]]:
    input_objects: list[str] = []
    input_axioms: list[str] = []
    input_parameters: list[str] = []
    lemma_dag_edges: list[dict[str, str]] = []

    def add_dep(x: str):
        if not isinstance(x, str):
            return
        if x.startswith(("axiom-", "def-axiom-")):
            input_axioms.append(x)
        elif x.startswith(("param-", "parameter-")):
            input_parameters.append(x)
        else:
            input_objects.append(x)

    # top-level dependencies
    deps = raw.get("dependencies")
    if isinstance(deps, list):
        for x in deps:
            add_dep(x)

    # parameters
    params = raw.get("parameters")
    if isinstance(params, list):
        for p in params:
            if isinstance(p, str):
                input_parameters.append(p)

    # relations field may encode graph-style dependencies
    rels = raw.get("relations")
    if isinstance(rels, dict):
        uses = rels.get("uses")
        if isinstance(uses, list):
            for x in uses:
                add_dep(x)
        required_by = rels.get("required_by")
        if isinstance(required_by, list):
            for tgt in required_by:
                if isinstance(tgt, str):
                    lemma_dag_edges.append({
                        "source": None,
                        "target": tgt,
                    })  # source to be filled later
        generalizes = rels.get("generalizes")
        if isinstance(generalizes, list):
            for tgt in generalizes:
                if isinstance(tgt, str):
                    lemma_dag_edges.append({"source": None, "target": tgt})

    return input_objects, input_axioms, input_parameters, lemma_dag_edges


def build_source_location(raw: dict[str, Any], label: str) -> dict[str, Any]:
    src: dict[str, Any] = {}
    # Various raw keys
    context = raw.get("context", {}) if isinstance(raw.get("context"), dict) else {}
    section = raw.get("section") or raw.get("source_section") or context.get("section")
    subsection = raw.get("subsection") or context.get("subsection")
    # file path variants
    fp = raw.get("source_file") or raw.get("source_document") or context.get("source_file")
    # Normalize to full path if only a doc id is provided
    file_path = None
    document_id = None
    if isinstance(fp, str) and fp:
        document_id = normalize_document_id(fp)
        # If path seems short, place into docs/source default path
        if fp.endswith(".md") and ("/" not in fp):
            file_path = f"docs/source/1_euclidean_gas/{fp}"
        else:
            file_path = fp
    else:
        # try chapter id
        chap = context.get("chapter")
        if isinstance(chap, str) and chap:
            document_id = normalize_document_id(chap.split("/")[-1])
            file_path = (
                f"docs/source/{chap}.md" if not chap.endswith(".md") else f"docs/source/{chap}"
            )

    # line info
    line_start = None
    line_end = None
    source_lines = raw.get("source_lines") or context.get("lines")
    if isinstance(source_lines, str):
        ls, le = parse_lines_range(source_lines)
        line_start, line_end = ls, le
    elif isinstance(source_lines, list) and len(source_lines) == 2:
        try:
            line_start = int(source_lines[0])
            line_end = int(source_lines[1])
        except Exception:
            pass
    # Single line number variant
    ln = raw.get("line_number") or context.get("line_start")
    if line_start is None and isinstance(ln, int):
        line_start = ln

    src["document_id"] = document_id
    src["file_path"] = file_path
    src["section"] = section
    src["subsection"] = subsection
    if line_start is not None:
        src["line_start"] = line_start
    if line_end is not None:
        src["line_end"] = line_end
    src["directive_label"] = label
    return src


def refine_one(raw_path: str) -> tuple[str, dict[str, Any]]:
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)

    out_dir, expected_entity_type = derive_output_dir(raw_path)
    label = ensure_label(raw, raw_path)
    name = pick_name(raw, label)
    # Determine actual entity type from raw if provided, else directory
    raw_type = raw.get("type") if isinstance(raw.get("type"), str) else None
    entity_type = expected_entity_type
    if isinstance(raw_type, str):
        raw_type_norm = raw_type.strip().lower()
        # Trust directory if mismatch; carry raw_type for traceability
        if raw_type_norm and raw_type_norm != expected_entity_type:
            entity_type = expected_entity_type
        else:
            entity_type = raw_type_norm

    statement_type = STATEMENT_TYPE_MAP.get(entity_type, "TextBox")
    nls = pick_statement(raw)
    description = raw.get("description") or raw.get("notes")

    input_objects, input_axioms, input_parameters, lemma_dag_edges = collect_dependencies(raw)

    # fill source
    source_location = build_source_location(raw, label)

    # convert lemma_dag_edges to include source label
    finalized_edges: list[dict[str, str]] = []
    for e in lemma_dag_edges:
        tgt = e.get("target")
        if isinstance(tgt, str):
            finalized_edges.append({"source": label, "target": tgt})

    # relations: keep only formula-like entries if present
    relations: list[dict[str, Any]] = []
    # Attempt to extract simple formulas from known fields
    if isinstance(raw.get("components"), list):
        for comp in raw["components"]:
            if isinstance(comp, dict):
                formula = comp.get("formula")
                if isinstance(formula, str):
                    relations.append({"formula": formula, "description": comp.get("description")})
    if isinstance(raw.get("equations"), list):
        for eq in raw["equations"]:
            if isinstance(eq, dict):
                formula = eq.get("formula") or eq.get("eq")
                if isinstance(formula, str):
                    relations.append({"formula": formula, "description": eq.get("description")})

    refined: dict[str, Any] = {
        "label": label,
        "name": name,
        "entity_type": entity_type,
        "statement_type": statement_type,
        "natural_language_statement": nls,
        "description": description,
        "input_objects": input_objects,
        "input_axioms": input_axioms,
        "input_parameters": input_parameters,
        "lemma_dag_edges": finalized_edges,
        "internal_lemmas": [],
        "source_location": source_location,
        "relations": relations,
        "proof_status": "n/a"
        if entity_type in {"axiom", "definition", "object", "parameter"}
        else None,
        "raw_fallback": raw,
    }

    # Remove None fields for cleanliness
    def prune(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if v is not None}

    refined = prune(refined)

    # Output path
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{label}.json")
    return out_file, refined


def main():
    ap = argparse.ArgumentParser(description="Refine raw mathematical entity JSONs")
    ap.add_argument(
        "--input-list", required=True, help="Path to file with list of raw_data/* JSONs"
    )
    ap.add_argument(
        "--limit", type=int, default=10, help="Number of files to process from the list"
    )
    args = ap.parse_args()

    with open(args.input_list, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    count = 0
    for raw_path in lines:
        if count >= args.limit:
            break
        if not os.path.exists(raw_path):
            print(f"SKIP: {raw_path} (not found)")
            count += 1
            continue
        try:
            out_file, refined = refine_one(raw_path)
            with open(out_file, "w", encoding="utf-8") as wf:
                json.dump(refined, wf, ensure_ascii=False, indent=2)
            print(f"OK: {raw_path} -> {out_file}")
        except Exception as e:
            print(f"ERROR: {raw_path}: {e}")
        count += 1


if __name__ == "__main__":
    main()
