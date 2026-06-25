"""
crate_merge.py — Merge per-TRE RO-Crates into flwrCrate's run-crate.

flwrCrate emits a run-crate describing the federated computation (the
CreateAction, strategy, frameworks, model, metrics). Separately, each TRE
contributes its own RO-Crate describing its institute and location. This
module folds the TRE provenance INTO the run-crate so there's a single,
self-contained provenance record.

Design (per project decisions):
  • Each TRE's Organization is attached to the run's CreateAction as a
    contributor/agent.
  • The full TRE entity set (Organization + GeoCoordinates + nested
    PostalAddress) is embedded, so the merged crate stands alone.

Collision handling:
  • TRE entities with ABSOLUTE-URI @ids (ROR org, GeoNames geo) are embedded
    as-is. If two TREs share an institution, the identical @id naturally
    dedupes.
  • The TRE crate's own local packaging entities ("./" root Dataset and
    "ro-crate-metadata.json" descriptor) are NOT carried over — they describe
    the TRE's standalone crate, not the run, and would collide with the
    run-crate's equivalents.

Safety:
  • The function never raises on bad input; it logs and returns a status dict.
  • If the run-crate can't be found/parsed, the TRE data is left untouched on
    disk so nothing is lost.
  • A backup of the original run-crate is written before modification.
"""

import json
import os
import shutil
from typing import Optional


# Entities whose @id is one of these are TRE-crate-local packaging artefacts
# and must not be merged into the run-crate.
_SKIP_LOCAL_IDS = {"./", "ro-crate-metadata.json"}

# Candidate @ids for flwrCrate's run action. The README documents "#fl-run";
# we try a few defensively in case the casing/prefix differs in practice.
_RUN_ACTION_ID_CANDIDATES = ["#fl-run", "#run", "fl-run"]


def _is_absolute_uri(entity_id: str) -> bool:
    return entity_id.startswith("http://") or entity_id.startswith("https://")


def _find_run_action(graph: list) -> Optional[dict]:
    """Locate flwrCrate's CreateAction entity in the run-crate graph."""
    # First, try the documented/likely @ids.
    by_id = {e.get("@id"): e for e in graph}
    for cand in _RUN_ACTION_ID_CANDIDATES:
        if cand in by_id:
            return by_id[cand]
    # Fall back: the first CreateAction in the graph.
    for e in graph:
        types = e.get("@type")
        types = types if isinstance(types, list) else [types]
        if "CreateAction" in types:
            return e
    return None


def _extract_tre_entities(crate: dict) -> list:
    """
    Pull the embeddable entities (absolute-URI ones) out of a TRE crate's
    @graph, skipping its local packaging entities.
    """
    out = []
    for entity in crate.get("@graph", []):
        eid = entity.get("@id", "")
        if eid in _SKIP_LOCAL_IDS:
            continue
        if not _is_absolute_uri(eid):
            # Anything else local/relative (unexpected for these crates) is
            # skipped to avoid id collisions; absolute URIs are safe.
            continue
        out.append(entity)
    return out


def _find_org_id(tre_entities: list) -> Optional[str]:
    """Return the @id of the Organization entity among the extracted set."""
    for e in tre_entities:
        types = e.get("@type")
        types = types if isinstance(types, list) else [types]
        if "Organization" in types:
            return e.get("@id")
    return None


def _append_ref(action: dict, prop: str, ref_id: str) -> None:
    """
    Append {"@id": ref_id} to action[prop], normalising to a list and
    deduplicating. RO-Crate allows a property to be a single object or a list.
    """
    existing = action.get(prop)
    if existing is None:
        action[prop] = {"@id": ref_id}
        return
    # Normalise to a list of {"@id": ...}
    items = existing if isinstance(existing, list) else [existing]
    if any(isinstance(i, dict) and i.get("@id") == ref_id for i in items):
        return  # already present
    items.append({"@id": ref_id})
    action[prop] = items


def merge_tre_crates_into_run_crate(
    run_crate_path: str,
    tre_provenance: dict,
    agent_property: str = "contributor",
) -> dict:
    """
    Merge each TRE's crate entities into the flwrCrate run-crate in place.

    Parameters
    ----------
    run_crate_path : str
        Path to flwrCrate's ro-crate-metadata.json (inside <output>/ro-crate/).
    tre_provenance : dict
        {tre_num: {"cohort": str, "crate_json": str, "present": bool}} —
        i.e. strategy.provenance.
    agent_property : str
        Which CreateAction property to attach the TRE orgs to. "contributor"
        keeps them distinct from flwrCrate's own "agent" (the executor),
        while still marking them as participating organisations.

    Returns
    -------
    dict : status report (merged count, skipped, warnings, whether written).
    """
    status = {
        "written"   : False,
        "merged_tres": [],
        "skipped_tres": [],
        "warnings"  : [],
        "run_crate_path": run_crate_path,
    }

    # 1. Load the run-crate.
    if not os.path.isfile(run_crate_path):
        status["warnings"].append(
            f"run-crate not found at {run_crate_path}; nothing merged.")
        return status
    try:
        with open(run_crate_path, "r", encoding="utf-8") as f:
            run_crate = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        status["warnings"].append(
            f"could not read run-crate ({e.__class__.__name__}); nothing merged.")
        return status

    graph = run_crate.get("@graph")
    if not isinstance(graph, list):
        status["warnings"].append("run-crate has no @graph list; nothing merged.")
        return status

    # 2. Find the run action to attach organisations to.
    action = _find_run_action(graph)
    if action is None:
        status["warnings"].append(
            "no CreateAction found in run-crate; embedding TRE entities but "
            "not linking them to a run action.")

    existing_ids = {e.get("@id") for e in graph}

    # 3. Process each TRE.
    for tre_num in sorted(tre_provenance.keys()):
        info       = tre_provenance[tre_num]
        cohort     = info.get("cohort", f"tre{tre_num}")
        present    = info.get("present", False)
        crate_json = info.get("crate_json", "")

        if not present or not crate_json:
            status["skipped_tres"].append({"tre_num": tre_num, "cohort": cohort,
                                            "reason": "no crate"})
            continue
        try:
            tre_crate = json.loads(crate_json)
        except json.JSONDecodeError:
            status["skipped_tres"].append({"tre_num": tre_num, "cohort": cohort,
                                           "reason": "malformed crate JSON"})
            continue

        tre_entities = _extract_tre_entities(tre_crate)
        if not tre_entities:
            status["skipped_tres"].append({"tre_num": tre_num, "cohort": cohort,
                                           "reason": "no embeddable entities"})
            continue

        # Embed entities, deduping by @id (shared institutions collapse).
        for entity in tre_entities:
            eid = entity.get("@id")
            if eid in existing_ids:
                continue
            graph.append(entity)
            existing_ids.add(eid)

        # Link the TRE's Organization to the run action.
        org_id = _find_org_id(tre_entities)
        if org_id and action is not None:
            _append_ref(action, agent_property, org_id)

        status["merged_tres"].append({"tre_num": tre_num, "cohort": cohort,
                                      "org_id": org_id})

    # 4. Back up the original, then write the merged crate.
    try:
        shutil.copy2(run_crate_path, run_crate_path + ".pre-merge.bak")
    except OSError as e:
        status["warnings"].append(
            f"could not write backup ({e.__class__.__name__}); proceeding.")

    try:
        with open(run_crate_path, "w", encoding="utf-8") as f:
            json.dump(run_crate, f, indent=2)
        status["written"] = True
    except OSError as e:
        status["warnings"].append(
            f"could not write merged crate ({e.__class__.__name__}).")

    return status