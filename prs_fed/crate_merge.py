"""
Merge per-TRE RO-Crates into the federated run's RO-Crate.

The provenance-capture layer emits a run-crate describing the federated
computation (the run action, strategy, frameworks, model, and metrics).
Separately, each TRE contributes its own RO-Crate describing its institute and
location. This module folds the TRE provenance into the run-crate to produce a
single, self-contained provenance record.

Behaviour:

  * Each TRE's ``Organization`` is attached to the run's ``CreateAction`` as a
    contributor. The full TRE entity set (Organization, GeoCoordinates, and the
    nested PostalAddress) is embedded so the merged crate stands alone.

Collision handling:

  * TRE entities with absolute-URI ``@id`` values (e.g. a ROR organisation or a
    GeoNames location) are embedded as-is. When two TREs share an institution,
    the identical ``@id`` causes them to be de-duplicated automatically.
  * The TRE crate's own packaging entities (the ``./`` root Dataset and the
    ``ro-crate-metadata.json`` descriptor) are not carried over: they describe
    the TRE's standalone crate, not the run, and would collide with the
    run-crate's own equivalents.

Safety:

  * The merge never raises on bad input; it logs through a returned status
    dictionary instead.
  * If the run-crate cannot be found or parsed, the TRE data is left untouched.
  * The original run-crate is backed up before modification.
"""

import json
import os
import shutil
from typing import Optional


# Entities with these @id values are TRE-crate-local packaging artefacts and
# must not be merged into the run-crate.
_SKIP_LOCAL_IDS = {"./", "ro-crate-metadata.json"}

# Candidate @id values for the run's CreateAction. The primary value is tried
# first; the others provide resilience to differences in crate generation.
_RUN_ACTION_ID_CANDIDATES = ["#fl-run", "#run", "fl-run"]


def _is_absolute_uri(entity_id: str) -> bool:
    return entity_id.startswith("http://") or entity_id.startswith("https://")


def _find_run_action(graph: list) -> Optional[dict]:
    """Locate the run's ``CreateAction`` entity in the run-crate graph."""
    # Try the known @id values first.
    by_id = {e.get("@id"): e for e in graph}
    for cand in _RUN_ACTION_ID_CANDIDATES:
        if cand in by_id:
            return by_id[cand]
    # Otherwise, fall back to the first CreateAction in the graph.
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
    Merge each TRE's crate entities into the run-crate, in place.

    Parameters
    ----------
    run_crate_path : str
        Path to the run-crate's ``ro-crate-metadata.json`` (inside
        ``<output>/ro-crate/``).
    tre_provenance : dict
        Maps ``tre_num`` to ``{"cohort", "crate_json", "present"}``, as
        collected by the strategy during the run.
    agent_property : str
        The ``CreateAction`` property to which the TRE organisations are
        attached. The default, ``"contributor"``, marks them as participating
        organisations while keeping them distinct from the run's executor
        (recorded separately as the action's agent).

    Returns
    -------
    dict
        A status report containing the merged and skipped TREs, any warnings,
        and whether the merged crate was written.
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

        # Embed entities, de-duplicating by @id so shared institutions appear once.
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