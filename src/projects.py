"""
Project definitions loaded from projects.json.

Used for: resolving project in ingestion, filtering queries by project,
and enriching LLM prompt. Not stored in Chroma.
"""

import json
from pathlib import Path
from typing import Any

_PROJECTS_PATH = Path(__file__).resolve().parent / "projects.json"

# Load once at import
with open(_PROJECTS_PATH, encoding="utf-8") as f:
    _raw: dict[str, Any] = json.load(f)

PROJECTS: dict[str, dict[str, Any]] = _raw


def resolve_project(project_key: str) -> dict[str, Any]:
    """
    Resolve project by folder key (first segment of document path).

    Args:
        project_key: Key from path (e.g. master_detox from documents/master_detox/file.pdf)

    Returns:
        Project dict with project_id, name, description, tags, domain, language

    Raises:
        RuntimeError: If project_key is not defined in projects.json
    """
    if project_key not in PROJECTS:
        raise RuntimeError(f"Projeto '{project_key}' não definido em projects.json")
    return PROJECTS[project_key]


def get_projects_by_tag(tag: str) -> list[dict[str, Any]]:
    """
    Return projects that have the given tag.

    Args:
        tag: Tag to filter by (e.g. "flutter")

    Returns:
        List of project dicts
    """
    return [p for p in PROJECTS.values() if tag in p.get("tags", [])]
