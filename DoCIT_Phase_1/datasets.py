"""Load the benchmark artifacts and check their published reference matrices."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


@dataclass
class Project:
    name: str
    requirements: dict[str, str]
    code: dict[str, str]
    references: dict[str, set[str]]
    groups: dict[str, str]
    stop_words: str | None


HOLDOUT = {"eANCI": ("eANCI", "eanci_solution_links.txt", (139, 55, 567), None)}

DEVELOPMENT = {
    "iTrust": ("iTrust", "itrust_solution_links.txt", (131, 226, 286), "english"),
    "eTour": ("eTour", "etour_solution_links_english.txt", (58, 116, 308), "english"),
    "SMOS": ("smos", "smos_solution_links_italian.txt", (67, 100, 1044), None),
}


def decode(data):
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252")


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assemble(name, requirements, code, pairs, groups, expected, stop_words):
    references = {}
    duplicates = 0
    for requirement, target in pairs:
        if requirement not in requirements or target not in code:
            raise ValueError(f"{name}: unresolved reference {requirement!r} -> {target!r}")
        existing = references.setdefault(requirement, set())
        duplicates += target in existing
        existing.add(target)
    if duplicates:
        raise ValueError(f"{name}: {duplicates} duplicate reference pairs")
    counts = len(requirements), len(code), sum(map(len, references.values()))
    if counts != expected:
        raise ValueError(f"{name}: observed {counts}, expected {expected}")
    return Project(name, requirements, code, references, groups, stop_words)


def benchmark_projects(archive, *, audit_holdout_only=False):
    with ZipFile(archive) as zf:
        files = [name for name in zf.namelist() if not name.endswith("/")]
        specs = HOLDOUT if audit_holdout_only else DEVELOPMENT
        for project, (folder, matrix, expected, stop_words) in specs.items():
            marker = f"/datasets/{folder}/"
            roots = {name.split(marker, 1)[0] + marker for name in files if marker in name}
            if len(roots) != 1:
                raise ValueError(f"{project}: expected one dataset directory in the ZIP")
            root = roots.pop()
            requirements = read_members(zf, files, root + "req/", ".txt", read_text=not audit_holdout_only)
            suffixes = (".java", ".css", ".html") if project == "SMOS" else ".java"
            code = read_members(zf, files, root + "code/", suffixes, read_text=not audit_holdout_only)
            pairs = [
                tuple(piece.strip() for piece in line.split(":"))
                for line in decode(zf.read(root + matrix)).splitlines()
                if line.strip()
            ]
            if project == "SMOS":
                # LiSSA uses the extensionless ID managerUser for this HTML/JSP artifact.
                pairs = [(r, "managerUser.html" if c == "managerUser.java" else c) for r, c in pairs]
            groups = {}
            for requirement in requirements:
                match = re.match(r"^(UC\d+)", requirement) if project == "iTrust" else None
                groups[requirement] = match[1] if match else Path(requirement).stem
            yield assemble(project, requirements, code, pairs, groups, expected, stop_words)


def read_members(zf, files, prefix, suffix, *, read_text):
    artifacts = {}
    for name in sorted(files):
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        ident = PurePosixPath(name).name
        if ident in artifacts:
            raise ValueError(f"Ambiguous artifact filename {ident!r} under {prefix}")
        artifacts[ident] = decode(zf.read(name)) if read_text else ""
    return artifacts


def audit_row(project, role):
    return {
        "project": project.name,
        "role": role,
        "requirements": len(project.requirements),
        "files": len(project.code),
        "links": sum(map(len, project.references.values())),
        "linked_requirements": len(project.references),
        "no_recorded_positive": len(project.requirements) - len(project.references),
        "unresolved_references": 0,
        "duplicate_pairs": 0,
        "reference_id_corrections": sum("managerUser.html" in row for row in project.references.values())
        if project.name == "SMOS" else 0,
    }
