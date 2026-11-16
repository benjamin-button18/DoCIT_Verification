"""Conservative static Java edges for the three development corpora."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import javalang
from javalang import tree


@dataclass(frozen=True, order=True)
class Edge:
    source: str
    target: str
    relation: str
    line: int


def static_graph(code):
    parsed = {}
    symbols = {}
    java = {name: text for name, text in code.items() if name.endswith(".java")}
    for source, text in java.items():
        try:
            unit = javalang.parse.parse(text)
        except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError, TypeError):
            continue
        parsed[source] = unit
        package = unit.package.name if unit.package else ""
        for declaration in unit.types:
            qualified = ".".join(filter(None, (package, declaration.name)))
            symbols.setdefault(qualified, set()).add(source)

    edges = set()
    unresolved = Counter()
    for source, unit in parsed.items():
        package = unit.package.name if unit.package else ""
        imports = {
            item.path.rsplit(".", 1)[-1]: item.path
            for item in unit.imports if not item.static and not item.wildcard
        }

        def resolve(name):
            if not name:
                return None
            qualified = imports.get(name, ".".join(filter(None, (package, name))))
            matches = symbols.get(name, set()) if "." in name else symbols.get(qualified, set())
            if len(matches) == 1:
                return next(iter(matches))
            return None

        def record(target, relation, line):
            if target is None:
                unresolved[relation] += 1
            elif target != source:
                edges.add(Edge(source, target, relation, line or 0))

        for item in unit.imports:
            if not item.wildcard:
                class_name = item.path.rsplit(".", 1)[0] if item.static else item.path
                target = symbols.get(class_name, set())
                record(next(iter(target)) if len(target) == 1 else None, "import", item.position.line)
        for declaration in unit.types:
            extended = getattr(declaration, "extends", None)
            inherited = extended if isinstance(extended, list) else [extended] if extended else []
            inherited += list(getattr(declaration, "implements", None) or [])
            for parent in inherited:
                record(resolve(parent.name), "inheritance", declaration.position.line)
        for _, node in unit.filter(tree.MethodInvocation):
            if node.qualifier and node.qualifier.rsplit(".", 1)[-1][:1].isupper():
                record(resolve(node.qualifier), "call", node.position.line)
    return sorted(edges), {
        "parsed_java_files": len(parsed),
        "failed_java_files": sorted(set(java) - set(parsed)),
        "directed_evidence_records": len(edges),
        "unique_directed_pairs": len({(edge.source, edge.target) for edge in edges}),
        "unresolved_or_external": dict(unresolved),
    }
