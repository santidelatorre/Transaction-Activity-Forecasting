"""Check an update to the sole allowed source branch without merging its code."""

import ast
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "e4aa4c58175242e198cefd32d6ac4558145523af"
UPDATE = "d5dddfd8d93df61d3a6421f860428bf491c995f7"


class WithoutImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def load(ref, name):
    source = subprocess.check_output(["git", "show", f"{ref}:{name}"], cwd=ROOT).decode(
        "utf-8-sig"
    )
    tree = WithoutImports().visit(ast.parse(source))
    definitions = {
        n.name: ast.dump(n, include_attributes=False)
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    return tree, definitions


def main():
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE, "src/ubs_recurrence"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    result = []
    for path in paths:
        if not path.endswith(".py"):
            continue
        before, a = load(BASE, path)
        after, b = load(UPDATE, path)
        result.append(
            {
                "path": path,
                "ast_equal_excluding_imports": ast.dump(
                    before, include_attributes=False
                )
                == ast.dump(after, include_attributes=False),
                "changed_definitions": sorted(
                    k for k in a.keys() | b.keys() if a.get(k) != b.get(k)
                ),
            }
        )
    receipt = {
        "allowed_source_branch": "research/import-v2-stream-identity",
        "experiment_base": BASE,
        "same_branch_update_during_investigation": UPDATE,
        "comparison": "AST comparison ignores formatting and import reordering/removal; not a proof of arbitrary semantic equivalence",
        "modules": result,
        "action": "Keep experiment code pinned; document source update without merging or reading another branch.",
    }
    (ROOT / "reports/stream_time_audit/upstream_update.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
