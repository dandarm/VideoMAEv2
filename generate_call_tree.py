import ast
import os
from typing import Dict, List


TARGETS = ["specialization.py", "classification.py", "tracking.py"]
ROOT = os.path.dirname(__file__)
BUILTINS = set(dir(__builtins__))


class CallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: List[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            parts = []
            while isinstance(func, ast.Attribute):
                parts.append(func.attr)
                func = func.value
            if isinstance(func, ast.Name):
                parts.append(func.id)
            name = ".".join(reversed(parts))
        else:
            name = ast.dump(func)
        self.calls.append(name)
        self.generic_visit(node)


def build_repo_maps() -> Dict[str, ast.FunctionDef]:
    func_map: Dict[str, ast.FunctionDef] = {}
    funcs_by_name: Dict[str, ast.FunctionDef] = {}
    for root, _, files in os.walk(ROOT):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                rel = os.path.relpath(path, ROOT)
                module = rel[:-3].replace(os.sep, ".")
                with open(path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        qual = f"{module}.{node.name}"
                        func_map[qual] = node
                        funcs_by_name.setdefault(node.name, node)
    return func_map, funcs_by_name


FUNC_MAP, FUNCS_BY_NAME = build_repo_maps()


def is_repo_module(module: str) -> bool:
    path = os.path.join(ROOT, module.replace(".", os.sep))
    return os.path.isdir(path) or os.path.exists(path + ".py")


def collect_imports(tree: ast.AST) -> List[str]:
    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                root = n.name.split(".")[0]
                if is_repo_module(root):
                    imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if is_repo_module(root):
                for n in node.names:
                    mod = f"{module}.{n.name}" if module else n.name
                    imports.append(mod)
    return sorted(set(imports))


def collect_calls(func: ast.FunctionDef) -> List[str]:
    visitor = CallVisitor()
    visitor.visit(func)
    calls = []
    for c in sorted(set(visitor.calls)):
        root = c.split(".")[0]
        if root in BUILTINS:
            continue
        if "." in c:
            module = ".".join(c.split(".")[:-1])
            if not is_repo_module(module):
                continue
        elif c not in FUNCS_BY_NAME:
            continue
        calls.append(c)
    return calls


def get_function(name: str) -> ast.FunctionDef | None:
    if "." in name:
        module, func = name.rsplit(".", 1)
        return FUNC_MAP.get(f"{module}.{func}")
    return FUNCS_BY_NAME.get(name)


def get_desc(name: str) -> str:
    node = get_function(name)
    if not node:
        return ""
    doc = ast.get_docstring(node) or ""
    return doc.strip().splitlines()[0] if doc else ""


def build_tree(func_name: str, func_node: ast.FunctionDef, depth: int = 0, max_depth: int = 2) -> Dict[str, Dict]:
    tree: Dict[str, Dict] = {}
    if depth >= max_depth:
        return tree
    for call in collect_calls(func_node):
        sub_node = get_function(call)
        tree[call] = build_tree(call, sub_node, depth + 1, max_depth) if sub_node else {}
    return tree


def format_tree(tree: Dict[str, Dict], indent: int = 0) -> List[str]:
    lines: List[str] = []
    for name, sub in tree.items():
        desc = get_desc(name)
        prefix = "  " * indent
        line = f"{prefix}- {name}"
        if desc:
            line += f": {desc}"
        lines.append(line + "\n")
        lines.extend(format_tree(sub, indent + 1))
    return lines


def tree_edges(parent: str, tree: Dict[str, Dict], edges: List[tuple[str, str]]) -> None:
    for child, sub in tree.items():
        edges.append((parent, child))
        tree_edges(child, sub, edges)


def write_svg(root: str, tree: Dict[str, Dict], path: str) -> None:
    nodes: List[str] = [root]
    edges: List[tuple[str, str]] = []
    tree_edges(root, tree[root], edges)
    def collect_nodes(subtree: Dict[str, Dict]) -> None:
        for name, sub in subtree.items():
            nodes.append(name)
            collect_nodes(sub)
    collect_nodes(tree[root])

    level_counts: Dict[int, int] = {}
    coords: Dict[str, tuple[int, int]] = {}

    def assign(name: str, sub: Dict[str, Dict], depth: int = 0) -> None:
        y = level_counts.get(depth, 0) * 70 + 30
        x = depth * 180 + 60
        coords[name] = (x, y)
        level_counts[depth] = level_counts.get(depth, 0) + 1
        for child, child_sub in sub.items():
            assign(child, child_sub, depth + 1)

    assign(root, tree[root])

    width = max(x for x, _ in coords.values()) + 80
    height = max(y for _, y in coords.values()) + 40

    with open(path, "w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n')
        for parent, child in edges:
            x1, y1 = coords[parent]
            x2, y2 = coords[child]
            f.write(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black"/>\n')
        for name, (x, y) in coords.items():
            f.write(f'<rect x="{x - 50}" y="{y - 15}" width="100" height="30" fill="#eef" stroke="#333"/>\n')
            f.write(f'<text x="{x}" y="{y}" font-size="10" text-anchor="middle" alignment-baseline="middle">{name}</text>\n')
        f.write('</svg>')


def main() -> None:
    md_lines: List[str] = ["# High-Level Training Call Tree\n"]
    os.makedirs("docs", exist_ok=True)
    for script in TARGETS:
        with open(script, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=script)
        imports = collect_imports(tree)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("launch")]
        md_lines.append(f"## {script}\n")
        if imports:
            md_lines.append("### Imports\n")
            for imp in imports:
                md_lines.append(f"- {imp}\n")
        for func in funcs:
            call_tree = build_tree(func.name, func)
            md_lines.append(f"\n### `{func.name}` flow\n")
            md_lines.extend(format_tree(call_tree))
            md_lines.append("\n```mermaid\n")
            md_lines.append("graph TD\n")
            root_node = script.replace('.', '_')
            md_lines.append(f"    {root_node}[{script}] --> {func.name}\n")
            edges: List[tuple[str, str]] = []
            tree_edges(func.name, call_tree, edges)
            for parent, child in edges:
                md_lines.append(f"    {parent.replace('.', '_')} --> {child.replace('.', '_')}\n")
            md_lines.append("```\n")

            svg_tree = {script: {func.name: call_tree}}
            svg_path = os.path.join("docs", f"{os.path.splitext(script)[0]}_call_tree.svg")
            write_svg(script, svg_tree, svg_path)
            md_lines.append(f"![{script} call tree]({os.path.splitext(script)[0]}_call_tree.svg)\n")
        md_lines.append("\n")

    with open(os.path.join("docs", "training_call_tree.md"), "w", encoding="utf-8") as f:
        f.writelines(md_lines)


if __name__ == "__main__":
    main()

