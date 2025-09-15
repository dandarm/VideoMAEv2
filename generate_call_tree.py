import ast
import os
from typing import List

TARGETS = ["specialization.py", "classification.py", "tracking.py"]

class CallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls: List[str] = []

    def visit_Call(self, node: ast.Call):
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


def collect_imports(tree: ast.AST) -> List[str]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(f"{n.name}" + (f" as {n.asname}" if n.asname else ""))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for n in node.names:
                mod = f"{module}.{n.name}" if module else n.name
                imports.append(mod + (f" as {n.asname}" if n.asname else ""))
    return sorted(set(imports))


def collect_calls(func: ast.FunctionDef) -> List[str]:
    visitor = CallVisitor()
    visitor.visit(func)
    return sorted(set(visitor.calls))


def main():
    md_lines: List[str] = ["# Training Scripts Dependency and Call Tree\n"]
    for script in TARGETS:
        with open(script, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=script)
        imports = collect_imports(tree)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("launch")]
        md_lines.append(f"## {script}\n")
        md_lines.append("### Imports\n")
        for imp in imports:
            md_lines.append(f"- {imp}\n")
        for func in funcs:
            calls = collect_calls(func)
            md_lines.append(f"\n### Function `{func.name}` Calls\n")
            for c in calls:
                md_lines.append(f"- {c}\n")
            # Mermaid diagram
            md_lines.append("\n```mermaid\n")
            md_lines.append("graph TD\n")
            file_node = script.replace('.', '_')
            func_node = func.name
            md_lines.append(f"    {file_node}[{script}] --> {func_node}\n")
            for c in calls:
                node = c.replace('.', '_')
                md_lines.append(f"    {func_node} --> {node}\n")
            md_lines.append("```\n")
        md_lines.append("\n")

    os.makedirs("docs", exist_ok=True)
    with open(os.path.join("docs", "training_call_tree.md"), "w", encoding="utf-8") as f:
        f.writelines(md_lines)

if __name__ == "__main__":
    main()
