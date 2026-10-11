"""Griffe extension: document the source folders that have no ``__init__.py``.

Deliberate workaround. Nine agent folders (``ajax/agents/SAC``, ``PPO``, ...)
and ``ajax/logging`` have no ``__init__.py``. Python imports them as implicit
namespace packages, but Griffe, which reads the source for mkdocstrings, skips a
folder without one inside a regular package, so ``ajax.agents.SAC.SAC.SAC``
"could not be found" and the build aborts. This extension adds each such folder
as a module and reads its files, as Griffe does for any other folder.

Remove it, and its line in ``mkdocs.yml``, once every folder under ``src/ajax``
has an ``__init__.py``: it then finds nothing to do.
"""

from pathlib import Path

import griffe


class NamespaceFolders(griffe.Extension):
    def on_package(self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **_kwargs):
        if pkg.name != "ajax":
            return
        root = Path(pkg.filepath).parent
        # Sorted, so a folder is added before the folders inside it.
        for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
            if (folder / "__init__.py").exists() or not any(folder.glob("*.py")):
                continue
            parent = pkg
            for part in folder.relative_to(root).parent.parts:
                parent = parent.get_member(part)
            module = griffe.Module(folder.name, filepath=[folder], parent=parent)
            parent.set_member(folder.name, module)
            for path in sorted(folder.glob("*.py")):
                code = path.read_text(encoding="utf-8")
                loader.lines_collection[path] = code.splitlines()
                submodule = griffe.visit(
                    path.stem,
                    filepath=path,
                    code=code,
                    extensions=loader.extensions,
                    parent=module,
                    docstring_parser=loader.docstring_parser,
                    docstring_options=loader.docstring_options,
                    lines_collection=loader.lines_collection,
                    modules_collection=loader.modules_collection,
                )
                module.set_member(path.stem, submodule)
