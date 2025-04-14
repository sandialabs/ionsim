"""Make a documentation stub for every module in the ionsim source
directory."""

import argparse
import os
import string
from typing import Union
from pathlib import Path
from dataclasses import dataclass


def main():
    args = get_args()
    srcdir = find_directory(args.src, ['src', 'ionsim'], Path(os.getcwd()))
    dstdir = find_directory(args.dst, ['docs', 'source', 'index.rst'], Path(os.getcwd()))
    module_template = read_template(args.module_template)
    package_template = read_template(args.package_template)
    package = assemble_package(srcdir)
    if package is None:
        raise ValueError(f"{srcdir} is not a package")
    write_package_stubs(package, dstdir, module_template, package_template,
                        overwrite=args.overwrite, update_toc=args.update_toc)


def get_args():
    parser = argparse.ArgumentParser(description="Make documentation stubs for all library modules")

    parser.add_argument('--src', help="The root directory for the source files. If omitted, the script will recursively back up from the current directory until it finds a src/ionsim set of directories, then use the src directory as src.")
    parser.add_argument('--dst', help="The directory to add the stub files to. If not given, tries to find docs/source/index.rst and using the source directory.")
    parser.add_argument('--module-template', help="The template file to create stubs from using the filename of the modules", required=True)
    parser.add_argument('--package-template', help="The template file to create stubs from using the filename of the packages", required=True)
    parser.add_argument('--overwrite', help="Overwrite existing module files with new stubs", action='store_true')
    parser.add_argument('--update-toc', help="Try to update existing subpackage rst files with any new modules", action='store_true')

    return parser.parse_args()


def find_directory(user_path, default_path, working_dir):
    """Recursively find the desired directory if the user has not given one explicitly."""
    if user_path is not None:
        return Path(user_path)
    if path_matches(default_path, working_dir):
        return working_dir.joinpath(*default_path).parent
    if working_dir.parent == working_dir:
        # The above is only true when we're at the filesystem or drive
        # root
        raise ValueError(f"Could not find {default_path}")
    return find_directory(user_path, default_path, working_dir.parent)


def path_matches(default_path, working_dir):
    """Return if working_dir has a subdirectory or file, possibly
    nested, of the form default_path."""
    return working_dir.joinpath(*default_path).exists()


def read_template(template_filename):
    with open(template_filename, 'r') as fd:
        return string.Template(fd.read())


@dataclass
class Module:
    """Represent a module that needs to be documented."""
    name: str


@dataclass
class Package:
    """Represent a package and all its submodules and subpackages."""
    name: str
    children: list[Union['Package', Module]]


def assemble_package(srcdir, prefixes=None):
    """Recursively create a Package object representing a simplified
    version of the directory tree."""
    in_package = (srcdir / "__init__.py").exists()
    if not in_package and prefixes is not None:
        # We allow srcdir to not be a package at the top level so that
        # we can see the name of the top level package by starting in
        # its containing directory.
        return None
    prefixes = prefixes or []
    children = []
    with os.scandir(srcdir) as entries:
        for entry in entries:
            entry = Path(entry)
            entry_prefixes = [*prefixes, entry.stem]
            if entry.suffix == '.py' and in_package and entry.name != '__init__.py':
                children.append(Module(make_name(entry_prefixes)))
            elif entry.is_dir():
                subpackage = assemble_package(srcdir / entry, entry_prefixes)
                if subpackage:
                    children.append(subpackage)
    return Package(name=make_name(prefixes), children=children)


def make_name(prefixes):
    return '.'.join(prefixes)


def write_package_stubs(package, dstdir, module_template, package_template, *, overwrite, update_toc):
    """Recursivly create all stub files needed for this (sub)package"""
    if package.name:
        write_package_stub(package, dstdir, package_template, update_toc=update_toc)
    for child in package.children:
        if isinstance(child, Module):
            write_module_stub(child, dstdir, module_template, overwrite=overwrite)
        elif isinstance(child, Package):
            write_package_stubs(child, dstdir, module_template, package_template,
                                overwrite=overwrite, update_toc=update_toc)
        else:
            raise ValueError(f"Unknown package child {child}")


def make_string_from_template(template, **kwargs):
    """Given a string.Template object, create the restructured text
    contents. This involves both a simple substitution and a
    length-matching of equal sign underlines."""
    s = template.substitute(kwargs)
    lastlen = None
    lines = []
    for line in s.split('\n'):
        sline = line.strip()
        if not sline:
            modline = ''
        elif all(c == '=' for c in sline):
            if lastlen is None:
                raise ValueError(f"First line cannot be equals signs")
            modline = '=' * lastlen
        elif all(c == '-' for c in sline):
            if lastlen is None:
                raise ValueError(f"First line cannot be minus signs")
            modline = '-' * lastlen
        else:
            lastlen = len(line)
            modline = line
        lines.append(modline)
    return '\n'.join(lines)


def write_module_stub(module, dstdir, template, overwrite=False):
    """Create the output file and write the stub. If overwrite is
    False (default), do nothing if the file already exists."""
    contents = make_string_from_template(template, module=module.name)
    path = dstdir / (module.name + '.rst')
    if path.exists():
        if overwrite:
            print(f"Overwriting {module.name}")
        else:
            print(f"Ignoring existing module {module.name}")
            return
    with open(path, 'w') as fd:
        fd.write(contents)


def write_package_stub(package, dstdir, template, update_toc=False):
    """Create the stub for a package, complete with a table of
    contents for all modules in that package. Write the stub to disk.
    """
    children_str = '\n   '.join(child.name for child in package.children)
    contents = make_string_from_template(template, package=package.name, children=children_str)
    path = dstdir / (package.name + '.rst')
    if path.exists():
        if update_toc:
            print(f"Updating table of contents in {package.name}")
            with open(path, 'r') as fd:
                old_contents = fd.read()
            contents = update_toc_string(package, old_contents)
        else:
            print(f"Ignoring existing package {package.name}")
            return
    with open(path, 'w') as fd:
        fd.write(contents)


def update_toc_string(package, contents):
    ref_by_line = {child.name: None for child in package.children}
    lines = contents.split('\n')
    last = 0
    for i, line in enumerate(lines):
        sline = line.strip()
        if sline in ref_by_line:
            ref_by_line[sline] = i
            if i > last:
                last = i
                prefix = line[:len(line) - len(line.lstrip())]
    for ref, idx in ref_by_line.items():
        if idx is None:
            lines.insert(last + 1, prefix + ref)
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
