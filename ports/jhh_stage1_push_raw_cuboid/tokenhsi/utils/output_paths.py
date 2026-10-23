"""Keep Stage1 verification artifacts under this checkout's output_etc/check."""
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]


def _under(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_output_path(experiment, mode, requested='', check=False,
                        max_iterations=0, root=ROOT):
    """Resolve paths without creating directories or touching run metadata."""
    root = Path(root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', experiment):
        raise ValueError('Invalid experiment name')
    if mode not in ('train', 'eval'):
        raise ValueError('Output mode must be train or eval')
    production = root / 'output'
    checks = root / 'output_etc' / 'check'
    path = Path(requested) if requested else None
    if path is not None:
        path = (path if path.is_absolute() else root / path).resolve()
        if path == production:
            path = None
    named_check = False
    if path is not None and _under(path, root):
        named_check = any(
            re.search(r'(^|[_-])(check|smoke|validation)([_-]|$)', part, re.I)
            for part in path.relative_to(root).parts
        )
    verification = (mode == 'eval' or check or int(max_iterations) > 0
                    or named_check or path is not None and _under(path, checks))
    if verification:
        if path is None:
            path = checks / experiment / mode
        elif _under(path, production):
            path = checks / path.relative_to(production)
        if not _under(path.resolve(), checks):
            raise ValueError('Verification output must be under ' + str(checks))
    elif path is None:
        path = production / experiment
    if _under(path, root):
        return path.relative_to(root).as_posix()
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--mode', choices=('train', 'eval'), required=True)
    parser.add_argument('--output', default='')
    parser.add_argument('--check', type=int, choices=(0, 1), default=0)
    parser.add_argument('--max-iterations', type=int, default=0)
    args = parser.parse_args()
    try:
        path = resolve_output_path(args.experiment, args.mode, args.output,
                                   bool(args.check), args.max_iterations)
    except ValueError as error:
        parser.error(str(error))
    print('[output] ' + path, file=sys.stderr)
    print(path)


if __name__ == '__main__':
    main()
