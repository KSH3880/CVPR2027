"""Rebase bundled runtime manifests after moving the handoff directory."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    for template in sorted((ROOT / 'handoff_support/manifests').glob('*.yaml')):
        destination = ROOT / 'tokenhsi/data' / template.name
        destination.write_text(template.read_text().replace('@ROOT@', str(ROOT)))
        print('prepared:', destination.relative_to(ROOT))
    from preflight import main as preflight
    preflight()


if __name__ == '__main__':
    main()
