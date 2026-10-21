"""Keep upstream tests out of the main checkout's default collection."""
import os

if os.environ.get('TOKENHSI_KSH_PORT_TESTS') != '1':
    collect_ignore_glob = ['test_*.py']
