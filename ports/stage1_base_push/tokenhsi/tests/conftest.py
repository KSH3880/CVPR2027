"""Keep this private snapshot's test modules out of main-checkout collection."""
import os

if os.environ.get('TOKENHSI_STAGE1_BASE_PUSH_TESTS') != '1':
    collect_ignore_glob = ['test_*.py']
