# Test the promoted checkout and import Isaac Gym before torch.
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE / 'tokenhsi'), str(SOURCE), str(SOURCE.parent)]
import isaacgym
from ksh_jointpush.ksh_torch_compat import install as install_torch_compat
install_torch_compat()
