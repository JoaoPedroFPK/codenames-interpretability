"""Allow ``python -m codenames_interpretability`` as an alternative to the
console script ``codenames-experiment``. Useful in Colab when the
``pip install -e .`` console-script shim isn't on PATH yet.
"""

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
