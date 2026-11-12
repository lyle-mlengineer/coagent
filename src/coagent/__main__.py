"""Allow `python -m coagent` (handy inside a Colab cell)."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
