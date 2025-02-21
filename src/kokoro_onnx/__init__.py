from .cli import app
from .cli_count_params import count
from .cli_export import export
from .cli_quantize import export_optimized
from .cli_verify import verify

__all__ = ["count", "export", "verify", "export_optimized"]


def main() -> None:
    app()
