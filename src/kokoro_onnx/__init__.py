from .cli import app
from .count_params import count_params
from .export import export
from .profile_community_onnx import prof_community
from .quantize import quantize
from .verify import verify

__all__ = ["count_params", "export", "prof_community", "verify", "quantize"]


def main() -> None:
    app()
