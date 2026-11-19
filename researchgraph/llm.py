"""Single shared API client (chat + embeddings) for the whole app."""
import sys

from . import ROOT

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api_client import ApiError, Client  # noqa: E402,F401

_client = None


def client() -> "Client":
    global _client
    if _client is None:
        _client = Client()
    return _client
