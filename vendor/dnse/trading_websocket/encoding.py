import json
from typing import Any, Dict

import msgpack

from .exceptions import EncodingError


def _coerce_msgpack_keys_to_str(obj: Any) -> Any:
    """Đảm bảo key map là str (msgpack có thể decode key thành bytes → .get('T') fail)."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, bytes):
                try:
                    nk = k.decode("utf-8")
                except Exception:
                    nk = k.decode("ascii", errors="replace")
            else:
                nk = str(k)
            out[nk] = _coerce_msgpack_keys_to_str(v)
        return out
    if isinstance(obj, list):
        return [_coerce_msgpack_keys_to_str(x) for x in obj]
    return obj


class MessageEncoder:
    """Encode messages for WebSocket transmission."""

    def __init__(self, encoding: str = "msgpack"):
        if encoding not in ("json", "msgpack"):
            raise ValueError(f"Invalid encoding: {encoding}. Must be 'json' or 'msgpack'")
        self.encoding = encoding

    def encode(self, data: Dict[str, Any]) -> bytes:
        try:
            if self.encoding == "json":
                return json.dumps(data).encode("utf-8")
            else:
                return msgpack.packb(data, use_bin_type=True)
        except Exception as e:
            raise EncodingError(f"Failed to encode message: {e}")


class MessageDecoder:
    """Decode messages from WebSocket."""

    def __init__(self, encoding: str = "msgpack"):
        if encoding not in ("json", "msgpack"):
            raise ValueError(f"Invalid encoding: {encoding}. Must be 'json' or 'msgpack'")
        self.encoding = encoding

    def decode(self, data: bytes) -> Any:
        if not data:
            raise EncodingError("Empty WebSocket frame")
        if self.encoding == "json":
            try:
                return _coerce_msgpack_keys_to_str(json.loads(data.decode("utf-8")))
            except Exception:
                try:
                    try:
                        out = msgpack.unpackb(data, raw=False, strict_map_key=False)
                    except TypeError:
                        out = msgpack.unpackb(data, raw=False)
                    return _coerce_msgpack_keys_to_str(out)
                except Exception as e2:
                    raise EncodingError(
                        f"Failed to decode as JSON or msgpack (encoding=json): {e2}"
                    ) from e2

        # Mặc định msgpack — một số bản gateway vẫn gửi JSON text; thử msgpack rồi fallback JSON.
        try:
            try:
                out = msgpack.unpackb(data, raw=False, strict_map_key=False)
            except TypeError:
                out = msgpack.unpackb(data, raw=False)
            return _coerce_msgpack_keys_to_str(out)
        except Exception:
            pass
        try:
            return _coerce_msgpack_keys_to_str(json.loads(data.decode("utf-8")))
        except Exception as e:
            raise EncodingError(f"Failed to decode as msgpack or JSON: {e}")
