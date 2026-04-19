import hmac
import hashlib
import time
from typing import Dict, Any, Union


class AuthManager:
    """HMAC-SHA256 authentication manager for WebSocket connections."""

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret

    def create_auth_message(self) -> Dict[str, Any]:
        timestamp = int(time.time())
        # Gateway DNSE (msgpack) từ chối khi ``nonce`` là int — báo ``nonce is required``; dùng chuỗi số.
        nonce = str(int(time.time() * 1_000_000))

        signature = self.compute_signature(timestamp, nonce)

        return {
            "action": "auth",
            "api_key": self.api_key,
            "signature": signature,
            "timestamp": timestamp,
            "nonce": nonce,
        }

    def compute_signature(self, timestamp: int, nonce: Union[int, str]) -> str:
        message = f"{self.api_key}:{timestamp}:{nonce}"

        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return signature
