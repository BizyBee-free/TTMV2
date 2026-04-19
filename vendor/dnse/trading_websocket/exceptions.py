class TradingWebSocketError(Exception):
    """Base exception for trading websocket errors."""
    pass


class ConnectionError(TradingWebSocketError):
    """Connection-related errors."""
    pass


class ConnectionClosed(TradingWebSocketError):
    """Connection was closed."""

    def __init__(self, message="Connection closed", recoverable=False):
        super().__init__(message)
        self.recoverable = recoverable


class AuthenticationError(TradingWebSocketError):
    """Authentication failed."""
    pass


class SubscriptionError(TradingWebSocketError):
    """Subscription-related errors."""
    pass


class EncodingError(TradingWebSocketError):
    """Message encoding/decoding errors."""
    pass
