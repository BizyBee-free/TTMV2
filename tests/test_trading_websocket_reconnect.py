"""WebSocket transport reconnect — no trading logic."""

import asyncio

import pytest

from vendor.dnse.trading_websocket.connection import WebSocketConnection
from vendor.dnse.trading_websocket.exceptions import ConnectionClosed, ConnectionError


@pytest.mark.asyncio
async def test_receive_not_connected_raises_recoverable_when_auto_reconnect():
    conn = WebSocketConnection("wss://example.test", auto_reconnect=True)
    conn._is_connected = False
    conn._ws = None

    with pytest.raises(ConnectionClosed) as exc_info:
        await conn.receive()

    assert exc_info.value.recoverable is True


@pytest.mark.asyncio
async def test_receive_not_connected_raises_connection_error_when_no_auto_reconnect():
    conn = WebSocketConnection("wss://example.test", auto_reconnect=False)
    conn._is_connected = False

    with pytest.raises(ConnectionError, match="Not connected"):
        await conn.receive()


@pytest.mark.asyncio
async def test_connect_resets_stale_retry_count_and_retries(monkeypatch):
    """After exhausting retries, a new connect() must not fail instantly with 'Not connected'."""
    import websockets

    conn = WebSocketConnection("wss://example.test", max_retries=3)
    conn._retry_count = 99  # stale state from prior failed reconnect session
    attempts: list[int] = []

    class _FakeWs:
        async def close(self) -> None:
            return None

    async def _fake_connect(*_a, **_k):
        attempts.append(1)
        if len(attempts) < 2:
            raise OSError("getaddrinfo failed")
        return _FakeWs()

    monkeypatch.setattr(websockets, "connect", _fake_connect)
    await conn.connect(max_attempts=3)
    assert conn.is_connected
    assert len(attempts) == 2
    assert conn._retry_count == 0


@pytest.mark.asyncio
async def test_prepare_reconnect_clears_stale_socket():
    conn = WebSocketConnection("wss://example.test")
    conn._is_connected = True
    conn._retry_count = 10

    class _Stale:
        closed = False

        async def close(self) -> None:
            self.closed = True

    conn._ws = _Stale()
    await conn.prepare_reconnect()
    assert not conn.is_connected
    assert conn._retry_count == 0
    assert conn._ws is None


def test_client_recoverable_connection_loss_detection():
    from vendor.dnse.trading_websocket.client import TradingClient

    assert TradingClient._is_recoverable_connection_loss(
        ConnectionClosed("gone", recoverable=True)
    )
    assert TradingClient._is_recoverable_connection_loss(ConnectionError("Not connected"))
    assert TradingClient._is_recoverable_connection_loss(ConnectionResetError())
    assert not TradingClient._is_recoverable_connection_loss(
        ConnectionClosed("bye", recoverable=False)
    )
    assert not TradingClient._is_recoverable_connection_loss(ValueError("other"))
