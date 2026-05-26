import asyncio
import logging
import ssl
from typing import Optional, AsyncIterator

import certifi
import websockets
from websockets.client import WebSocketClientProtocol
from .exceptions import ConnectionError, ConnectionClosed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class WebSocketConnection:
    """WebSocket connection manager with automatic reconnection."""

    def __init__(
        self,
        url: str,
        timeout: float = 60.0,
        heartbeat_interval: float = 25.0,
        auto_reconnect: bool = True,
        max_retries: int = 10,
    ):
        self.url = url
        self.timeout = timeout
        self.heartbeat_interval = heartbeat_interval
        self.auto_reconnect = auto_reconnect
        self.max_retries = max_retries

        self._ws: Optional[WebSocketClientProtocol] = None
        self._retry_count = 0
        self._is_connected = False

    async def prepare_reconnect(self) -> None:
        """Close stale socket and reset retry budget for a new connect session."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._mark_disconnected()
        self._retry_count = 0

    async def connect(self, *, max_attempts: Optional[int] = None) -> None:
        """Connect with retries. Each call gets a fresh retry budget (fixes stuck reconnect)."""
        await self.prepare_reconnect()
        limit = int(max_attempts) if max_attempts is not None else int(self.max_retries)
        limit = max(1, limit)
        last_err: Optional[Exception] = None
        for attempt in range(1, limit + 1):
            try:
                logger.info(f"Connecting to {self.url} (attempt {attempt}/{limit})")
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                # Gateway DNSE + PyPI ``dnse`` dùng ``websockets.connect(..., ssl=True)`` không bật ping tự động;
                # ping_interval mặc định có thể gây đóng 1006 sớm với một số phiên bản websockets.
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        self.url,
                        ssl=ssl_context,
                        ping_interval=None,
                        ping_timeout=None,
                        close_timeout=10,
                        max_queue=None,
                    ),
                    timeout=self.timeout,
                )

                self._is_connected = True
                self._retry_count = 0
                logger.info("Connected successfully")
                return

            except (websockets.exceptions.WebSocketException, OSError) as e:
                last_err = e
                self._mark_disconnected()
                if attempt >= limit:
                    raise ConnectionError(
                        f"Failed to connect after {limit} attempts: {e}"
                    ) from e

                delay = min(2 ** (attempt - 1), 60)
                logger.warning(f"Connection failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

        raise ConnectionError(
            f"Failed to connect after {limit} attempts: {last_err or 'unknown'}"
        )

    def _mark_disconnected(self) -> None:
        self._is_connected = False
        self._ws = None

    def _raise_not_connected(self) -> None:
        """Signal listen loop to reconnect instead of spinning on ConnectionError."""
        if self.auto_reconnect:
            raise ConnectionClosed("Not connected", recoverable=True)
        raise ConnectionError("Not connected")

    def _connection_closed_exc(
        self, e: websockets.exceptions.ConnectionClosed
    ) -> ConnectionClosed:
        self._mark_disconnected()
        if e.code in (1000, 1001):
            logger.info(f"Connection closed normally: {e.code}")
            return ConnectionClosed(f"Connection closed normally: {e}")
        logger.warning(f"Connection closed: code={e.code} reason={e.reason!r}")
        if self.auto_reconnect:
            return ConnectionClosed(f"Connection closed: {e}", recoverable=True)
        return ConnectionClosed(f"Connection closed: {e}")

    async def send(self, message: bytes) -> None:
        if not self._ws or not self._is_connected:
            self._raise_not_connected()
        try:
            await self._ws.send(message)
        except websockets.exceptions.ConnectionClosed as e:
            raise self._connection_closed_exc(e) from e

    async def receive(self) -> bytes:
        if not self._ws or not self._is_connected:
            self._raise_not_connected()

        try:
            message = await self._ws.recv()
            return message if isinstance(message, bytes) else message.encode()
        except websockets.exceptions.ConnectionClosed as e:
            raise self._connection_closed_exc(e) from e

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
        self._mark_disconnected()
        logger.info("Connection closed")

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._ws is not None

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self

    async def __anext__(self) -> bytes:
        try:
            return await self.receive()
        except ConnectionClosed:
            raise StopAsyncIteration
