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

    async def connect(self) -> None:
        while self._retry_count < self.max_retries:
            try:
                logger.info(
                    f"Connecting to {self.url} (attempt {self._retry_count + 1}/{self.max_retries})"
                )
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
                self._retry_count += 1

                if self._retry_count >= self.max_retries:
                    raise ConnectionError(
                        f"Failed to connect after {self.max_retries} attempts: {e}"
                    )

                delay = min(2 ** (self._retry_count - 1), 60)
                logger.warning(f"Connection failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

    async def send(self, message: bytes) -> None:
        if not self._ws or not self._is_connected:
            raise ConnectionError("Not connected")
        await self._ws.send(message)

    async def receive(self) -> bytes:
        if not self._ws or not self._is_connected:
            raise ConnectionError("Not connected")

        try:
            message = await self._ws.recv()
            return message if isinstance(message, bytes) else message.encode()
        except websockets.exceptions.ConnectionClosed as e:
            self._is_connected = False

            if e.code in (1000, 1001):
                logger.info(f"Connection closed normally: {e.code}")
                raise ConnectionClosed(f"Connection closed normally: {e}")
            elif e.code in (1006, 1011, 1012):
                logger.warning(f"Connection closed abnormally: {e.code}")
                if self.auto_reconnect:
                    raise ConnectionClosed(
                        f"Connection closed abnormally: {e}", recoverable=True
                    )
                else:
                    raise ConnectionClosed(f"Connection closed abnormally: {e}")
            else:
                logger.error(f"Connection closed with unexpected code: {e.code}")
                raise ConnectionClosed(f"Connection closed: {e}")

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
        self._is_connected = False
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
