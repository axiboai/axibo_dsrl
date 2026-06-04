"""WebSocket policy client compatible with piperx-openpi's flat-obs server protocol.

The nakamotoo openpi-client wraps requests as ``{method, obs}``; piperx-openpi's
``websocket_policy_server.py`` passes the unpacked message straight to
``policy.infer(obs)``.  That mismatch causes ``KeyError: 'state'`` on the server.

This client sends a **flat** observation dict (state / images / prompt).  For DSRL
noise steering, ``noise`` is embedded in the payload; the policy server must pop it
before calling ``policy.infer(obs, noise=noise)`` — see the server patch in
``examples/scripts/run_piper.sh`` comments.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import websockets.sync.client

from openpi_client import msgpack_numpy


class PiperWebsocketClientPolicy:
    """Flat-obs websocket client for piperx-openpi policy servers."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = 8000,
        api_key: Optional[str] = None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict[str, Any]:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info("Waiting for policy server at %s ...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: Dict[str, Any], noise=None) -> Dict[str, Any]:
        payload = dict(obs)
        if noise is not None:
            payload["noise"] = noise
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        pass
