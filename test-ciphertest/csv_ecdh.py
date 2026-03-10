#!/usr/bin/env python3
"""
使用 OpenSSL X25519 做会话密钥协商：
- generate_x25519_keypair_pem
- derive_x25519_shared_secret
"""

from __future__ import annotations

import hashlib
import hmac
import subprocess
import tempfile
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, capture_output=True)


def generate_x25519_keypair_pem() -> tuple[bytes, bytes]:
    with tempfile.TemporaryDirectory(prefix="csv-x25519-") as td:
        root = Path(td)
        priv = root / "priv.pem"
        pub = root / "pub.pem"

        _run(["openssl", "genpkey", "-algorithm", "X25519", "-out", str(priv)])
        _run(["openssl", "pkey", "-in", str(priv), "-pubout", "-out", str(pub)])

        return priv.read_bytes(), pub.read_bytes()


def derive_x25519_shared_secret(
    private_key_pem: bytes,
    peer_public_key_pem: bytes,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="csv-x25519-") as td:
        root = Path(td)
        priv = root / "priv.pem"
        peer = root / "peer.pub.pem"
        out = root / "secret.bin"

        priv.write_bytes(private_key_pem)
        peer.write_bytes(peer_public_key_pem)

        _run(
            [
                "openssl",
                "pkeyutl",
                "-derive",
                "-inkey",
                str(priv),
                "-peerkey",
                str(peer),
                "-out",
                str(out),
            ]
        )
        return out.read_bytes()


def derive_session_key(
    shared_secret: bytes,
    session_id: str,
    nonce: str,
    context: str = "basic-ops-test/csv-bench/v1",
) -> bytes:
    """
    使用共享密钥派生会话业务密钥（32 字节）。
    """
    material = f"{context}|{session_id}|{nonce}".encode("utf-8")
    return hmac.new(shared_secret, material, hashlib.sha256).digest()
