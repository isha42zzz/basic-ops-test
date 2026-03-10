#!/usr/bin/env python3
"""
CSV 测试链路里的业务加密工具：
- seal/unseal: 使用标准 AEAD (AES-GCM) 保护 JSON 负载
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

TOKEN_VERSION = 1
NONCE_SIZE = 12
AAD = b"basic-ops-test/csv-secure-channel/v1"


def _validate_key(key: bytes) -> None:
    if len(key) not in (16, 24, 32):
        raise ValueError(f"invalid AES-GCM key length: {len(key)}")


def seal_json(payload: dict[str, Any], key: bytes) -> str:
    _validate_key(key)
    plain = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    nonce = os.urandom(NONCE_SIZE)
    ciphertext_and_tag = AESGCM(key).encrypt(nonce, plain, AAD)
    token = bytes([TOKEN_VERSION]) + nonce + ciphertext_and_tag
    return base64.urlsafe_b64encode(token).decode("ascii")


def unseal_json(token: str, key: bytes) -> dict[str, Any]:
    _validate_key(key)
    raw = base64.urlsafe_b64decode(token.encode("ascii"))
    min_len = 1 + NONCE_SIZE + 16
    if len(raw) < min_len:
        raise ValueError("sealed token too short")

    version = raw[0]
    if version != TOKEN_VERSION:
        raise ValueError(f"unsupported sealed token version: {version}")

    nonce = raw[1 : 1 + NONCE_SIZE]
    ciphertext_and_tag = raw[1 + NONCE_SIZE :]
    try:
        plain = AESGCM(key).decrypt(nonce, ciphertext_and_tag, AAD)
    except InvalidTag as exc:
        raise ValueError("sealed token authentication failed") from exc
    return json.loads(plain.decode("utf-8"))
