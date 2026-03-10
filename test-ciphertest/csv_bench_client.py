#!/usr/bin/env python3
"""
本地主机客户端：
1) 请求 CSV VM 服务端生成 attestation report（绑定 nonce）
2) 使用 csv_attestation.py 中 Verifier 在本地验签
3) 客户端/服务端通过 X25519 公钥协商派生会话密钥
4) 使用会话密钥保护 bench 请求和结果
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import ssl
import time
from pathlib import Path
from typing import Any

import urllib.error
import urllib.request

from csv_attestation import AttestationReportVerifier
from csv_bench_common import build_client_bench_payload
from csv_ecdh import (
    derive_session_key,
    derive_x25519_shared_secret,
    generate_x25519_keypair_pem,
)
from csv_secure_channel import (
    seal_json,
    unseal_json,
)


def decode_userdata_text(userdata_hex: str) -> str:
    raw = bytes.fromhex(userdata_hex)
    # csv_attestation 里 userdata 为 64 字节，短字符串右对齐，左侧补 0
    stripped = raw.lstrip(b"\x00")
    return stripped.decode("utf-8")


def verify_attestation_report(
    report_path: Path,
    expected_userdata: str,
    require_csv_policy: bool = True,
) -> dict[str, Any]:
    verifier = AttestationReportVerifier(str(report_path))
    if not verifier.verify_signature():
        raise RuntimeError("attestation report signature verify failed")

    claims = verifier.parse_attestation_report(print_json=False)

    userdata_hex = str(claims.get("Userdata", ""))
    userdata_text = decode_userdata_text(userdata_hex)
    if userdata_text != expected_userdata:
        raise RuntimeError(
            f"attestation userdata mismatch: got={userdata_text}, expected={expected_userdata}"
        )

    if require_csv_policy:
        policy_items = claims.get("POLICY_ITEMS", [])
        if "CSV" not in policy_items:
            raise RuntimeError(f"policy does not include CSV: {policy_items}")

    return claims


def print_bench_result(item: dict[str, Any]) -> None:
    status = "PASS" if item["ok"] else "FAIL"
    decrypt_sec = float(item.get("decrypt_data_sec", item.get("decrypt_sec", 0.0)))
    compute_sec = float(item.get("compute_sec", 0.0))
    print(
        f"{item['name']}: {status} | total {item['elapsed_sec']:.4f}s "
        f"(decrypt_data {decrypt_sec:.4f}s + compute {compute_sec:.4f}s) / "
        f"limit {item['limit_sec']:.2f}s | {item['detail']}"
    )


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout_sec: float,
    ssl_context: ssl.SSLContext | None = None,
) -> tuple[int, dict[str, Any] | str]:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec, context=ssl_context) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8")
            if body:
                try:
                    return status, json.loads(body)
                except json.JSONDecodeError:
                    return status, body
            return status, {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return exc.code, parsed
    except urllib.error.URLError as exc:
        return 0, f"network error: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="CSV 基准测试客户端（运行在本地主机）")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:18080",
        help="CSV VM 服务端地址，例如 https://10.0.0.2:18080",
    )
    parser.add_argument(
        "--profile",
        choices=["full", "quick"],
        default="full",
        help="基准规模",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=120.0,
        help="HTTP 超时秒数",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="test-ciphertest/artifacts",
        help="本地保存 report 的目录",
    )
    parser.add_argument(
        "--ca-cert",
        default="",
        help="服务端 TLS CA 证书路径（HTTPS 时可选）",
    )
    parser.add_argument(
        "--insecure-skip-verify",
        action="store_true",
        help="HTTPS 时跳过证书校验（仅测试）",
    )
    parser.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="允许使用 HTTP（仅测试，不建议）",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ssl_context: ssl.SSLContext | None = None
    if args.server.startswith("https://"):
        if args.insecure_skip_verify:
            ssl_context = ssl._create_unverified_context()  # noqa: SLF001
        elif args.ca_cert:
            ssl_context = ssl.create_default_context(cafile=args.ca_cert)
        else:
            ssl_context = ssl.create_default_context()
    elif args.server.startswith("http://"):
        if not args.allow_insecure_http:
            raise RuntimeError(
                "Refusing insecure HTTP. Use https://... or pass --allow-insecure-http for test only."
            )
    else:
        raise RuntimeError("server must start with http:// or https://")

    nonce = secrets.token_hex(16)
    start = time.perf_counter()
    client_priv_pem, client_pub_pem = generate_x25519_keypair_pem()

    # Step 1: 请求 report
    status, payload = post_json(
        url=f"{args.server.rstrip('/')}/attest/start",
        payload={
            "nonce": nonce,
            "client_pubkey_b64": base64.b64encode(client_pub_pem).decode("ascii"),
        },
        timeout_sec=args.timeout_sec,
        ssl_context=ssl_context,
    )
    if status != 200 or not isinstance(payload, dict):
        raise SystemExit(f"attest/start failed: HTTP {status}, body={payload}")

    session_id = str(payload["session_id"])
    report_b64 = str(payload["report_b64"])
    server_pubkey_b64 = str(payload["server_pubkey_b64"])
    report_path = artifacts_dir / f"client-received-{session_id}.bin"
    report_path.write_bytes(base64.b64decode(report_b64.encode("ascii")))
    server_pub_pem = base64.b64decode(server_pubkey_b64.encode("ascii"))
    pubkey_hash16_hex = hashlib.sha256(server_pub_pem).hexdigest()[:32]
    expected_userdata = f"{nonce}{pubkey_hash16_hex}"

    # Step 2: 本地验签 + nonce/policy 校验
    claims = verify_attestation_report(
        report_path=report_path,
        expected_userdata=expected_userdata,
    )
    print("[1/3] Attestation verify: PASS")
    print(
        f"      CHIP_ID={claims.get('CHIP_ID')}, POLICY={claims.get('POLICY_ITEMS')}, "
        f"report={report_path}"
    )

    # Step 3: X25519 公钥协商会话密钥
    shared_secret = derive_x25519_shared_secret(
        private_key_pem=client_priv_pem,
        peer_public_key_pem=server_pub_pem,
    )
    workload_secret = derive_session_key(
        shared_secret=shared_secret,
        session_id=session_id,
        nonce=nonce,
    )
    print("[2/3] X25519 session key derived")

    # Step 4: 客户端生成原始测试数据并加密发送给服务端
    bench_request_payload = build_client_bench_payload(
        profile=args.profile,
        seed=args.seed,
        key=workload_secret,
    )
    sealed_request = seal_json(bench_request_payload, workload_secret)
    run_status, bench_payload = post_json(
        url=f"{args.server.rstrip('/')}/bench/run",
        payload={
            "session_id": session_id,
            "nonce": nonce,
            "sealed_request": sealed_request,
        },
        timeout_sec=args.timeout_sec,
        ssl_context=ssl_context,
    )
    if run_status != 200 or not isinstance(bench_payload, dict):
        raise SystemExit(f"bench/run failed: HTTP {run_status}, body={bench_payload}")
    print("[3/3] Encrypted bench request: PASS")

    sealed_result = str(bench_payload.get("sealed_result", ""))
    bench_data = unseal_json(sealed_result, workload_secret)
    bench_exec_sec = float(
        bench_data.get("bench_exec_sec", bench_data.get("server_elapsed_sec", 0.0))
    )
    print("      Encrypted bench response decrypted")
    print(
        f"      bench_exec={bench_exec_sec:.4f}s, "
        f"all_pass={bench_data['all_pass']}"
    )
    for item in bench_data["results"]:
        print_bench_result(item)

    total = time.perf_counter() - start
    print(f"client_total_elapsed={total:.4f}s")
    return 0 if bool(bench_data["all_pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
