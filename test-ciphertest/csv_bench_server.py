#!/usr/bin/env python3
"""
CSV VM 侧服务端：
1) /attest/start: 根据客户端 nonce 生成真实 CSV 报告，并完成 X25519 密钥协商
2) /bench/run: 仅接受已证明 session，处理加密请求并返回加密结果

注意：本服务必须运行在包含 /dev/csv-guest 的 CSV 虚拟机里。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import ssl
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

from csv_attestation import AttestationReportProducor
from csv_bench_common import run_plaintext_bench_from_payload
from csv_ecdh import (
    derive_session_key,
    derive_x25519_shared_secret,
    generate_x25519_keypair_pem,
)
from csv_secure_channel import seal_json, unseal_json


def _now() -> float:
    return time.time()


@dataclass
class Session:
    nonce: str
    report_path: Path
    created_at: float
    workload_secret: bytes


class CSVBenchServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        artifacts_dir: Path,
        session_ttl_sec: int,
    ) -> None:
        super().__init__(server_address, CSVBenchHandler)
        self.artifacts_dir = artifacts_dir
        self.session_ttl_sec = session_ttl_sec
        self.sessions: dict[str, Session] = {}
        self.sessions_lock = Lock()


class CSVBenchHandler(BaseHTTPRequestHandler):
    server: CSVBenchServer

    def do_POST(self) -> None:  # noqa: N802 (stdlib hook name)
        if self.path == "/attest/start":
            self._handle_attest_start()
            return
        if self.path == "/bench/run":
            self._handle_bench_run()
            return
        self._write_json(404, {"error": f"unknown path: {self.path}"})

    def _read_json(self) -> dict[str, Any]:
        try:
            content_len = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("invalid Content-Length") from None

        body = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            raise ValueError("invalid json body") from None
        if not isinstance(data, dict):
            raise ValueError("json body must be object")
        return data

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _purge_expired_sessions(self) -> None:
        now = _now()
        ttl = self.server.session_ttl_sec
        with self.server.sessions_lock:
            dead = [
                sid
                for sid, sess in self.server.sessions.items()
                if now - sess.created_at > ttl
            ]
            for sid in dead:
                self.server.sessions.pop(sid, None)

    def _handle_attest_start(self) -> None:
        try:
            data = self._read_json()
        except ValueError as exc:
            self._write_json(400, {"error": str(exc)})
            return

        nonce = str(data.get("nonce", ""))
        client_pubkey_b64 = str(data.get("client_pubkey_b64", ""))
        nonce_bytes = nonce.encode("utf-8")
        if not nonce or len(nonce_bytes) > 32:
            self._write_json(400, {"error": "nonce must be 1~32 bytes"})
            return
        if not client_pubkey_b64:
            self._write_json(400, {"error": "client_pubkey_b64 is required"})
            return

        try:
            client_pubkey_pem = base64.b64decode(client_pubkey_b64.encode("ascii"))
        except Exception:
            self._write_json(400, {"error": "client_pubkey_b64 invalid"})
            return

        if not Path("/dev/csv-guest").exists():
            self._write_json(500, {"error": "/dev/csv-guest not found on server"})
            return

        self._purge_expired_sessions()

        session_id = secrets.token_hex(16)
        report_path = self.server.artifacts_dir / f"csv-report-{session_id}.bin"

        try:
            server_privkey_pem, server_pubkey_pem = generate_x25519_keypair_pem()
            pubkey_hash16_hex = hashlib.sha256(server_pubkey_pem).hexdigest()[:32]
            report_userdata = f"{nonce}{pubkey_hash16_hex}"
            producer = AttestationReportProducor(report_userdata)
            producer.persistent_report(str(report_path))
            report_bytes = report_path.read_bytes()
            shared_secret = derive_x25519_shared_secret(
                private_key_pem=server_privkey_pem,
                peer_public_key_pem=client_pubkey_pem,
            )
            workload_secret = derive_session_key(
                shared_secret=shared_secret,
                session_id=session_id,
                nonce=nonce,
            )
        except Exception as exc:
            self._write_json(500, {"error": f"failed to generate attestation report: {exc}"})
            return

        with self.server.sessions_lock:
            self.server.sessions[session_id] = Session(
                nonce=nonce,
                report_path=report_path,
                created_at=_now(),
                workload_secret=workload_secret,
            )

        self._write_json(
            200,
            {
                "session_id": session_id,
                "nonce": nonce,
                "report_b64": base64.b64encode(report_bytes).decode("ascii"),
                "report_size": len(report_bytes),
                "server_pubkey_b64": base64.b64encode(server_pubkey_pem).decode("ascii"),
            },
        )

    def _handle_bench_run(self) -> None:
        try:
            data = self._read_json()
        except ValueError as exc:
            self._write_json(400, {"error": str(exc)})
            return

        session_id = str(data.get("session_id", ""))
        nonce = str(data.get("nonce", ""))
        sealed_request = str(data.get("sealed_request", ""))

        self._purge_expired_sessions()

        # 原子消费 session，避免并发请求复用同一 session 重放执行。
        with self.server.sessions_lock:
            session = self.server.sessions.get(session_id)
            if session is None:
                self._write_json(403, {"error": "invalid or expired session_id"})
                return
            if session.nonce != nonce:
                self._write_json(403, {"error": "nonce mismatch"})
                return
            session = self.server.sessions.pop(session_id)

        try:
            bench_req = unseal_json(sealed_request, session.workload_secret)
        except Exception as exc:
            self._write_json(400, {"error": f"invalid sealed_request: {exc}"})
            return

        start = time.perf_counter()
        try:
            run_summary = run_plaintext_bench_from_payload(
                bench_req,
                session.workload_secret,
            )
        except Exception as exc:
            self._write_json(500, {"error": f"benchmark execution failed: {exc}"})
            return

        elapsed = time.perf_counter() - start
        results = list(run_summary["results"])
        all_pass = bool(run_summary["all_pass"])
        profile = str(run_summary["profile"])
        seed = int(run_summary["seed"])

        plain_result = {
            "session_id": session_id,
            "profile": profile,
            "seed": seed,
            "bench_exec_sec": elapsed,
            # Keep legacy field for compatibility with older clients.
            "server_elapsed_sec": elapsed,
            "all_pass": all_pass,
            "results": results,
        }
        sealed_result = seal_json(plain_result, session.workload_secret)

        self._write_json(
            200,
            {
                "session_id": session_id,
                "sealed_result": sealed_result,
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="CSV 基准测试服务端（运行在 CSV VM）")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=18080, help="监听端口")
    parser.add_argument(
        "--artifacts-dir",
        default="test-ciphertest/artifacts",
        help="报告文件目录",
    )
    parser.add_argument(
        "--session-ttl-sec",
        type=int,
        default=300,
        help="session 有效期（秒）",
    )
    parser.add_argument("--tls-cert", default="", help="TLS 证书路径（PEM）")
    parser.add_argument("--tls-key", default="", help="TLS 私钥路径（PEM）")
    parser.add_argument(
        "--require-tls",
        action="store_true",
        help="要求启用 TLS（建议生产开启）",
    )
    args = parser.parse_args()

    if not Path("/dev/csv-guest").exists():
        raise SystemExit("ERROR: /dev/csv-guest not found; this server must run on CSV VM")

    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    server = CSVBenchServer(
        server_address=(args.host, args.port),
        artifacts_dir=artifacts_dir,
        session_ttl_sec=args.session_ttl_sec,
    )

    if args.require_tls and (not args.tls_cert or not args.tls_key):
        raise SystemExit("ERROR: --require-tls requires --tls-cert and --tls-key")
    if bool(args.tls_cert) != bool(args.tls_key):
        raise SystemExit("ERROR: --tls-cert and --tls-key must be provided together")

    if args.tls_cert and args.tls_key:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=args.tls_cert, keyfile=args.tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        print(f"CSV bench server listening with TLS on {args.host}:{args.port}")
    else:
        print(f"CSV bench server listening on {args.host}:{args.port} (NO TLS)")

    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
