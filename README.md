# basic-ops-test

一个用于基础算子性能验证的测试项目，包含两条链路：

- 明文基准测试：本地直接运行 Numpy 算子。
- 密文基准测试：基于 CSV 远程证明 + 会话密钥 + 加密请求执行算子。

## 目录结构

- `test-plaintext/plaintext_bench.py`：明文基准入口
- `test-ciphertest/csv_bench_server.py`：密文服务端（需运行在 CSV VM）
- `test-ciphertest/csv_bench_client.py`：密文客户端
- `test-ciphertest/csv_bench_common.py`：算子与计时统计公共逻辑

## 环境要求

- Python `>=3.11`
- 依赖：`numpy`、`cryptography`、`snowland-smx`
- 运行密文服务端时，机器需存在 `/dev/csv-guest`

## 快速运行

### 1) 明文基准

```bash
uv run python test-plaintext/plaintext_bench.py
```

### 2) 密文基准

服务端（CSV VM 内）：

```bash
uv run python test-ciphertest/csv_bench_server.py --host 0.0.0.0 --port 18080
```

客户端（主机侧）：

```bash
uv run python test-ciphertest/csv_bench_client.py \
  --server http://<vm-ip>:18080 \
  --allow-insecure-http \
  --profile quick \
  --seed 42
```

## 指标说明

每个算子单独统计：

- `decrypt_data_sec`：该算子测试数据解密/解包耗时
- `compute_sec`：该算子计算耗时
- `elapsed_sec`：`decrypt_data_sec + compute_sec`

服务端汇总字段：

- `bench_exec_sec`：服务端基准执行窗口耗时（不等同于端到端网络总耗时）

客户端字段：

- `client_total_elapsed`：客户端端到端总耗时（含证明、协商、网络、结果解密等）
