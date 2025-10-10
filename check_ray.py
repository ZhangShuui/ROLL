#!/usr/bin/env python3
import os, sys, time, json, socket
import subprocess
import ray


def run(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=10)
        return out.decode("utf-8", "ignore").strip()
    except Exception as e:
        return f"[run-error] {e}"


def main():
    addr = f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    expect = int(os.environ.get("EXPECT_NODES", "1"))
    node_ip = os.environ.get("RAY_OVERRIDE_NODE_IP_ADDRESS") or os.environ.get("THIS_NODE_IP")
    deadline = time.time() + 360  # 最多 6 分钟
    attempt = 0

    print("enter python ", flush=True)
    print("imported", flush=True)
    print(f"[wait-ray] connecting to {addr}", flush=True)
    print(f"[wait-ray] expecting {expect} nodes", flush=True)
    print(f"[wait-ray] deadline at {time.ctime(deadline)}", flush=True)
    if node_ip:
        print(f"[wait-ray] forcing _node_ip_address={node_ip}", flush=True)

    # 先探测 GCS 端口是否可达（避免一上来 init 就卡）
    try:
        s = socket.create_connection((os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"])), timeout=3)
        s.close()
    except Exception as e:
        print(f"[wait-ray] WARNING: cannot connect to GCS {addr}: {e}", flush=True)

    last_nodes = []
    while time.time() < deadline:
        attempt += 1
        print(f"[wait-ray] trying to init at {time.ctime()} (attempt {attempt})", flush=True)
        try:
            # 关键：把 _node_ip_address 也传进去，避免 SDK 走错网卡
            ray.init(
                address=addr,
                namespace="roll_wait",
                ignore_reinit_error=True,
                _node_ip_address=node_ip if node_ip else None,
            )
            nodes = [n for n in ray.nodes() if n.get("Alive", False)]
            addrs = [n.get("NodeManagerAddress") for n in nodes]
            print(f"[wait-ray] alive nodes = {len(nodes)} -> {addrs}", flush=True)
            if len(nodes) >= expect:
                # 打印资源概览
                resources = ray.cluster_resources()
                print("[wait-ray] cluster resources:", json.dumps(resources, indent=2, sort_keys=True), flush=True)
                print("[wait-ray] SUCCESS", flush=True)
                return 0
            last_nodes = addrs
        except Exception as e:
            print("[wait-ray] init error:", repr(e), flush=True)
        time.sleep(3)

    print("[wait-ray] TIMEOUT", flush=True)
    # 打印一次 ray status，帮助诊断
    try:
        print("[wait-ray] ray status:", flush=True)
        out = run(f"ray status --address='{addr}'")
        print(out, flush=True)
    except Exception as e:
        print("[wait-ray] status error:", e, flush=True)
    # 额外打印已见到的节点
    print(f"[wait-ray] last seen nodes: {last_nodes}", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
