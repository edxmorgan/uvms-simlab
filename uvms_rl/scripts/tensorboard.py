"""Launch TensorBoard for uvms_rl training runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _default_logdir() -> str:
    return str(Path.home() / "ros_ws" / "recordings" / "rl_runs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start TensorBoard for uvms_rl training logs.")
    parser.add_argument("--logdir", default=_default_logdir(), help="TensorBoard log directory")
    parser.add_argument("--port", type=int, default=6006, help="TensorBoard port")
    parser.add_argument("--host", default="localhost", help="TensorBoard host")
    parser.add_argument("--reload-interval", type=float, default=5.0, help="TensorBoard reload interval in seconds")
    parser.add_argument("--bind-all", action="store_true", help="Bind to all network interfaces")
    parser.add_argument("tensorboard_args", nargs=argparse.REMAINDER, help="Extra arguments passed to TensorBoard")
    args = parser.parse_args()

    try:
        from tensorboard.main import run_main
    except ImportError as exc:
        raise SystemExit("TensorBoard is not installed. Install it with: python3 -m pip install tensorboard") from exc

    logdir = str(Path(args.logdir).expanduser())
    tb_args = [
        "tensorboard",
        "--logdir",
        logdir,
        "--port",
        str(args.port),
        "--reload_interval",
        str(args.reload_interval),
    ]
    if args.bind_all:
        tb_args.append("--bind_all")
    else:
        tb_args.extend(["--host", args.host])
    if args.tensorboard_args:
        extra = args.tensorboard_args
        if extra and extra[0] == "--":
            extra = extra[1:]
        tb_args.extend(extra)

    print(f"TensorBoard logdir: {logdir}")
    print(f"TensorBoard URL: http://{args.host if not args.bind_all else 'localhost'}:{args.port}")
    sys.argv = tb_args
    run_main()


if __name__ == "__main__":
    main()
