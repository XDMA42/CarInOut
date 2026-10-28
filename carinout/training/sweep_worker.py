"""参数遍历的训练进程入口。

每个进程只通过自己的管道与主进程通信：接收任务，回传进度和结果。停止信号是无锁的共享字节。
这样某个进程意外退出（如内存不足被系统结束）时，不会留下其他进程或主进程要等待的锁或队列。

本模块顶层只导入标准库：torch 等在 worker_main 里导入，导入失败（如 Windows 上虚拟内存不足导致
WinError 1455）时能把错误原因发回主进程，而不是只看到进程退出。
"""

from __future__ import annotations

import signal
import traceback


def worker_main(conn, data_file: str, stop_flag, threads: int) -> None:
    # 终端里按 Ctrl+C 时子进程也会收到，交给主进程统一停止（保存已训练的结果）
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import pickle

        import torch

        from .sweep import run_trial

        torch.set_num_threads(threads)
        with open(data_file, "rb") as fh:
            data = pickle.load(fh)
    except BaseException:  # noqa: BLE001 —— 启动失败的原因都要报告给主进程
        _send(conn, ("init_error", traceback.format_exc()))
        return
    if not _send(conn, ("ready",)):
        return

    while True:
        try:
            task = conn.recv()
        except (EOFError, OSError):  # 主进程已关闭管道
            return
        if task is None:
            return
        index, cfg_dict, run_dir = task
        try:
            row = run_trial(index, cfg_dict, run_dir, data, conn.send, lambda: bool(stop_flag.value))
        except Exception:  # noqa: BLE001 —— 单个组合出错只影响它自己
            row = {"status": "失败", "error": last_line(traceback.format_exc())}
        if not _send(conn, ("result", index, row)):
            return


def last_line(text: str) -> str:
    """traceback 的最后一行（异常类型和说明）。"""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1].strip() if lines else text


def _send(conn, message) -> bool:
    try:
        conn.send(message)
    except (OSError, ValueError):  # 主进程已退出
        return False
    return True
