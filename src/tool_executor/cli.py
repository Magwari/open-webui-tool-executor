"""
CLI entry point for tool-executer-server.

Provides start, stop, and status commands for managing the server process.
"""

import argparse
import logging
import os
import signal
import sys
import subprocess
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)

# PID file location
_PID_DIR = Path.home() / '.cache' / 'tool_executor'
_PID_FILE = _PID_DIR / 'server.pid'


def _ensure_pid_dir() -> None:
    """Ensure the PID file directory exists."""
    _PID_DIR.mkdir(parents=True, exist_ok=True)


def _write_pid(pid: int) -> None:
    """Write process PID to file."""
    _ensure_pid_dir()
    _PID_FILE.write_text(str(pid))


def _read_pid() -> Optional[int]:
    """Read process PID from file."""
    if not _PID_FILE.exists():
        return None
    try:
        return int(_PID_FILE.read_text().strip())
    except (ValueError, IOError):
        return None


def _remove_pid() -> None:
    """Remove PID file."""
    if _PID_FILE.exists():
        _PID_FILE.unlink()


def _is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is currently running."""
    if sys.platform == 'win32':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_INFORMATION = 0x0400

        try:
            handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    else:
        # Linux/macOS: os.kill(pid, 0) checks if process exists without sending a signal
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def cmd_start(args: argparse.Namespace) -> None:
    """Start the server as a background process."""
    pid = _read_pid()
    if pid and _is_process_running(pid):
        log.info(f'Server is already running (PID: {pid})')
        return

    # Clean up stale PID file
    if pid:
        _remove_pid()
        log.info('Removed stale PID file')

    from .config import settings

    cmd = [
        sys.executable, '-m', 'uvicorn',
        'tool_executor.main:app',
        '--host', settings.SERVER_HOST,
        '--port', str(settings.SERVER_PORT),
    ]

    log.info(f'Starting server: {" ".join(cmd)}')

    try:
        if sys.platform == 'win32':
            # Windows: create detached process with no console window
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            # Linux/macOS: start in new session, redirect output to log file
            log_path = os.getenv('LOG_PATH')
            if log_path:
                log_file = Path(log_path)
                log_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                log_file = _PID_DIR / 'server.log'
            with open(log_file, 'a') as f_out:
                process = subprocess.Popen(
                    cmd,
                    stdout=f_out,
                    stderr=f_out,
                    start_new_session=True,
                )

        _write_pid(process.pid)
        log.info(f'Server started (PID: {process.pid})')
        log.info(f'Listening on {settings.SERVER_HOST}:{settings.SERVER_PORT}')
    except Exception as e:
        log.error(f'Failed to start server: {e}')
        sys.exit(1)


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop the running server."""
    pid = _read_pid()

    if not pid:
        log.warning('No PID file found. Server may not be running.')
        return

    if not _is_process_running(pid):
        log.info('Process is not running. Cleaning up PID file.')
        _remove_pid()
        return

    try:
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_TERMINATE = 1

            handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
            if handle:
                kernel32.TerminateProcess(handle, 0)
                kernel32.CloseHandle(handle)
                log.info(f'Server stopped (PID: {pid})')
            else:
                log.error(f'Could not open process {pid}')
                sys.exit(1)
        else:
            # Linux/macOS: send SIGTERM for graceful shutdown
            os.kill(pid, signal.SIGTERM)
            log.info(f'Server stopped (PID: {pid})')
    except Exception as e:
        log.error(f'Failed to stop server: {e}')
        sys.exit(1)
    finally:
        _remove_pid()


def cmd_status(args: argparse.Namespace) -> None:
    """Show the server status."""
    pid = _read_pid()

    if not pid:
        log.info('Server is not running (no PID file found)')
        return

    if _is_process_running(pid):
        log.info(f'Server is running (PID: {pid})')
    else:
        log.warning('Server is not running (stale PID file found)')
        _remove_pid()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='tool-executer-server',
        description='Manage the Tool Executor Server',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # start command
    start_parser = subparsers.add_parser('start', help='Start the server')
    start_parser.set_defaults(func=cmd_start)

    # stop command
    stop_parser = subparsers.add_parser('stop', help='Stop the server')
    stop_parser.set_defaults(func=cmd_stop)

    # status command
    status_parser = subparsers.add_parser('status', help='Check server status')
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()