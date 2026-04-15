"""Optuna storage backend with fsspec support."""

import errno
import json
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

from optuna.storages.journal import (
    BaseJournalBackend,
)
from optuna.storages.journal._file import (
    LOCK_FILE_SUFFIX,
    RENAME_FILE_SUFFIX,
    BaseJournalFileLock,
    get_lock_file,
)
from upath import UPath


class JournalFsspecFileOpenLock(BaseJournalFileLock):
    """Lock class for synchronizing processes using file open locks with fsspec support."""

    def __init__(self, filepath: str | Path | UPath, grace_period: int | None = 30) -> None:
        self._lock_file: UPath = UPath(filepath).with_suffix(LOCK_FILE_SUFFIX)
        if grace_period is not None:
            if grace_period <= 0:
                raise ValueError("The value of `grace_period` should be a positive integer.")
            if grace_period < 3:
                warnings.warn("The value of `grace_period` might be too small.", stacklevel=1)
        self.grace_period = grace_period

    def acquire(self) -> bool:
        """Acquire a lock in a blocking way by creating a lock file.

        Returns:
            :obj:`True` if it succeeded in creating a ``self._lock_file``.

        Raises:
            OSError: If it failed to create a lock file due to reasons other than the file
                already existing.
            BaseException: If an unexpected exception occurred during lock acquisition.

        """
        sleep_secs = 0.001
        last_update_monotonic_time = time.monotonic()
        mtime = None
        while True:
            try:
                with self._lock_file.open("xb"):
                    return True
            except OSError as err:
                if err.errno == errno.EEXIST:
                    if self.grace_period is not None:
                        try:
                            current_mtime = self._lock_file.stat().st_mtime
                        except OSError:
                            continue
                        if current_mtime != mtime:
                            mtime = current_mtime
                            last_update_monotonic_time = time.monotonic()

                        if time.monotonic() - last_update_monotonic_time > self.grace_period:
                            warnings.warn(
                                "The existing lock file has not been released "
                                "for an extended period. Forcibly releasing the lock file.",
                                stacklevel=1,
                            )
                            try:
                                self.release()
                                sleep_secs = 0.001
                            except RuntimeError:
                                continue

                    time.sleep(sleep_secs)
                    sleep_secs = min(sleep_secs * 2, 1)
                    continue
                raise err
            except BaseException:
                self.release()
                raise

    def release(self) -> None:
        """Release a lock by removing the created file."""
        lock_rename_file = self._lock_file.with_suffix("." + str(uuid.uuid4()) + RENAME_FILE_SUFFIX)
        try:
            UPath(self._lock_file).rename(lock_rename_file)
            lock_rename_file.unlink()
        except OSError as e:
            raise RuntimeError("Error: did not possess lock") from e
        except BaseException:
            lock_rename_file.unlink()
            raise


class JournalFsspecFileBackend(BaseJournalBackend):
    """File storage class for Optuna Journal log backend with fsspec support."""

    def __init__(self, file_path: UPath, lock_obj: BaseJournalFileLock | None = None) -> None:
        self._file_path: UPath = file_path
        self._lock = lock_obj or JournalFsspecFileOpenLock(self._file_path)
        if not self._file_path.exists():
            self._file_path.touch()  # Create a file if it does not exist.
        self._log_number_offset: dict[int, int] = {0: 0}

    def read_logs(self, log_number_from: int) -> list[dict[str, Any]]:
        """Read journal log entries from a specific log number."""
        logs = []
        with self._file_path.open("rb") as f:
            # Maintain remaining_log_size to allow writing by another process
            # while reading the log.
            remaining_log_size = self._file_path.stat().st_size
            log_number_start = 0
            if log_number_from in self._log_number_offset:
                f.seek(self._log_number_offset[log_number_from])
                log_number_start = log_number_from
                remaining_log_size -= self._log_number_offset[log_number_from]

            last_decode_error = None
            for log_number, line in enumerate(f, start=log_number_start):
                byte_len = len(line)
                remaining_log_size -= byte_len
                if remaining_log_size < 0:
                    break
                if last_decode_error is not None:
                    raise last_decode_error
                if log_number + 1 not in self._log_number_offset:
                    self._log_number_offset[log_number + 1] = self._log_number_offset[log_number] + byte_len
                if log_number < log_number_from:
                    continue

                # Ensure that each line ends with line separators (\n, \r\n).
                if not line.endswith(b"\n"):
                    last_decode_error = ValueError("Invalid log format.")
                    del self._log_number_offset[log_number + 1]
                    continue
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError as err:
                    last_decode_error = err
                    del self._log_number_offset[log_number + 1]
            return logs

    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        """Add new journal log entry."""
        with get_lock_file(self._lock):
            what_to_write = "\n".join([json.dumps(log, separators=(",", ":")) for log in logs]) + "\n"
            with self._file_path.open("ab") as f:
                f.write(what_to_write.encode("utf-8"))
                f.flush()
