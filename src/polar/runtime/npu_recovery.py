"""Paused-window Box16 maintenance, shared by Rollout and a live-run monitor.

Existing session cancellation and lease release are deliberately unchanged.
Quarantine markers fence new NPU leases even in gateways started before this module.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import signal
import time
import urllib.request

from polar.config.topology import NpuRecoveryConfig
from polar.runtime.factory import create_runtime
from polar.runtime.models import RuntimeSpec

LOG = logging.getLogger(__name__)
SETTLE_SECONDS = 150
BUSY_WAIT_SECONDS = 60
PROBE_TIMEOUT = 90

# Do not authorize anything until the native tool actually reaches its prompt.
RESET_PROGRAM = r'''
import errno,json,os,pty,select,subprocess,sys,time
master,slave=pty.openpty()
p=subprocess.Popen(['/usr/local/bin/npu-smi','set','-t','reset','-i',sys.argv[1],'-c','0'],stdin=slave,stdout=slave,stderr=slave)
os.close(slave);started=time.monotonic();text='';answers=0;timed_out=False
while True:
    if time.monotonic()-started > (600 if answers else 120):
        timed_out=True;p.kill();break
    ready,_,_=select.select([master],[],[],.2)
    if ready:
        try:chunk=os.read(master,65536)
        except OSError as exc:
            if exc.errno==errno.EIO:break
            raise
        if not chunk:break
        part=chunk.decode(errors='replace');text+=part;print(part,end='',flush=True)
        if answers==0 and 'are you sure' in text.lower():
            os.write(master,b'y\n');answers=1
        if answers==1 and 'do you want to continue reboot?' in text.lower():
            os.write(master,b'y\n');answers=2
    if p.poll() is not None and not ready:break
try:p.wait(timeout=2)
except subprocess.TimeoutExpired:pass
os.close(master)
print('\nPOLAR_RESET_RESULT '+json.dumps({'return_code':p.poll(),'confirmed':answers>0,'timed_out':timed_out,'child_reaped':p.poll() is not None,'success':p.poll()==0 and 'Reset server successfully' in text}),flush=True)
'''

PROBE_PROGRAM = r'''
import json,os
import torch
import torch_npu
card=int(os.environ['ASCEND_RT_VISIBLE_DEVICES']);torch.npu.set_device(0)
x_cpu=(torch.arange(4096,dtype=torch.float32).reshape(64,64)%7)-3
x=x_cpu.to('npu:0');w=torch.ones((64,64),device='npu:0')
torch.testing.assert_close((x+2).cpu(),x_cpu+2,rtol=0,atol=0)
torch.testing.assert_close((x@w).cpu(),x_cpu@torch.ones(64,64),rtol=0,atol=0)
torch.npu.synchronize();del x,w;torch.npu.empty_cache();torch.npu.synchronize()
print('POLAR_NPU_HEALTH '+json.dumps({'card':card,'ok':True}),flush=True)
'''


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def try_lock(path: Path) -> int | None:
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise


class NpuRecovery:
    def __init__(self, config: NpuRecoveryConfig, control_dir: Path, gateway_url: str):
        self.config = config
        self.control_dir = control_dir.resolve()
        self.directory = self.control_dir / "npu-recovery"
        self.path = self.directory / "state.json"
        self.locks = Path(config.lock_dir)
        self.gateway_url = gateway_url.rstrip("/")
        self.task: asyncio.Task | None = None

    def snapshot(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict):
                raise ValueError("recovery state is not an object")
            return data
        except FileNotFoundError:
            return {"state": "idle", "resume_allowed": True}
        except (OSError, ValueError):
            return {"state": "unreadable", "resume_allowed": False}

    def resume_error(self) -> str | None:
        state = self.snapshot()
        if not state.get("resume_allowed", False):
            return f"NPU recovery {state.get('state')}; inspect {self.path}"
        return None

    def _save(self, state: dict, phase: str, **values) -> None:
        state.update(state=phase, updated_at=time.time(), **values)
        write_json(self.path, state)
        write_json(self.directory / state["job_id"] / "state.json", state)
        LOG.info("NPU recovery %s: %s", state["transition_id"], phase)

    def start(self, record: dict) -> asyncio.Task | None:
        if record.get("kind") != "update" or record.get("phase") != "ready_for_training":
            return None
        job_id = hashlib.sha256(json.dumps([record["policy_namespace"], record["transition_id"]]).encode()).hexdigest()[:24]
        self.locks.mkdir(parents=True, exist_ok=True)
        owner = try_lock(self.locks / "npu-recovery.lock")
        if owner is None:
            return None
        previous = self.snapshot()
        if previous.get("job_id") == job_id or not previous.get("resume_allowed", False):
            os.close(owner)
            return None
        faults = [c for c in self.config.pool if self._marker(c).exists()]
        if not faults:
            os.close(owner)
            return None
        state = {"job_id": job_id, "transition_id": record["transition_id"],
                 "policy_namespace": record["policy_namespace"], "fault_cards": faults,
                 "started_at": time.time(), "resume_allowed": False, "owner_pid": os.getpid()}
        try:
            self._save(state, "pending")
            self.task = asyncio.create_task(self._run(state, owner), name=f"npu-recovery-{job_id}")
        except BaseException:
            os.close(owner)
            raise
        return self.task

    def _marker(self, card: int) -> Path:
        return self.locks / f"npu{card}.quarantine.json"

    def _window_matches(self, state: dict) -> bool:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 60
        while True:
            current = json.loads((self.control_dir / "policy-transition.json").read_text())["current"]
            if any(current.get(k) != state[k] for k in ("transition_id", "policy_namespace")):
                return False
            if current.get("phase") != "ready_for_training":
                return False
            try:
                with opener.open(self.gateway_url + "/admin/inference/status", timeout=5) as response:
                    gate = json.load(response)
                return gate.get("paused") is True and all(gate.get(k) == state[k] for k in ("transition_id", "policy_namespace"))
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                # Cleanup can briefly delay the Gateway; retry only the read, never reset.
                LOG.warning("Gateway status unavailable; rechecking the training window")
                time.sleep(1)

    @staticmethod
    def _remove_unchanged(path: Path, expected: bytes) -> bool:
        if not path.exists() or path.read_bytes() != expected:
            return False
        path.unlink()
        return True

    async def _pause_collector(self) -> list[int]:
        """Pause only our idle query collector, never training/evaluation processes."""
        target = Path(__file__).resolve().parents[3] / "deploy/ascend_operator/telemetry/npu_smi_snapshot.py"
        paused = []
        try:
            for proc in Path("/proc").iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    if os.fsencode(target) not in (proc / "cmdline").read_bytes().split(b"\0"):
                        continue
                    if (proc / "stat").read_text().rsplit(")", 1)[1].split()[0] == "T":
                        continue
                    fd = os.pidfd_open(int(proc.name))
                except (FileNotFoundError, ProcessLookupError):
                    continue
                stopped = False
                try:
                    for _ in range(40):
                        signal.pidfd_send_signal(fd, signal.SIGSTOP)
                        stopped = True
                        await asyncio.sleep(.05)
                        if not (proc / "task" / proc.name / "children").read_text().strip():
                            paused.append(fd)
                            fd = None
                            break
                        signal.pidfd_send_signal(fd, signal.SIGCONT)
                        stopped = False
                        await asyncio.sleep(.2)
                    else:
                        raise RuntimeError("NPU query collector still has an in-flight query")
                finally:
                    if fd is not None:
                        try:
                            if stopped:
                                signal.pidfd_send_signal(fd, signal.SIGCONT)
                        except ProcessLookupError:
                            pass
                        finally:
                            os.close(fd)
            return paused
        except BaseException:
            self._resume_collectors(paused)
            raise

    @staticmethod
    def _resume_collectors(paused: list[int]) -> None:
        for fd in paused:
            try:
                signal.pidfd_send_signal(fd, signal.SIGCONT)
            except ProcessLookupError:
                pass
            finally:
                os.close(fd)

    async def _run(self, state: dict, owner: int) -> None:
        holds: dict[int, int] = {}
        markers: dict[int, bytes] = {}
        created: set[int] = set()
        runtime = None
        paused = []
        collector_lock = None
        reset_pending = False
        reset_finished = False
        job = self.directory / state["job_id"]
        try:
            model = Path("/sys/class/dmi/id/product_name").read_text().strip()
            if model != "Atlas 200T A2 Box16":
                raise RuntimeError(f"Unsupported reset platform: {model}")
            if not await asyncio.to_thread(self._window_matches, state):
                raise RuntimeError("Training pause window is no longer current")
            for card in self.config.pool:
                marker = self._marker(card)
                temporary = marker.with_name(f".recovery-marker-{card}-{os.getpid()}")
                try:
                    if not marker.exists():
                        write_json(temporary, {"schema_version": 1, "device_id": str(card),
                            "source": "npu_recovery", "reason": "maintenance_in_progress",
                            "job_id": state["job_id"]})
                        try:
                            os.link(temporary, marker)
                            created.add(card)
                        except FileExistsError:
                            pass
                finally:
                    temporary.unlink(missing_ok=True)
                markers[card] = marker.read_bytes()
            self._save(state, "waiting_for_existing_cleanup")
            deadline = time.monotonic() + BUSY_WAIT_SECONDS
            while True:
                for card in created:
                    if card not in holds:
                        fd = try_lock(self.locks / f"npu{card}.lock")
                        if fd is not None:
                            holds[card] = fd
                if created <= holds.keys():
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Healthy-card evaluations are still releasing; no reset attempted")
                await asyncio.sleep(1)
            # Fault-card leases deliberately do not prevent the reset attempt.
            if not await asyncio.to_thread(self._window_matches, state):
                raise RuntimeError("Training pause window ended before reset")
            paused = await self._pause_collector()
            collector_dir = self.locks / "npu-smi-snapshot"
            collector_dir.mkdir(exist_ok=True)
            collector_lock = try_lock(collector_dir / "collector.lock")
            if collector_lock is None and not paused:
                raise RuntimeError("Query collector is active outside this PID namespace; no reset attempted")
            spec = RuntimeSpec(image=self.config.image, workdir="/polar/session", cpus=8,
                memory_mb=16384, env={"ASCEND_SLOG_PRINT_TO_STDOUT": "0", "ASCEND_GLOBAL_LOG_LEVEL": "3", "OMP_NUM_THREADS": "1"},
                kwargs={"ascend": {"pool": list(self.config.pool), "lock_dir": str(self.locks), "lease_at_start": False}})
            runtime = create_runtime(spec, f"npu-recovery-{state['job_id']}", job)
            await runtime.start()
            if not await asyncio.to_thread(self._window_matches, state):
                raise RuntimeError("Training pause window ended while starting maintenance container")
            self._save(state, "resetting")
            reset_pending = True
            result = await runtime.exec(shlex.join(["python", "-u", "-c", RESET_PROGRAM, str(state["fault_cards"][0])]), timeout_sec=630)
            (job / "reset.log").write_text((result.stdout or "") + (result.stderr or ""))
            replies = [json.loads(l.split("POLAR_RESET_RESULT ", 1)[1]) for l in (result.stdout or "").splitlines() if l.startswith("POLAR_RESET_RESULT ")]
            if result.return_code != 0 or len(replies) != 1:
                raise RuntimeError("Unknown reset outcome; keep the whole reset domain isolated")
            reply = replies[0]
            state["reset_result"] = reply
            if not reply["confirmed"]:
                reset_pending = False
                raise RuntimeError("Native reset never reached confirmation; no reset authorized")
            if reply["timed_out"] or not reply["child_reaped"]:
                raise RuntimeError("Reset did not finish; keep the whole reset domain isolated")
            reset_pending = False
            reset_finished = True
            self._save(state, "settling", reset_result=reply)
            await asyncio.sleep(SETTLE_SECONDS)
            self._save(state, "checking")
            checks = []
            for card in self.config.pool:
                if card not in holds:
                    fd = try_lock(self.locks / f"npu{card}.lock")
                    if fd is not None:
                        holds[card] = fd
                if card not in holds:
                    checks.append({"card": card, "ok": False, "reason": "old_lease_still_held"})
                    continue
                try:
                    check = await runtime.exec(shlex.join(["timeout", "-k", "5s", str(PROBE_TIMEOUT), "python", "-c", PROBE_PROGRAM]),
                        env={"ASCEND_RT_VISIBLE_DEVICES": str(card)}, timeout_sec=PROBE_TIMEOUT + 10)
                    (job / f"card{card}.log").write_text((check.stdout or "") + (check.stderr or ""))
                    records = [json.loads(l.split("POLAR_NPU_HEALTH ", 1)[1]) for l in (check.stdout or "").splitlines() if l.startswith("POLAR_NPU_HEALTH ")]
                    ok = check.return_code == 0 and records == [{"card": card, "ok": True}]
                    checks.append({"card": card, "ok": ok, "return_code": check.return_code})
                except Exception as exc:
                    checks.append({"card": card, "ok": False, "reason": str(exc)})
            await runtime.stop()
            runtime = None
            available = []
            for check in checks:
                card = check["card"]
                can_clear = check["ok"] and (reply["success"] or card in created)
                marker = self._marker(card)
                if can_clear and (not marker.exists() or self._remove_unchanged(marker, markers[card])):
                    available.append(card)
                elif not marker.exists() or marker.read_bytes() == markers[card]:
                    data = json.loads(markers[card])
                    data["last_recovery"] = {"job_id": state["job_id"], **check, "reset_success": reply["success"]}
                    write_json(marker, data)
            self._save(state, "complete", checks=checks, available_cards=available,
                       resume_allowed=bool(available), reset_result=reply)
        except BaseException as exc:
            if not reset_pending and not reset_finished:
                for card in created:
                    if card in markers and self._marker(card).exists():
                        self._remove_unchanged(self._marker(card), markers[card])
            available = [c for c in self.config.pool if not self._marker(c).exists()]
            self._save(state, "unknown" if reset_pending or reset_finished else "not_started",
                       error=str(exc), available_cards=available,
                       resume_allowed=bool(available) and not reset_pending and not reset_finished)
            LOG.exception("NPU maintenance failed; existing session cleanup was not changed")
        finally:
            if runtime is not None and not reset_pending:
                try:
                    await runtime.stop()
                except Exception:
                    LOG.exception("Maintenance container cleanup failed")
            if reset_pending:
                # An unconfirmed completion must not restart device queries.
                for fd in paused:
                    os.close(fd)
                LOG.error("Reset outcome unknown; query collector remains paused for manual recovery")
            else:
                self._resume_collectors(paused)
            if collector_lock is not None:
                os.close(collector_lock)
            for fd in holds.values():
                os.close(fd)
            os.close(owner)


async def monitor(profile: Path, control_dir: Path, once: bool) -> None:
    import yaml
    data = yaml.safe_load(profile.read_text())
    settings = data["operator_runtime"]
    if settings.get("npu_recovery", {}).get("enabled") is not True:
        raise ValueError("operator_runtime.npu_recovery.enabled must be true")
    config = NpuRecoveryConfig(image=data["operator"]["runtime"]["image"],
        lock_dir=settings["npu_lease"]["lock_dir"], pool=settings["npu_lease"]["pool"])
    recovery = NpuRecovery(config, control_dir, data["service"]["gateway_url"])
    recovery.directory.mkdir(parents=True, exist_ok=True)
    fd = try_lock(recovery.directory / "monitor.lock")
    if fd is None:
        raise RuntimeError("A live-run NPU recovery monitor is already running")
    try:
        initial = json.loads((control_dir / "policy-transition.json").read_text())["current"]
        namespace = initial["policy_namespace"]
        LOG.info("Armed for namespace=%s; current phase=%s, transition=%s", namespace, initial["phase"], initial["transition_id"])
        while True:
            current = json.loads((control_dir / "policy-transition.json").read_text())["current"]
            if current["policy_namespace"] != namespace:
                LOG.info("Trainer namespace changed; leaving the new run untouched")
                return
            write_json(recovery.directory / "monitor.json", {"pid": os.getpid(), "checked_at": time.time(),
                "phase": current["phase"], "transition_id": current["transition_id"], "policy_namespace": namespace})
            if current.get("kind") == "update" and current["phase"] == "ready_for_training":
                task = recovery.start(current)
                if task:
                    await task
                if once:
                    return
            await asyncio.sleep(2)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(monitor(args.profile, args.control_dir, args.once))


if __name__ == "__main__":
    main()
