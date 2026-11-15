"""Exact reviewed process-session cleanup helpers, with local identity provider."""
import os,signal,subprocess,time
from pathlib import Path
from .common import require
from .lifecycle import identity as process_identity

def exact_live(identity):
    return identity is not None and process_identity(identity['pid']) == identity

def session_members(owner):
    current_leader = process_identity(owner['pid'])
    if current_leader is not None and current_leader != owner:
        return []
    members = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            current = process_identity(int(path.name))
            if current and current['session'] == owner['session'] and current['process_group'] == owner['process_group']:
                members.append(current)
    return members

def exact_signal(identity, signum):
    try:
        fd = os.pidfd_open(identity['pid'])
    except ProcessLookupError:
        return False
    try:
        if not exact_live(identity):
            return False
        try:
            signal.pidfd_send_signal(fd, signum)
            return True
        except ProcessLookupError:
            return False
    finally:
        os.close(fd)

def stop_owned_group(process, owner, grace=5):
    """Clean the exact owned session even if its leader already exited."""
    observed = session_members(owner)
    for member in observed:
        exact_signal(member, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        process.poll()
        if not session_members(owner):
            break
        time.sleep(.05)
    for member in session_members(owner):
        exact_signal(member, signal.SIGKILL)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        raise RuntimeError('Owned stage failed to exit after exact cleanup')
    deadline = time.monotonic() + grace
    while session_members(owner) and time.monotonic() < deadline:
        time.sleep(.05)
    require(not session_members(owner), 'Owned stage left a live session member')
    return {'actual_returncode': process.returncode, 'observed_members': observed, 'survivors': []}
