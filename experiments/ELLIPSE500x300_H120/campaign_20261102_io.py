"""Bounded SSH operations; never repeat an uncertain scheduler mutation."""
import contextlib
import json
import posixpath
import time
from pathlib import Path
from ehe_5e9_workflow import connection, q


def execute(host, command, timeout=180, readonly=False):
    attempts = 4 if readonly else 1
    for attempt in range(attempts):
        try:
            with connection(host) as client:
                stdin, stdout, stderr = client.exec_command(command, timeout=30)
                channel = stdout.channel
                try:
                    stdin.close()
                    channel.settimeout(timeout)
                    output = stdout.read().decode('utf8', 'replace')
                    errors = stderr.read().decode('utf8', 'replace')
                    status = channel.recv_exit_status()
                finally:
                    channel.close()
                if status:
                    raise RuntimeError(f'Remote exit {status}: {(output + errors)[-10000:]}')
                return output
        except RuntimeError:
            raise
        except Exception:
            if attempt + 1 == attempts:
                raise
            print('READ_ONLY_SSH_RETRY', host, attempt + 1, flush=True)


def python_json(host, interpreter, source, arguments=None, timeout=180, readonly=True):
    invocation = q(interpreter) + ' -B -c ' + q(source)
    if arguments is not None:
        invocation += ' ' + q(json.dumps(arguments))
    raw = execute(host, invocation, timeout, readonly)
    marker = 'CAMPAIGN_JSON '
    if marker not in raw:
        raise ValueError('Remote complete JSON marker missing: ' + raw[-2000:])
    return json.loads(raw.rsplit(marker, 1)[1])


@contextlib.contextmanager
def sftp(host):
    with connection(host) as client:
        session = client.open_sftp()
        session.get_channel().settimeout(180)
        try:
            yield session
        finally:
            session.close()


def mkdirs(session, path):
    parts = []
    cursor = path
    while cursor not in ('', '/'):
        try:
            session.stat(cursor)
            break
        except FileNotFoundError:
            parts.append(cursor)
            cursor = posixpath.dirname(cursor)
    for folder in reversed(parts):
        session.mkdir(folder)


def upload_files(host, root, files):
    """Small immutable files; partial transfers never become published inputs."""
    with sftp(host) as session:
        mkdirs(session, root)
        for relative, local in files.items():
            remote = root + '/' + relative
            mkdirs(session, posixpath.dirname(remote))
            body = Path(local).read_bytes()
            try:
                with session.open(remote, 'rb') as handle:
                    if handle.read() != body:
                        raise ValueError('Different remote immutable member: ' + remote)
                continue
            except FileNotFoundError:
                pass
            temporary = remote + '.uploading_' + str(time.time_ns())
            with session.open(temporary, 'wx') as handle:
                handle.write(body)
            session.rename(temporary, remote)
