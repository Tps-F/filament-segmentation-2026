"""Small local file helpers; no numerical dependencies."""
import hashlib,json,os
from pathlib import Path
def require(value,message):
    if not value:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text())
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)

