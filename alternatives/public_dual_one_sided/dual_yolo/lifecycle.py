"""Published shared-session children; the outer controller owns complete session cleanup."""
import math,os,signal,subprocess,sys,time
from pathlib import Path
from .common import require,read,save,digest
SIGNALS={signal.SIGTERM,signal.SIGINT,signal.SIGALRM}
def identity(pid=None):
    pid=os.getpid() if pid is None else pid
    try:
        f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if f[0] in ('Z','X'):return None
        return {'pid':pid,'start_token':f[19],'process_group':int(f[2]),'session':int(f[3])}
    except (FileNotFoundError,ProcessLookupError):return None

def interrupted(sig,frame):raise TimeoutError('Original portable reproduction cutoff/interruption:'+str(sig))

def arm(deadline):
    require(type(deadline) in (int,float) and math.isfinite(deadline) and deadline>time.time(),'Original cutoff expired/unbound')
    for sig in SIGNALS:signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
    signal.pthread_sigmask(signal.SIG_SETMASK,set())

def check_time(deadline):require(time.time()<deadline,'Original cutoff expired')

def signal_exact(owner,sig):
    if identity(owner['pid'])!=owner:return False
    try:fd=os.pidfd_open(owner['pid'])
    except ProcessLookupError:return False
    try:
        if identity(owner['pid'])!=owner:return False
        try:signal.pidfd_send_signal(fd,sig);return True
        except ProcessLookupError:return False
    finally:os.close(fd)

def stop_worker(process,owner):
    # This worker shares the outer controller's owned session. Never killpg here.
    if process.poll() is None:
        signal_exact(owner,signal.SIGTERM)
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            signal_exact(owner,signal.SIGKILL);process.wait(timeout=5)
    require(identity(owner['pid'])!=owner,'Owned worker still live')
    return {'actual_worker_returncode':process.returncode,'recorded_worker_dead':True,
      'outer_owned_session_cleanup_required':True,'all_session_descendants_claimed_dead':False}

def run_child(command,*,cwd,env,output,stage,deadline,inputs):
    output=Path(output);process=owner=code=None;start=time.time();parent=identity()
    require(deadline>start,'Original child cutoff expired')
    try:
        with (output/(stage+'.log')).open('xb') as log:
            old=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS);r,w=os.pipe()
            try:
                gate='import os,signal,sys;f=int(sys.argv[1]);t=os.read(f,1);os.close(f);assert t==b"G";signal.pthread_sigmask(signal.SIG_SETMASK,set());os.execvpe(sys.argv[2],sys.argv[2:],os.environ)'
                process=subprocess.Popen([sys.executable,'-B','-c',gate,str(r),*command],cwd=cwd,env=env,
                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(r,))
                owner=identity(process.pid);require(owner is not None and owner['session']==parent['session'] and owner['process_group']==parent['process_group'],'Child escaped outer owned session')
                save(output/(stage+'_started.json'),{'worker':owner,'parent':parent,'deadline_epoch':deadline,'inputs_sha256':digest(inputs)})
                os.write(w,b'G')
            finally:os.close(r);os.close(w);signal.pthread_sigmask(signal.SIG_SETMASK,old)
            require(time.time()<deadline,'Original child cutoff exhausted');code=process.wait(timeout=deadline-time.time())
            require(type(code) is int and code==0 and time.time()<=deadline,'Actual owned child failure/cutoff')
    finally:
        if process is not None and owner is not None:
            old=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
            try:
                cleanup=stop_worker(process,owner)
                save(output/(stage+'_exited.json'),{'worker':owner,'parent':parent,'normal_wait_returncode':code,'cleanup':cleanup,
                  'started_epoch':start,'completed_epoch':time.time(),'deadline_epoch':deadline})
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,old)
    return read(output/(stage+'_exited.json'))


def admitted(inputs_path,stage):
    inputs=read(inputs_path);arm(inputs['deadline_epoch']);output=Path(inputs['output']);started=read(output/(stage+'_started.json'))
    require(started['worker']==identity() and started['parent']==identity(os.getppid())==inputs['parent']
      and started['inputs_sha256']==digest(inputs_path) and started['deadline_epoch']==inputs['deadline_epoch'],'Unpublished portable child')
    require(identity()['session']==inputs['parent']['session'] and identity()['process_group']==inputs['parent']['process_group'],'Portable child session differs')
    return inputs


def parent_live(inputs):
    require(identity(os.getppid())==inputs['parent'] and time.time()<inputs['deadline_epoch'],'Original parent/cutoff no longer live')
