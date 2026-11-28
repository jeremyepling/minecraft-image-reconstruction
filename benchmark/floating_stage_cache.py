"""Verified per-frame generations for Floating. An incomplete generation cannot become current."""
import ast
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def publish(path,value):
    tmp=path.with_name(path.name+"."+uuid4().hex+".tmp")
    tmp.write_text(json.dumps(value,sort_keys=True,separators=(",",":")),encoding="utf-8")
    tmp.replace(path)


def inventory(directory):
    return {p.relative_to(directory).as_posix():digest(p) for p in sorted(directory.rglob("*")) if p.is_file()}


@lru_cache(maxsize=512)
def definitions(name,source):
    tree=ast.parse(source,filename=name)
    entries={}
    config=[]
    for item in tree.body:
        if isinstance(item,ast.ClassDef):
            header=ast.ClassDef(name=item.name,bases=item.bases,keywords=item.keywords,decorator_list=item.decorator_list,
                                body=[v for v in item.body if not isinstance(v,(ast.FunctionDef,ast.AsyncFunctionDef))],
                                type_params=getattr(item,"type_params",[]))
            config.append(ast.dump(header,include_attributes=False))
        if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            entries[item.name]=identity(ast.dump(item,include_attributes=False))
        else:
            config.append(ast.dump(item,include_attributes=False))
    entries["<module>"]=identity(config)
    return entries


def code_versions(used):
    answer={}
    for name,symbols in sorted(used.items()):
        path=ROOT/name
        if not path.exists():
            return None
        current=definitions(name,path.read_bytes())
        answer[name]={s:current.get(s) for s in sorted(set(symbols)|{"<module>"})}
    return answer


@contextmanager
def trace_code():
    """Record actually executed Python producer functions; capture module-level settings separately."""
    seen=set()
    monitor=sys.monitoring
    tool=next(i for i in range(6) if monitor.get_tool(i) is None)
    monitor.use_tool_id(tool,"floating-stage-cache")
    def entered(code,offset):
        seen.add((code.co_filename,code.co_qualname.split(".")[0]))
        return monitor.DISABLE
    monitor.register_callback(tool,monitor.events.PY_START,entered)
    monitor.set_events(tool,monitor.events.PY_START)
    used={}
    try:
        yield used
    finally:
        monitor.set_events(tool,0)
        monitor.register_callback(tool,monitor.events.PY_START,None)
        monitor.free_tool_id(tool)
        monitor.restart_events()
        for name,symbol in seen:
            if not name.endswith(".py"):
                continue
            try:
                rel=Path(name).resolve().relative_to(ROOT).as_posix()
            except ValueError:
                continue
            if rel.split("/")[0] in ("benchmark","calibration","engine","runtime","training"):
                used.setdefault(rel,set()).add(symbol)


@contextmanager
def frame_lock(directory):
    """The OS releases the lock even if the process dies. Different frames use different files."""
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/"writer.lock").open("a+b") as stream:
        if os.name=="nt":
            import msvcrt
            stream.seek(0,2)
            if not stream.tell():
                stream.write(b"0");stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def execute(directory,stages,inputs,produce,rerun_from=None):
    directory=Path(directory)
    with frame_lock(directory):
        current=directory/"current.json"
        checkpoint=directory/"checkpoints.json"
        previous=json.loads(checkpoint.read_bytes()) if checkpoint.is_file() else {}
        generations={}
        profile=[]
        force=False
        for stage in stages:
            force=force or stage==rerun_from
            start=time.perf_counter()
            source=inputs(stage,generations)
            key=identity(source)
            prior=previous.get(stage,{})
            location=directory/prior.get("generation","missing")
            valid=(not force and prior.get("input")==key and location.is_dir() and
                   code_versions(prior["symbols"])==prior["code"] and inventory(location)==prior["outputs"])
            if valid:
                record=prior
                action="reuse"
            else:
                publish(checkpoint,generations)
                location=directory/("g"+uuid4().hex[:12])
                location.mkdir()
                try:
                    with trace_code() as used:
                        produce(stage,location,{k:directory/v["generation"] for k,v in generations.items()})
                    outputs=inventory(location)
                    if not outputs:
                        raise ValueError(f"{stage} produced no files")
                    record=dict(input=key,inputs=source,symbols={k:sorted(v) for k,v in used.items()},
                                code=code_versions(used),generation=location.name,outputs=outputs)
                    record["fingerprint"]=identity([key,record["code"],outputs])
                    # Even behavior-preserving forced rebuilds rerun every later consumer.
                    force=True
                except BaseException:
                    shutil.rmtree(location)
                    raise
                action="build"
            generations[stage]=record
            if action=="build":
                publish(checkpoint,generations)
            profile.append(dict(stage=stage,action=action,seconds=round(time.perf_counter()-start,4),
                                fingerprint=record["fingerprint"],generation=record["generation"]))
        if not force:
            generations={**previous,**generations}
        publish(current,generations)
        publish(directory/"last-profile.json",profile)
        return profile
