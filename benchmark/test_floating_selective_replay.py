"""Checkpoint invariants without Minecraft, ONNX or a downloaded release."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmark.floating_stage_cache import code_versions, definitions, execute, frame_lock


class StageReuse(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.stages=("source","fit","score")
        self.settings={s:"v1" for s in self.stages}
        self.fail=None

    def inputs(self,s,records):
        return {"setting":self.settings[s],"parent":next(reversed(records.values()))["fingerprint"] if records else None}

    def producer(self,stage,path,upstream):
        (path/"result").write_text(stage+":"+self.settings[stage])
        if stage==self.fail:
            raise ValueError("interrupted")

    def actions(self,**kw):
        return [r["action"] for r in execute(self.root,self.stages,self.inputs,self.producer,**kw)]

    def test_warm_run_and_stage_changes(self):
        self.assertEqual(self.actions(),["build"]*3)
        self.assertEqual(self.actions(),["reuse"]*3)
        self.settings["score"]="v2"
        self.assertEqual(self.actions(),["reuse","reuse","build"])
        self.settings["fit"]="v2"
        self.assertEqual(self.actions(),["reuse","build","build"])
        self.assertEqual(self.actions(rerun_from="source"),["build"]*3)

    def test_tamper_and_failed_producer(self):
        self.actions()
        manifest=json.loads((self.root/"current.json").read_bytes())
        (self.root/manifest["fit"]["generation"]/"result").write_text("bad content")
        self.fail="score"
        with self.assertRaisesRegex(ValueError,"interrupted"):
            self.actions()
        self.assertEqual(set(json.loads((self.root/"checkpoints.json").read_bytes())),{"source","fit"})
        self.fail=None
        self.assertEqual(self.actions(),["reuse","reuse","build"])

    def test_short_prefix_retains_valid_later_checkpoint(self):
        self.actions()
        execute(self.root,self.stages[:2],self.inputs,self.producer)
        self.assertEqual(self.actions(),["reuse"]*3)
        self.settings["fit"]="changed"
        execute(self.root,self.stages[:2],self.inputs,self.producer)
        self.assertEqual(self.actions(),["reuse","reuse","build"])

    def test_code_hashes_module_constants_and_called_function(self):
        a=definitions("demo",b"knob=1\ndef score(x): return x\n")
        b=definitions("demo",b"knob=1\ndef score(x): return x+1\n")
        c=definitions("demo",b"knob=2\ndef score(x): return x+1\n")
        self.assertEqual(a["<module>"],b["<module>"])
        self.assertNotEqual(a["score"],b["score"])
        self.assertNotEqual(b["<module>"],c["<module>"])
        self.assertIsNone(code_versions({"benchmark/no-such-source.py":["run"]}))

    def test_concurrent_frame_writer_cannot_publish(self):
        with frame_lock(self.root):
            with self.assertRaises(OSError):
                self.actions()
        self.assertEqual(self.actions(),["build"]*3)
