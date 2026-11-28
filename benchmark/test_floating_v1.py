"""Regression checks for the original V1 reconstruction benchmark."""
import hashlib
import unittest
from zipfile import ZipFile
from benchmark.floating_v1_report import ARMS, REVIEW, read


class FrozenV1(unittest.TestCase):
    def test_admission_and_published_exports(self):
        cases=read(REVIEW/"fixtures.json.gz")["cases"]
        rows=read(REVIEW/"conditions.json.gz")
        self.assertEqual(len(cases),144)
        self.assertEqual({r["id"] for r in rows},{c["id"] for c in cases})
        self.assertTrue(all("orchid" not in c["id"].lower() and c["framing"]["accepted"] for c in cases))
        seen=[r for r in rows if r["seen_panel"]]
        self.assertEqual(len(seen),30)
        self.assertTrue(all(r["same_image"][a]["registered_equal"] for r in seen for a in ARMS))
        with ZipFile(REVIEW/"exports.zip") as archive:
            self.assertEqual(len(archive.namelist()),432)
            for r in rows:
                for a in ARMS:
                    packed=archive.read(r["id"]+"/"+a+".zip")
                    self.assertEqual(hashlib.sha256(packed).hexdigest(),r["arms"][a]["pack_sha256"])

    def test_orientation_policy_and_noop(self):
        primary=read(REVIEW/"strict-orientation.json")["all"]
        diagnostic=read(REVIEW/"summary.json")["all"]
        self.assertEqual(primary["sparse"],diagnostic["sparse"])
        self.assertEqual(primary["current"]["nonempty_unresolved"],7)
        self.assertEqual(primary["current"]["empty"],0)
        score=primary["current"]["geometry_strict"]
        self.assertEqual([score[k] for k in ("correct","missed","extras")],[10156,24298,6974])
        self.assertEqual(primary["current"]["scores_strict"]["correct"],4399)
        receipts=read(REVIEW/"no-op.json.gz")
        self.assertEqual(len(receipts),144)
        self.assertTrue(all(r["registered_equal"] and r["decoded_equal"] for r in receipts))
