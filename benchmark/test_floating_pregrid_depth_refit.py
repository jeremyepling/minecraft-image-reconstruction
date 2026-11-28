"""Source-corner acceptance and benchmark invariants for the pre-grid replay."""
import unittest

from benchmark.floating_pregrid_depth_refit import REVIEW, corner_proof, read


class PregridTest(unittest.TestCase):
    def test_corner_requires_distinct_source_planes_and_supported_return(self):
        selected=["wall~p2","rim","end"]
        joins=[dict(kind="edge",chart_ids=pair,edges=count) for pair,count in
               ((selected[:2],11),(selected[::2],1),(selected[1:],11))]
        fresh={selected[0]}
        self.assertEqual(len(corner_proof(selected,joins,fresh,{})),1)
        self.assertFalse(corner_proof(selected,joins[:-1],fresh,{}))
        self.assertFalse(corner_proof(selected,joins,fresh,{"wall~p2":"rim"}))
        joins[0]["kind"]="coplanar"
        self.assertFalse(corner_proof(selected,joins,fresh,{}))

    def test_archived_whole_export_choices_and_gross_changes(self):
        rows=read(REVIEW/"conditions.json.gz")
        self.assertEqual(len(rows),84)
        total=0
        for row in rows:
            self.assertEqual(row["score"]["noop"],row["score"]["adopted"])
            choice=row["roots"]["nonshrinking"]
            self.assertIn(choice,[row["roots"][a] for a in ("noop","complete","partial")])
            total+=choice!=row["roots"]["noop"]
            if row["panel"]=="sky":
                self.assertEqual(choice,row["roots"]["noop"])
        self.assertEqual(total,3)
        metrics=read(REVIEW/"summary.json")["all"]["nonshrinking"]
        self.assertEqual(metrics["paired"],dict(gained=211,lost=28,added=0,removed=0))
        self.assertEqual(metrics["exact_paired"],dict(gained=200,lost=25))
