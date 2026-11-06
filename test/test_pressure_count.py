"""Check the MC volume Jacobian count without CUDA: python test_pressure_count.py."""

from pathlib import Path
import re
from types import SimpleNamespace
import unittest


class PressureCountTest(unittest.TestCase):
    def test_volume_scaling_objects(self):
        source = Path(__file__).resolve().parents[1] / "src/update/pressure.cu"
        body = source.read_text().split("void pressure_coupling(System *system)", 1)[1]
        expression = " ".join(re.search(r"\bN\s*=([^;]+);", body).group(1).split())
        expression = expression.replace("->", ".")
        cases = [
            ("TIP3", 3, [(0, 1, 2)], [], [], []),
            ("OPC", 4, [(0, 1, 2)], [], [], [3]),
            ("TIP5P", 5, [(0, 1, 2)], [], [], [3, 4]),
            ("two-host virtual site", 3, [], [], [2], []),
            ("CH", 2, [], [(0, 1)], [], []),
            ("CH2", 3, [], [(0, 1, 2)], [], []),
            ("CH3", 4, [], [(0, 1, 2, 3)], [], []),
            ("mixed waters, solute, ion", 17,
             [(0, 1, 2), (4, 5, 6)],
             [(7, 8), (9, 10, 11), (12, 13, 14, 15)], [], [3]),
        ]
        for name, atoms, triangles, branches, virtual2, virtual3 in cases:
            with self.subTest(name=name):
                # Count connected components of constrained physical atoms independently.
                physical = set(range(atoms)) - set(virtual2) - set(virtual3)
                components = [{i} for i in physical]
                for group in triangles + branches:
                    touched = [c for c in components if c.intersection(group)]
                    components = [c for c in components if not c.intersection(group)]
                    components.append(set().union(*touched))
                p = SimpleNamespace(
                    triangleConsCount=len(triangles),
                    branch1ConsCount=sum(len(g) == 2 for g in branches),
                    branch2ConsCount=sum(len(g) == 3 for g in branches),
                    branch3ConsCount=sum(len(g) == 4 for g in branches),
                    virtualSite2Count=len(virtual2),
                    virtualSite3Count=len(virtual3),
                )
                actual = eval(expression, {"__builtins__": {}},
                              {"s": SimpleNamespace(atomCount=atoms), "p": p})
                self.assertEqual(actual, len(components))


if __name__ == "__main__":
    unittest.main()
