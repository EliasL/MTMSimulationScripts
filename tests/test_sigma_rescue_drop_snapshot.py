import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Management.sigmaRescueDropSnapshot import (
    ValidatedSigmaFragment,
    build_interim_drop_table,
    merge_available_sigma,
)


class SigmaRescueDropSnapshotTests(unittest.TestCase):
    def _write_source(self, directory: Path) -> Path:
        source = directory / "macroData.csv"
        source.write_text(
            "load_step,load,total_energy,total_energy_change,total_init_energy,"
            "total_e_change_from_init,avg_sigmaxy,avg_init_sigmaxy,"
            "avg_sigmaxy_change_from_init,avg_Pxy\n"
            "1,0.15,10,0,10,0,0.01,0.01,0,0.02\n"
            "2,0.16,9.8,-0.2,9.9,-0.1,0.02,0.02,-0.01,0.03\n"
            "3,0.17,9.5,-0.3,9.8,-0.3,0.03,0.03,-0.02,0.04\n"
            "4,0.18,9.1,-0.4,9.6,-0.5,0.04,0.04,-0.03,0.05\n"
        )
        (directory / "config.conf").write_text(
            "loadIncrement=0.01\nenergyFunction=contiSquare\nbulkModulus=4\n"
        )
        return source

    @staticmethod
    def _write_fragment(path: Path, rows: list[tuple]) -> None:
        pd.DataFrame(
            rows,
            columns=(
                "load_step",
                "load",
                "avg_sigma12",
                "avg_init_sigma12",
                "avg_sigma12_change_from_init",
            ),
        ).to_csv(path, index=False)

    def test_cross_fragment_transition_is_audited_not_used(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = self._write_source(directory)
            first = directory / "first.csv"
            second = directory / "second.csv"
            self._write_fragment(
                first,
                [(1, 0.15, 0.11, 0.10, 0.01), (2, 0.16, 0.12, 0.11, 0.01)],
            )
            self._write_fragment(
                second,
                [(3, 0.17, 0.13, 0.12, 0.01), (4, 0.18, 0.14, 0.13, 0.01)],
            )
            merged = merge_available_sigma(
                source,
                [
                    ValidatedSigmaFragment("task-a", first),
                    ValidatedSigmaFragment("task-b", second),
                ],
                size=2,
            )
            table = build_interim_drop_table(
                merged,
                source_macro=source,
                run_name="run",
                size=2,
                seed=0,
            )

        self.assertEqual(table["usable"].tolist(), [True, False, True])
        self.assertEqual(table.loc[1, "exclusion_reason"], "cross-provider-boundary")
        self.assertEqual(table["event_id"].tolist(), ["run:2", "run:3", "run:4"])

    def test_conflicting_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = self._write_source(directory)
            first = directory / "first.csv"
            second = directory / "second.csv"
            self._write_fragment(first, [(1, 0.15, 0.11, 0.10, 0.01)])
            self._write_fragment(second, [(1, 0.15, 0.22, 0.10, 0.01)])

            with self.assertRaisesRegex(ValueError, "Conflicting rescue overlap"):
                merge_available_sigma(
                    source,
                    [
                        ValidatedSigmaFragment("task-a", first),
                        ValidatedSigmaFragment("task-b", second),
                    ],
                    size=2,
                )


if __name__ == "__main__":
    unittest.main()
