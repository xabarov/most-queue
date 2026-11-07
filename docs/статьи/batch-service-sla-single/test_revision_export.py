"""Check the article's derived minima without repeating the experiments."""

import importlib.util
import itertools
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("article_build_revision", ROOT / "build_revision.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def test_export_contains_every_minimum(monkeypatch):
    rows = BUILD.load("decision")
    rendered_tables = {}
    monkeypatch.setattr(BUILD, "save", lambda fig, name: BUILD.plt.close(fig))
    monkeypatch.setattr(BUILD, "table", lambda name, columns, headers, data: rendered_tables.update({name: data}))
    counts, minima = BUILD.decisions(rows)
    assert len(minima) == 288
    actual = {(m["metric"], m["family"], m["gamma"], m["a"], m["deadline"], m["target"]): m["minima"] for m in minima}
    assert len(actual) == 288
    keys = ("tails", "ordinary_gamma_tails", "atom_gamma_tails")
    expected = {}
    # Independent enumeration, using min rather than relying on JSON row order.
    for metric, targets in (("conditional", (0.01, 0.001, 0.0001)), ("all_arrivals", (0.1, 0.05, 0.01))):
        for family, gamma, threshold, (index, deadline), target in itertools.product(
            ("E3", "H2"), (0.0, 0.1), range(1, 5), enumerate((5, 10, 20)), targets
        ):
            accepted = {key: [] for key in keys}
            for row in rows:
                case = row["case"]
                if (case["family"], case["gamma"], case["a"]) != (family, gamma, threshold):
                    continue
                if "selected" not in row:
                    continue
                selected = row["selected"]
                for key in keys:
                    failure = selected[key][index]
                    if metric == "all_arrivals":
                        failure = 1 - selected["served"] * (1 - failure)
                    if failure <= target:
                        accepted[key].append(case["b"])
            expected[(metric, family, gamma, threshold, deadline, target)] = {
                key: min(values, default=None) for key, values in accepted.items()
            }
    assert actual == expected
    # The last row of the printed table must not overwrite the full export.
    assert len(rendered_tables["caps"]) == 9
    assert actual[("conditional", "E3", 0.0, 1, 10, 0.001)] == dict(zip(keys, (5, 5, 4)))
    assert any(value is None for entry in actual.values() for value in entry.values())
    assert counts["conditional_ordinary_gamma_tails"]["false_accept"] == 20
    assert counts["conditional_atom_gamma_tails"]["false_accept"] == 34
    saved = json.loads((ROOT / "revision_results.json").read_text())
    assert saved["configuration_minima"] == minima
    assert saved["decision_counts"] == counts


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("[0.1234; 0.4567]", "[0,1234; 0,4567]"),
        ("1.55e-13", r"$1{,}55\cdot10^{-13}$"),
        ("0.0e+00", "$0$"),
        (r"$E_3$", r"$E_3$"),
        ("228/228", "228/228"),
    ],
)
def test_russian_number_display(source, expected):
    assert BUILD.russian_cell(source) == expected
