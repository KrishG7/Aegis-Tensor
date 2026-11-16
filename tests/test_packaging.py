"""The optional `engines` extra must install and import on every supported Python (issue #39).

- Fickling 0.1.12 imports `typing_extensions` without declaring it; 0.1.13 declares it but
  needs Python 3.11+. So Python 3.10 gets 0.1.12 plus `typing_extensions`, and 3.11+ gets 0.1.13.
- ModelScan has no release for Python 3.13+, so it is skipped there instead of failing the install.
"""

from importlib.metadata import requires

from packaging.requirements import Requirement


def _engines_on(python: str) -> dict[str, Requirement]:
    env = {"extra": "engines", "python_version": python, "python_full_version": f"{python}.0"}
    found = {}
    for line in requires("aegis-tensor") or []:
        requirement = Requirement(line)
        if requirement.marker is not None and requirement.marker.evaluate(env):
            found.setdefault(requirement.name, []).append(requirement)
    return found


def test_python_310_gets_fickling_with_its_undeclared_dependency():
    engines = _engines_on("3.10")
    assert [r for r in engines["fickling"] if r.specifier.contains("0.1.12")]
    assert "typing-extensions" in engines or "typing_extensions" in engines
    assert "modelscan" in engines


def test_python_311_and_later_get_a_fickling_that_declares_its_dependencies():
    for python in ("3.11", "3.12", "3.13"):
        fickling = _engines_on(python)["fickling"]
        assert all(not r.specifier.contains("0.1.12") and r.specifier.contains("0.1.13") for r in fickling)


def test_modelscan_is_skipped_where_it_has_no_release():
    assert "modelscan" in _engines_on("3.12")
    assert "modelscan" not in _engines_on("3.13")
