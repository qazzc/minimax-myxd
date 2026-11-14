"""CPU-only regression coverage for the Hygon libnl loader fix."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_hygon_libnl.py"
_SPEC = importlib.util.spec_from_file_location("hygon_libnl_compat", _SCRIPT)
compat = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compat)


def _wheel(root):
    torch = root / "site-packages" / "torch"
    torch.mkdir(parents=True)
    (torch / "__init__.py").touch()
    libs = torch.parent / "torch.libs"
    libs.mkdir()
    pair = {}
    for alias, filename in (
        ("libnl-3.so.200", "libnl-3-aaaa.so.200.26.0"),
        ("libnl-route-3.so.200", "libnl-route-3-bbbb.so.200.26.0"),
    ):
        pair[alias] = libs / filename
        pair[alias].touch()
    return torch, pair


def test_libnl_discovery_does_not_import_torch():
    with TemporaryDirectory() as tmp:
        torch, pair = _wheel(Path(tmp))
        with patch.object(compat.sys, "platform", "linux"), patch.object(
            compat.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(torch / "__init__.py"))
        ):
            assert compat.bundled_libnl() == {name: path.resolve() for name, path in pair.items()}


def test_libnl_discovery_skips_unbundled_wheel():
    with TemporaryDirectory() as tmp:
        with patch.object(compat.sys, "platform", "linux"), patch.object(
            compat.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(Path(tmp) / "torch" / "__init__.py"))
        ):
            assert compat.bundled_libnl() == {}


def test_libnl_discovery_rejects_incomplete_pair():
    with TemporaryDirectory() as tmp:
        torch, pair = _wheel(Path(tmp))
        pair["libnl-3.so.200"].unlink()
        with patch.object(compat.sys, "platform", "linux"), patch.object(
            compat.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(torch / "__init__.py"))
        ):
            try:
                compat.bundled_libnl()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Incomplete vendor pair was accepted")


def test_libnl_discovery_skips_non_linux():
    with patch.object(compat.sys, "platform", "darwin"):
        assert compat.bundled_libnl() == {}


def test_libnl_aliases_use_same_files_without_copying():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _, pair = _wheel(root)
        directory = compat.prepare_aliases(root / "cache", pair)
        assert compat.prepare_aliases(root / "cache", pair) == directory
        for alias, original in pair.items():
            assert (directory / alias).is_symlink()
            assert (directory / alias).samefile(original)
        assert not list(directory.glob("*.tmp"))


def test_libnl_alias_directory_is_specific_to_wheel_location():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _, one = _wheel(root / "one")
        _, two = _wheel(root / "two")
        assert compat.prepare_aliases(root / "cache", one) != compat.prepare_aliases(root / "cache", two)


def test_libnl_aliases_do_not_overwrite_existing_files():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _, pair = _wheel(root)
        directory = compat.prepare_aliases(root / "cache", pair)
        alias = directory / "libnl-3.so.200"
        alias.unlink()
        alias.write_text("not a managed symlink")
        try:
            compat.prepare_aliases(root / "cache", pair)
        except FileExistsError:
            assert alias.read_text() == "not a managed symlink"
        else:
            raise AssertionError("Unexpected file was overwritten")


def test_libnl_no_aliases_needed_for_unbundled_wheel():
    with TemporaryDirectory() as tmp:
        cache = Path(tmp) / "unused-cache"
        assert compat.prepare_aliases(cache, {}) is None
        assert not cache.exists()
