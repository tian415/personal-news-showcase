import json
from pathlib import Path

import pytest

from news_assistant.models import Preferences
from news_assistant.source_registry import load_source_registry
from news_assistant.workspace import initialize_workspace


def test_initialize_workspace_creates_runnable_isolated_configuration(tmp_path: Path) -> None:
    target = tmp_path / "reader"

    result = initialize_workspace(target)

    registry_path = target / "config/source_registry.yaml"
    preferences_path = target / "config/preferences.json"
    registry = load_source_registry(registry_path)
    preferences = Preferences.model_validate(json.loads(preferences_path.read_text()))

    assert result == {
        "workspace": str(target),
        "created_files": [
            ".gitignore",
            "README.md",
            "config/preferences.json",
            "config/source_registry.yaml",
        ],
        "created_directories": ["config", "work/private-state"],
        "source_registry": "config/source_registry.yaml",
        "preferences": "config/preferences.json",
        "private_state": "work/private-state",
    }
    assert registry.version == "source-registry-v2"
    assert registry.enabled_feed_sources()
    assert all(weight == 1.0 for weight in preferences.topic_weights.values())
    assert preferences.source_weights == {}
    assert (target / "work/private-state").is_dir()
    assert not (target / "public").exists()
    assert "DEEPSEEK_API_KEY" in (target / "README.md").read_text()
    assert ".env" in (target / ".gitignore").read_text().splitlines()


def test_initialize_workspace_accepts_an_existing_empty_directory(tmp_path: Path) -> None:
    target = tmp_path / "reader"
    target.mkdir()

    initialize_workspace(target)

    assert (target / "config/source_registry.yaml").is_file()


def test_initialize_workspace_restores_existing_empty_directory_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from news_assistant import workspace

    target = tmp_path / "reader"
    target.mkdir()

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(workspace.os, "replace", fail_replace)

    with pytest.raises(OSError, match="cannot replace"):
        initialize_workspace(target)

    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_initialize_workspace_refuses_repeat_without_changing_existing_files(tmp_path: Path) -> None:
    target = tmp_path / "reader"
    initialize_workspace(target)
    preferences_path = target / "config/preferences.json"
    preferences_path.write_text('{"owner": "keep-me"}\n')

    with pytest.raises(FileExistsError, match="not empty"):
        initialize_workspace(target)

    assert preferences_path.read_text() == '{"owner": "keep-me"}\n'


def test_initialize_workspace_refuses_nonempty_directory_without_changing_data(
    tmp_path: Path,
) -> None:
    target = tmp_path / "reader"
    target.mkdir()
    existing = target / "notes.txt"
    existing.write_text("keep me")

    with pytest.raises(FileExistsError, match="not empty"):
        initialize_workspace(target)

    assert existing.read_text() == "keep me"
    assert sorted(path.name for path in target.iterdir()) == ["notes.txt"]


def test_initialize_workspace_refuses_target_symlink_without_touching_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "destination"
    destination.mkdir()
    target = tmp_path / "reader"
    target.symlink_to(destination, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        initialize_workspace(target)

    assert list(destination.iterdir()) == []


def test_initialize_workspace_refuses_symbolic_link_in_target_path(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    destination.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(destination, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        initialize_workspace(linked_parent / "reader")

    assert list(destination.iterdir()) == []


def test_invalid_packaged_template_does_not_create_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from news_assistant import workspace

    def invalid_resource(name: str) -> str:
        if name == "source_registry.yaml":
            return "version: source-registry-v2\nsources: ["
        return "{}"

    monkeypatch.setattr(workspace, "_resource_text", invalid_resource)
    target = tmp_path / "reader"

    with pytest.raises(ValueError, match="source registry"):
        initialize_workspace(target)

    assert not target.exists()


@pytest.mark.parametrize("resource_name", ["source_registry.yaml", "preferences.json"])
def test_missing_installed_resource_fails_without_guessing_parent_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resource_name: str,
) -> None:
    from news_assistant import workspace

    installed_package = tmp_path / "lib/python3.12/site-packages/news_assistant"
    installed_package.mkdir(parents=True)
    guessed_config = tmp_path / "lib/python3.12/config/source_registry.yaml"
    guessed_config.parent.mkdir()
    guessed_config.write_text("must not be read")
    monkeypatch.setattr(workspace, "files", lambda _package: installed_package)
    monkeypatch.setattr(workspace, "__file__", str(installed_package / "workspace.py"))

    with pytest.raises(FileNotFoundError, match="packaged resource is missing"):
        workspace._resource_text(resource_name)
