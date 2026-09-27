import json

import pytest

from ai_diffusion.backend.client import CheckpointInfo, ClientModels
from ai_diffusion.backend.resources import Arch
from ai_diffusion.model_resolution import edit_distance, resolve_model_name
from ai_diffusion.style import Style, Styles
from ai_diffusion.style_resolution import (
    ModelIdentityCache,
    checkpoint_input,
    heal_styles,
)


def make_models(checkpoints: dict[str, str] | None = None, loras=None, vae=None):
    models = ClientModels()
    models.checkpoints = {
        name: CheckpointInfo(name, Arch.sd15, hash=model_hash)
        for name, model_hash in (checkpoints or {}).items()
    }
    models.loras = list((loras or {}).keys())
    models.lora_hashes = dict(loras or {})
    models.vae = list((vae or {}).keys())
    models.vae_hashes = dict(vae or {})
    return models


@pytest.fixture
def style_env(tmp_path, monkeypatch):
    builtin_dir = tmp_path / "builtin"
    user_dir = tmp_path / "user"
    builtin_dir.mkdir()
    user_dir.mkdir()
    styles = Styles(builtin_dir, user_dir)
    monkeypatch.setattr(Styles, "_instance", styles)
    cache = ModelIdentityCache(tmp_path / "identity.json")
    monkeypatch.setattr("ai_diffusion.style_resolution._instance", cache)
    return styles, cache


# --- model name resolution -------------------------------------------------


def test_resolve_exact_match():
    assert resolve_model_name("Foo.safetensors", ["other.safetensors", "foo.safetensors"]) == (
        "foo.safetensors"
    )


def test_resolve_subfolder_match():
    assert resolve_model_name("foo.safetensors", ["bar/foo.safetensors"]) == "bar/foo.safetensors"


def test_resolve_prefers_exact_over_subfolder():
    available = ["foo.safetensors", "bar/foo.safetensors"]
    assert resolve_model_name("foo.safetensors", available) == "foo.safetensors"


def test_resolve_prefers_closest_subfolder():
    available = ["x/y/foo.safetensors", "z/foo.safetensors", "x/fooo.safetensors"]
    assert resolve_model_name("foo.safetensors", available) == "z/foo.safetensors"
    assert resolve_model_name("x/y/foo.safetensors", available) == "x/y/foo.safetensors"


def test_resolve_subfolder_tie_breaks_by_depth():
    available = ["a/foo.safetensors", "b/c/foo.safetensors"]
    assert resolve_model_name("foo.safetensors", available) == "a/foo.safetensors"


def test_resolve_hash_match_beats_basename():
    available = ["bar/foo.safetensors", "bazz.safetensors"]
    hashes = {"bar/foo.safetensors": "h1", "bazz.safetensors": "h2"}
    result = resolve_model_name(
        "foo.safetensors", available, reference_hash="h2", available_hashes=hashes
    )
    assert result == "bazz.safetensors"


def test_resolve_exact_match_beats_conflicting_hash():
    available = ["foo.safetensors", "bazz.safetensors"]
    hashes = {"foo.safetensors": "h1", "bazz.safetensors": "h2"}
    result = resolve_model_name(
        "foo.safetensors", available, reference_hash="h2", available_hashes=hashes
    )
    assert result == "foo.safetensors"


def test_resolve_fuzzy_match():
    assert resolve_model_name("flux1-dev.sft", ["flux1-dev.safetensors"]) == (
        "flux1-dev.safetensors"
    )
    assert resolve_model_name("foo.safetensors", ["entirely-different.safetensors"]) is None


def test_resolve_no_fuzzy_for_loras():
    assert resolve_model_name("foo.safetensors", ["fooo.safetensors"], allow_fuzzy=False) is None


def test_edit_distance():
    assert edit_distance("", "") == 0
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "abd") == 1
    assert edit_distance("kitten", "sitting") == 3


# --- style healing ----------------------------------------------------------


def make_style(styles: Styles, name: str, **kwargs):
    style = Style(styles.user_folder / f"{name}.json")
    style.name = name
    for key, value in kwargs.items():
        setattr(style, key, value)
    style.save()
    styles.reload()
    result = styles.find(f"{name}.json")
    assert result is not None
    return result


def test_heal_styles_subfolder(style_env):
    styles, _ = style_env
    style = make_style(styles, "test", checkpoints=["foo.safetensors"])

    changed = heal_styles(make_models(checkpoints={"bar/foo.safetensors": "h1"}))

    assert changed is True
    assert style.checkpoints == ["bar/foo.safetensors"]


def test_heal_styles_records_hashes(style_env):
    styles, cache = style_env
    make_style(styles, "test", checkpoints=["foo.safetensors"])
    heal_models = make_models(checkpoints={"foo.safetensors": "h1", "other.safetensors": "h2"})

    heal_styles(heal_models)

    assert cache.hash_of("test.json", "checkpoints", "foo.safetensors") == "h1"


def test_heal_styles_survives_rename_via_hash(style_env):
    styles, _ = style_env
    style = make_style(styles, "test", checkpoints=["foo.safetensors"])
    heal_styles(make_models(checkpoints={"foo.safetensors": "h1"}))

    # Model renamed to a completely different name
    changed = heal_styles(make_models(checkpoints={"bazz.safetensors": "h1"}))

    assert changed is True
    assert style.checkpoints == ["bazz.safetensors"]


def test_heal_styles_built_in_style(style_env):
    styles, _ = style_env
    builtin = Style(styles.builtin_folder / "builtin.json")
    builtin.name = "Built-in"
    builtin.checkpoints = ["foo.safetensors"]
    builtin.save()
    styles.reload()

    changed = heal_styles(make_models(checkpoints={"sub/foo.safetensors": "h1"}))

    assert changed is True
    builtin_style = styles.find("builtin.json")
    assert builtin_style is not None
    assert builtin_style.checkpoints == ["sub/foo.safetensors"]
    # built-in style files are not rewritten
    assert "sub/foo.safetensors" not in builtin.filepath.read_text()


def test_heal_styles_loras_and_vae(style_env):
    styles, cache = style_env
    style = make_style(styles, "test", checkpoints=["cp.safetensors"])
    style.loras = [{"name": "my-lora.safetensors", "strength": 0.8}]
    style.vae = "vae-old.safetensors"

    models = make_models(
        checkpoints={"cp.safetensors": "c1"},
        loras={"detailer/my-lora.safetensors": "l1"},
        vae={"vae-new.safetensors": "v1"},
    )
    # Prime the vae hash so the rename heals via identity
    cache.record("test.json", "vae", "vae-old.safetensors", "v1")

    changed = heal_styles(models)

    assert changed is True
    assert style.loras[0]["name"] == "detailer/my-lora.safetensors"
    assert style.vae == "vae-new.safetensors"
    assert cache.hash_of("test.json", "loras", "detailer/my-lora.safetensors") == "l1"


def test_heal_styles_no_match_keeps_reference(style_env):
    styles, _ = style_env
    style = make_style(styles, "test", checkpoints=["foo.safetensors"])

    changed = heal_styles(make_models(checkpoints={"unrelated.safetensors": "h1"}))

    assert changed is False
    assert style.checkpoints == ["foo.safetensors"]


def test_checkpoint_input_resolves_and_records(style_env):
    styles, cache = style_env
    style = make_style(styles, "test", checkpoints=["foo.safetensors"])
    style.loras = [{"name": "my-lora.safetensors", "strength": 0.5}]
    models = make_models(
        checkpoints={"bar/foo.safetensors": "h1"},
        loras={"my-lora.safetensors": "l1"},
    )

    result = checkpoint_input(style, models)

    assert result.checkpoint == "bar/foo.safetensors"
    assert result.loras[0].name == "my-lora.safetensors"
    assert cache.hash_of("test.json", "checkpoints", "bar/foo.safetensors") == "h1"
    assert cache.hash_of("test.json", "loras", "my-lora.safetensors") == "l1"
    # style itself is not modified at use time
    assert style.checkpoints == ["foo.safetensors"]


def test_checkpoint_input_keeps_local_lora(style_env):
    styles, _ = style_env
    style = make_style(styles, "test", checkpoints=["foo.safetensors"])
    style.loras = [{"name": "local-only.safetensors", "strength": 0.5}]

    result = checkpoint_input(style, make_models(checkpoints={"foo.safetensors": "h1"}))

    assert result.loras[0].name == "local-only.safetensors"


# --- identity cache ---------------------------------------------------------


def test_identity_cache_roundtrip(tmp_path):
    path = tmp_path / "identity.json"
    cache = ModelIdentityCache(path)
    cache.record("a.json", "checkpoints", "foo.safetensors", "h1")
    cache.record("a.json", "loras", "l.safetensors", "h2")
    cache.save()

    loaded = ModelIdentityCache(path)
    assert loaded.hash_of("a.json", "checkpoints", "foo.safetensors") == "h1"
    assert loaded.hash_of("a.json", "loras", "l.safetensors") == "h2"
    assert loaded.hash_of("missing.json", "checkpoints", "foo.safetensors") is None

    loaded.prune({"b.json"})
    assert loaded.hash_of("a.json", "checkpoints", "foo.safetensors") is None


def test_identity_cache_overwrites_hash(tmp_path):
    cache = ModelIdentityCache(tmp_path / "identity.json")
    cache.record("a.json", "checkpoints", "foo.safetensors", "h1")
    cache.record("a.json", "checkpoints", "foo.safetensors", "h2")
    assert cache.hash_of("a.json", "checkpoints", "foo.safetensors") == "h2"


def test_identity_cache_ignores_invalid_file(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text(json.dumps({"bad": "entry"}))
    cache = ModelIdentityCache(path)
    assert cache.hash_of("bad", "checkpoints", "x") is None


# --- client models ----------------------------------------------------------


def test_checkpoint_info_hash_roundtrip():
    info = CheckpointInfo("foo.safetensors", Arch.sd15, hash="h1")
    data = info.to_dict()
    assert data["hash"] == "h1"
    assert CheckpointInfo.from_dict(data) == info

    without_hash = CheckpointInfo("foo.safetensors", Arch.sd15)
    assert "hash" not in without_hash.to_dict()
    assert CheckpointInfo.from_dict({"filename": "foo.safetensors", "arch": "sd15"}) == (
        without_hash
    )


def test_client_models_hash_maps_roundtrip():
    models = ClientModels.from_dict({
        "loras": ["a.safetensors"],
        "lora_hashes": {"a.safetensors": "h1"},
        "vae": ["v.safetensors"],
        "vae_hashes": {"v.safetensors": "h2"},
    })
    assert models.lora_hashes == {"a.safetensors": "h1"}
    assert models.vae_hashes == {"v.safetensors": "h2"}
