import numpy as np
import pytest


def test_vector_features_match_original():
    from trimodal_joint.emg_features import extract_features
    from trimodal_joint.features import batch_features

    x = np.random.default_rng(5).normal(size=(4, 100, 6)).astype("float32")
    expected = np.stack([extract_features(w, 1000) for w in x])
    np.testing.assert_allclose(batch_features(x), expected, rtol=2e-5, atol=2e-4)
    assert np.isfinite(batch_features(np.zeros_like(x))).all()


def test_window_boundaries():
    from trimodal_joint.raw import segment_windows

    segments = np.broadcast_to(np.arange(5)[:, None, None], (5, 10000, 6)).copy()
    w = segment_windows(segments)
    assert w.shape == (5, 331, 100, 6)
    for i in range(5):
        assert (w[i] == i).all()


def test_config_rejects_unknown_and_invalid():
    from trimodal_joint.config import Config

    with pytest.raises(ValueError):
        Config(epochs=0).validate()
    assert Config().epochs == 100


def test_renamed_signal_stays_in_same_patient(tmp_path):
    from trimodal_joint.raw import resolve_signal

    raw = tmp_path / "原始数据"
    raw.mkdir()
    target = raw / "renamed.csv"
    target.write_text("time,data\n0,1\n")
    assert resolve_signal(tmp_path, False) == target
    (raw / "second.csv").write_text("")
    with pytest.raises(ValueError):
        resolve_signal(tmp_path, False)


def test_epoch_budget_is_bounded():
    from trimodal_joint.config import Config

    with pytest.raises(ValueError):
        Config(epochs=101).validate()


def test_source_recipe_changes_cache_signature(tmp_path):
    from trimodal_joint.prepare import preprocessing_signature

    source = tmp_path / "raw.py"
    source.write_text("version1")
    first = preprocessing_signature({"file": "hash"}, 1, [source])
    source.write_text("version2")
    assert preprocessing_signature({"file": "hash"}, 1, [source]) != first


def test_unicode_json_is_locale_independent(tmp_path):
    import locale

    from trimodal_joint.io import save_json

    previous = locale.setlocale(locale.LC_CTYPE)
    try:
        locale.setlocale(locale.LC_CTYPE, "C")
        save_json(tmp_path / "test.json", {"label": "肌少症"})
        assert "肌少症" in (tmp_path / "test.json").read_text(encoding="utf-8")
    finally:
        locale.setlocale(locale.LC_CTYPE, previous)
