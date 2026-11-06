import logging

import pytest

from src.utils import config as config_mod
from src.utils import logger as logger_mod
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger


def test_get_project_root_points_to_repo_root():
    root = get_project_root()
    assert (root / "config" / "config.yaml").exists()
    assert (root / "src").is_dir()


def test_load_config_default():
    config = load_config()
    assert config["project"]["random_seed"] == 42
    assert "paths" in config and "modeling" in config


def test_load_config_custom_relative_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "get_project_root", lambda: tmp_path)
    (tmp_path / "custom.yaml").write_text("hello: world\n", encoding="utf-8")
    config = load_config("custom.yaml")
    assert config == {"hello": "world"}


def test_load_config_custom_absolute_path(tmp_path):
    path = tmp_path / "abs.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    config = load_config(path)
    assert config == {"a": 1}


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_get_logger_returns_configured_logger():
    logger = get_logger("test_logger_unique")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger_unique"
    assert logger.hasHandlers()
    # Calling again returns the same cached logger without duplicating handlers
    handler_count = len(logger.handlers)
    again = get_logger("test_logger_unique")
    assert again is logger
    assert len(again.handlers) == handler_count


def test_get_logger_falls_back_when_config_fails(monkeypatch):
    def boom():
        raise RuntimeError("no config")

    monkeypatch.setattr(logger_mod, "load_config", boom)
    logger = get_logger("test_logger_fallback")
    assert isinstance(logger, logging.Logger)
    assert logger.hasHandlers()
