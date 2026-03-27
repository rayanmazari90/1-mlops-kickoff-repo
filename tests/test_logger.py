import logging

from src.logger import get_logger


def test_get_logger_returns_logger():
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"
    assert logger.level == logging.INFO


def test_get_logger_has_dual_handlers():
    logger = get_logger("test_dual_handlers")
    handler_types = {type(h).__name__ for h in logger.handlers}
    assert "StreamHandler" in handler_types
    assert "FileHandler" in handler_types


def test_get_logger_idempotent():
    l1 = get_logger("test_idempotent")
    count = len(l1.handlers)
    l2 = get_logger("test_idempotent")
    assert len(l2.handlers) == count
