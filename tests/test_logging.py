from __future__ import annotations

import logging as py_logging

from asmr_dub_pipeline.logging import configure_logging


def test_configure_logging_suppresses_httpx_request_info(caplog) -> None:
    httpx_logger = py_logging.getLogger("httpx")
    httpcore_logger = py_logging.getLogger("httpcore")
    original_httpx_level = httpx_logger.level
    original_httpcore_level = httpcore_logger.level
    try:
        httpx_logger.setLevel(py_logging.NOTSET)
        httpcore_logger.setLevel(py_logging.NOTSET)

        configure_logging(py_logging.INFO)

        with caplog.at_level(py_logging.INFO):
            httpx_logger.info('HTTP Request: POST http://127.0.0.1:9880/tts "HTTP/1.1 200 OK"')

        assert not caplog.records
        assert httpx_logger.getEffectiveLevel() == py_logging.WARNING
        assert httpcore_logger.getEffectiveLevel() == py_logging.WARNING
    finally:
        httpx_logger.setLevel(original_httpx_level)
        httpcore_logger.setLevel(original_httpcore_level)
