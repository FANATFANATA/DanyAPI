import logging
import logging.handlers
import tempfile
import unittest
from pathlib import Path

from danyapi import logging as dlog
from danyapi.config import settings


def _root_handler_names(root):
    return [getattr(h, "name", None) for h in root.handlers]


class TestLoggingHelpers(unittest.TestCase):
    def test_has_handler(self):
        logger = logging.Logger("probe")
        self.assertFalse(dlog._has_handler(logger, "x"))
        handler = logging.StreamHandler()
        handler.name = "x"
        logger.addHandler(handler)
        self.assertTrue(dlog._has_handler(logger, "x"))
        logger.removeHandler(handler)

    def test_make_file_handler_creates_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "app.log"
            handler = dlog._make_file_handler(str(path), 1024, 2)
            self.assertTrue(path.exists())
            self.assertIsInstance(handler, logging.handlers.RotatingFileHandler)
            handler.close()


class TestConfigure(unittest.TestCase):
    def setUp(self):
        self._orig_level = settings.log_level
        self._orig_file = settings.log_file
        self._orig_bytes = settings.log_max_bytes
        self._orig_backup = settings.log_backup_count

    def tearDown(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "name", None) in (dlog.CONSOLE_HANDLER_NAME, dlog.FILE_HANDLER_NAME):
                root.removeHandler(handler)
                handler.close()
        settings.log_level = self._orig_level
        settings.log_file = self._orig_file
        settings.log_max_bytes = self._orig_bytes
        settings.log_backup_count = self._orig_backup

    def test_configure_adds_console_once(self):
        root = logging.getLogger()
        dlog.configure()
        self.assertEqual(_root_handler_names(root).count(dlog.CONSOLE_HANDLER_NAME), 1)
        dlog.configure()
        self.assertEqual(_root_handler_names(root).count(dlog.CONSOLE_HANDLER_NAME), 1)

    def test_configure_sets_root_level(self):
        settings.log_level = "DEBUG"
        dlog.configure()
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_configure_sets_noisy_loggers(self):
        settings.log_level = "INFO"
        dlog.configure()
        self.assertEqual(logging.getLogger("httpx").level, logging.WARNING)
        self.assertEqual(logging.getLogger("httpcore").level, logging.WARNING)

    def test_configure_adds_file_handler(self):
        root = logging.getLogger()
        with tempfile.TemporaryDirectory() as tmp:
            settings.log_file = str(Path(tmp) / "danyapi.log")
            settings.log_max_bytes = 1024
            settings.log_backup_count = 1
            dlog.configure()
            self.assertEqual(_root_handler_names(root).count(dlog.FILE_HANDLER_NAME), 1)
            logging.getLogger("danyapi.test").warning("hello file")
            self.assertTrue(Path(settings.log_file).exists())
            for handler in list(root.handlers):
                if getattr(handler, "name", None) == dlog.FILE_HANDLER_NAME:
                    root.removeHandler(handler)
                    handler.close()


if __name__ == "__main__":
    unittest.main()
