import unittest
from unittest.mock import patch

import danyapi.__main__ as main_mod


class TestMain(unittest.TestCase):
    def test_main_runs_uvicorn(self):
        with (
            patch.object(main_mod.uvicorn, "run") as run,
            patch("danyapi.config.settings") as settings,
        ):
            settings.host = "1.2.3.4"
            settings.port = 9999
            main_mod.main()
            run.assert_called_once_with(
                "danyapi.api.openai:app",
                host="1.2.3.4",
                port=9999,
                log_config=None,
            )


if __name__ == "__main__":
    unittest.main()
