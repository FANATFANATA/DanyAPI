from unittest.mock import patch

import danyapi.__main__ as main_mod


def test_main_runs_uvicorn():
    with (
        patch.object(main_mod.uvicorn, "run") as run,
        patch("danyapi.config.settings") as settings,
        patch("danyapi.logging.uvicorn_log_config", return_value={"version": 1}) as log_cfg,
    ):
        settings.host = "1.2.3.4"
        settings.port = 9999
        main_mod.main()
        log_cfg.assert_called_once_with()
        run.assert_called_once_with(
            "danyapi.api.openai:app",
            host="1.2.3.4",
            port=9999,
            log_config={"version": 1},
        )
