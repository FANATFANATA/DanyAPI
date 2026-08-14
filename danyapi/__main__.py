import uvicorn


def main() -> None:
    from danyapi.config import settings
    from danyapi.logging import uvicorn_log_config

    uvicorn.run(
        "danyapi.api.openai:app",
        host=settings.host,
        port=settings.port,
        log_config=uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
