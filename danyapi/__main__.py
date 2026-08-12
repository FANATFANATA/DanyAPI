import uvicorn


def main() -> None:
    from danyapi.config import settings

    uvicorn.run(
        "danyapi.api.openai:app",
        host=settings.host,
        port=settings.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
