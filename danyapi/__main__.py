import logging
import os

import uvicorn

logging.basicConfig(level=os.environ.get("DANYAPI_LOG_LEVEL", "INFO"))


def main() -> None:
    from danyapi.config import settings

    uvicorn.run("danyapi.api.openai:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
