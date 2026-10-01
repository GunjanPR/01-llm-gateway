"""`python -m llm_gateway` starts the gateway with Uvicorn."""

import argparse

import uvicorn

from llm_gateway.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="llm_gateway")
    parser.add_argument("--host", default=settings.gateway_host)
    parser.add_argument("--port", type=int, default=settings.gateway_port)
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    uvicorn.run(
        "llm_gateway.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
