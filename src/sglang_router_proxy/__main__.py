import argparse
import hashlib
import json
import logging

import httpx
from fastapi import FastAPI, Request, Response
import uvicorn


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache-aware data parallel routing proxy for SGLang. "
        "Routes requests to consistent DP ranks based on a routing key header, "
        "ensuring prefix cache hits across requests with shared context.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 16 DP ranks
  python3 -m sglang_router_proxy --backend http://10.20.2.21:8000 --dp-size 16

  # Custom port and routing header
  python3 -m sglang_router_proxy --backend http://gpu01:8000 --dp-size 8 --port 8080 --routing-header X-My-Key

How it works:
  1. Client sends a request with a routing key header (default: X-SMG-Routing-Key)
  2. Proxy hashes the key and maps it to a DP rank: hash(key) %% dp-size
  3. Proxy injects {"data_parallel_rank": <rank>} into the request JSON body
  4. SGLang server routes the request to that specific DP rank's radix cache

  Requests without the routing header are forwarded unmodified (round-robin).
""",
    )
    parser.add_argument(
        "--backend",
        default="http://127.0.0.1:8000",
        help="SGLang server URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        required=True,
        help="number of data parallel ranks (must match --dp N in SGLang server)",
    )
    parser.add_argument(
        "--routing-header",
        default="X-SMG-Routing-Key",
        help="request header containing the routing key (default: X-SMG-Routing-Key)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="proxy listen address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="proxy listen port (default: 9000)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="upstream request timeout in seconds (default: 600)",
    )
    return parser.parse_args()


args = parse_args()
app = FastAPI()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request):
    body = await request.body()

    if request.method == "POST" and body:
        routing_key = request.headers.get(args.routing_header)
        if routing_key:
            data = json.loads(body)
            rank = int(hashlib.sha256(routing_key.encode()).hexdigest(), 16) % args.dp_size
            data["data_parallel_rank"] = rank
            body = json.dumps(data).encode()
            logger.info(f'Routing key `{routing_key}` sent to DP rank `{rank}`')

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        resp = await client.request(
            method=request.method,
            url=f"{args.backend}/{path}",
            headers={
                k: v
                for k, v in request.headers.items()
                if k.lower() not in ("host", "content-length")
            },
            content=body,
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


if __name__ == "__main__":
    print(f"Routing {args.routing_header} -> {args.dp_size} DP ranks @ {args.backend}")
    uvicorn.run(app, host=args.host, port=args.port)