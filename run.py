#!/usr/bin/env python3
"""
Simple script to start the LLM Metrics Proxy server.

Usage:
    python run.py                    # Uses config.yaml settings
    python run.py --port 3000        # Override port
    python run.py --host 127.0.0.1   # Override host
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmMetricsRAG.config_loader import get_config
from llmMetricsRAG.proxy.server import start_server


def main():
    parser = argparse.ArgumentParser(
        description="Start the LLM Metrics Proxy Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                    # Use config.yaml settings
    python run.py --port 3000        # Run on port 3000
    python run.py --host 127.0.0.1   # Bind to localhost only
    
Configuration:
    Edit config.yaml to change default settings.
        """
    )
    
    config = get_config()
    
    parser.add_argument(
        "--host",
        type=str,
        default=config.proxy.host,
        help=f"Server host (default: {config.proxy.host})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.proxy.port,
        help=f"Server port (default: {config.proxy.port})"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    if args.reload:
        import uvicorn
        print(f"Starting LLM Metrics Proxy on {args.host}:{args.port} (with auto-reload)")
        print(f"Target API: {config.proxy.target_url}")
        uvicorn.run(
            "llmMetricsRAG.proxy.server:app",
            host=args.host,
            port=args.port,
            reload=True
        )
    else:
        start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
