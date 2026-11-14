#!/usr/bin/env python3
"""Start pinned vLLM with separate engine and frontend monotonic traces."""

import argparse
import sys
from pathlib import Path


def main():
    """Install observational hooks and invoke the pinned vLLM CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", required=True, type=Path)
    parser.add_argument("serve_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    serve_args = args.serve_args
    if serve_args[:1] == ["--"]:
        serve_args = serve_args[1:]
    if not serve_args:
        parser.error("Provide vllm serve arguments after --")
    if any(value == "--middleware" or value.startswith("--middleware=") for value in serve_args):
        parser.error("Frontend wrapper controls --middleware")
    # The CLI resolves middleware by basename even when this bootstrap is imported
    # through tools. Both import paths must share the configured module state.
    # pylint: disable=import-outside-toplevel,import-error
    if __package__:
        from tools import runpod_frontend_trace as frontend
        from tools.runpod_vllm_server import install_trace
    else:
        import runpod_frontend_trace as frontend
        from runpod_vllm_server import install_trace
    sys.modules["runpod_frontend_trace"] = frontend
    sys.modules["tools.runpod_frontend_trace"] = frontend
    import vllm
    from vllm.entrypoints.cli.main import main as vllm_main
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from vllm.v1.engine.output_processor import RequestState

    if vllm.__version__ != "0.32.0":
        raise RuntimeError(f"Expected vLLM 0.32.0, got {vllm.__version__}")
    frontend.configure_trace(args.trace_path.with_name("frontend_timestamps.jsonl"))
    frontend.install_frontend_hooks(OpenAIServingCompletion)
    install_trace(RequestState, args.trace_path)
    sys.argv = ["vllm", "serve", *serve_args, "--middleware", "runpod_frontend_trace.FrontendTraceMiddleware"]
    vllm_main()


if __name__ == "__main__":
    main()
