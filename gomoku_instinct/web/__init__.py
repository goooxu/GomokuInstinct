"""Web 版人机对战（标准库 http.server，无额外依赖）。"""

from .server import App, build_app, serve

__all__ = ["App", "build_app", "serve"]
