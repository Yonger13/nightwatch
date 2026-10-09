"""Nightwatch 测试包。

这个 `__init__.py` 的存在是为了让 `unittest discover` 能把 tests 目录
当作一个包来导入 —— 否则它会报 "Start directory is not importable"。

（Windows 上尤其容易碰这个错，因为路径含空格时 Python 的包发现机制会更挑剔。）
"""
