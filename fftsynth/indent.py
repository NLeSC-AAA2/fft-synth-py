# ~\~ language=Python filename=fftsynth/indent.py
# ~\~ begin <<lit/code-generator.md|fftsynth/indent.py>>[0]
from contextlib import (contextmanager, redirect_stdout)
import io
import textwrap


@contextmanager
def indent(prefix: str):
    f = io.StringIO()
    with redirect_stdout(f):
        yield
    output = f.getvalue()
    print(textwrap.indent(output, prefix), end="")
# ~\~ end
