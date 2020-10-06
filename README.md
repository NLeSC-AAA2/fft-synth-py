# fft-synth-py
FFT Synthesis for OpenCL on FPGA

This is a literate code. To work on this you will want to have [Entangled](https://entangled.github.io/) installed. Some of the documentation contains output of Python snippets passed through Jupyter. Make sure you work in a virtual-environment where Jupyter is installed.

The best way to setup the edit workflow:

- have `nodejs` with `browser-sync` (`npm install -g browser-sync`)
- have Entangled filters installed (`pip install --user entangled-filters`)
- have `dhall-to-json` ([Download here](https://github.com/dhall-lang/dhall-haskell/releases))
- `pip install --user pandoc-eqnos pandoc-fignos`
- have a recent version of Pandoc (&le; 2.7)
- `tmux`
- `inotify-tools`
- GNU `make`

To start `entangled`, `browser-sync` and the `inotify-wait` loop for running Pandoc:

```
make watch
```

If that doesn't work, `pip install tmuxp` and run

```
tmuxp load ./tmux-session.yaml
```

## Run unit tests
The unit tests rely on an OpenCL platform being available. Run `clinfo` to see if you have any.

```
pip install -e .[test]
```

And then

```
pytest
```

