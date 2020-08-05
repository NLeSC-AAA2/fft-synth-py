# fft-synth-py
FFT Synthesis for OpenCL on FPGA

This is a literate code. To work on this you will want to have [Entangled](https://entangled.github.io/) installed. Some of the documentation contains output of Python snippets passed through Jupyter. Make sure you work in a virtual-environment where Jupyter is installed.

The best way to setup the edit workflow:

- have `nodejs` with `browser-sync` (`npm install -g browser-sync`)
- have Entangled filters installed (`pip install --user entangled-filters`)
- have a recent version of Pandoc (&le; 2.7)
- `tmux`
- `inotify-tools`
- GNU `make`

To start `entangled`, `browser-sync` and the `inotify-wait` loop for running Pandoc:

```
make watch
```

