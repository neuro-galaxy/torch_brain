# Documentation
## Building the documentation

Install requirements:
```bash
pip install furo sphinx myst-parser sphinx-autodoc-typehints sphinx-inline-tabs sphinx-rtd-theme sphinx-tabs sphinx-design sphinx-copybutton
```

To build the documentation, run the following from current directory:
```bash
make html
```
One can then view the documentation by opening `build/html/index.html` in a web browser.

Alternatively, the source files for the documentation are readable in markdown, and are located in `source/`.