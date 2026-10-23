## List a community brainsets plugin

**Plugin namespace:** <!-- e.g. my_brainsets -->
**Repository:** <!-- https://github.com/... -->

This PR adds (or updates) one table in `torch_brain/pipeline/plugins.toml`.
See [Writing a brainsets plugin](https://torch-brain.readthedocs.io/en/latest/guides/pipeline/plugins.html#getting-listed).

### Checklist

- [ ] The `install` spec is pinned to a tag, commit or PyPI version.
- [ ] `brainsets check <namespace>` passes with the plugin installed from that spec.
- [ ] The listed `brainsets` are exactly the ones the plugin ships.
- [ ] The plugin has an OSI-approved licence and a README explaining how to use it.
- [ ] The data is publicly available, and each brainset links to its original source.
- [ ] Descriptions are accurate and the original data/paper is cited in the README.
- [ ] I maintain this plugin and understand it is listed as a community plugin,
      maintained by its authors and not by torch_brain.
