from sphinx.ext.autodoc import ModuleLevelDocumenter


class ShortSummaryDocumenter(ModuleLevelDocumenter):
    """An autodocumenter that only renders the short summary of the object."""

    # Defines the usage: .. autoshortsummary:: {{ object }}
    objtype = "shortsummary"

    # Disable content indentation
    content_indent = ""

    priority = -99

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        """Never auto-select this documenter; it is only ever invoked explicitly
        via the ``.. autoshortsummary::`` directive.

        Returning ``False`` keeps it out of ``autosummary``'s objtype detection.
        Otherwise, for module-level data/constants (whose ``DataDocumenter`` declines
        the ``isattr=False`` probe used by ``get_documenter``) this would be the only
        qualifying documenter, so the generated stub would render the one-line summary
        instead of the object's full docstring via ``autodata``.
        """
        return False

    def get_object_members(self, want_all):
        """Document no members."""
        return (False, [])

    def add_directive_header(self, sig):
        """Override default behavior to add no directive header or options."""
        pass

    def add_content(self, more_content):
        """Override default behavior to add only the first line of the docstring.

        Modified based on the part of processing docstrings in the original
        implementation of this method.

        https://github.com/sphinx-doc/sphinx/blob/faa33a53a389f6f8bc1f6ae97d6015fa92393c4a/sphinx/ext/autodoc/__init__.py#L609-L622
        """
        sourcename = self.get_sourcename()
        docstrings = self.get_doc()

        if docstrings is not None:
            if not docstrings:
                docstrings.append([])
            # Get the first non-empty line of the processed docstring; this could lead
            # to unexpected results if the object does not have a short summary line.
            short_summary = next(
                (s for s in self.process_doc(docstrings) if s), "<no summary>"
            )
            self.add_line(short_summary, sourcename, 0)


def setup(app):
    app.add_autodocumenter(ShortSummaryDocumenter)
