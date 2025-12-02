# tests/frameworks/embedding/test_embedding_registry_self_check.py

from tests.frameworks.registries.embedding_registry import (
    EMBEDDING_FRAMEWORKS,
    iter_embedding_framework_descriptors,
)


def test_embedding_registry_keys_match_descriptor_name() -> None:
    """
    Sanity check: registry keys should always match descriptor.name.

    This keeps lookups and reporting consistent and prevents copy/paste errors
    when adding new frameworks.
    """
    for key, descriptor in EMBEDDING_FRAMEWORKS.items():
        assert key == descriptor.name, f"Registry key '{key}' != descriptor.name '{descriptor.name}'"


def test_embedding_registry_descriptors_validate_cleanly() -> None:
    """
    Run the descriptor-level validation hook to catch obvious inconsistencies
    (e.g. async query defined without async batch).
    """
    for descriptor in iter_embedding_framework_descriptors():
        # validate() may emit warnings but should not raise
        descriptor.validate()
