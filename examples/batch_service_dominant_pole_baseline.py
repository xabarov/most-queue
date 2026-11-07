"""Withdrawn EPIC-066 surrogate; this was NOT Claeys' dominant-pole method.

The service phase rate is not the infinite-queue waiting-tail exponent.
Even M/M/1 has exponent mu-lambda, not mu. Historical JSON is retained for
provenance only; see docs/research/bulk-sla-revision-2026.md. Claeys (2013)
is openly available, DOI 10.1016/j.cor.2012.10.009, and computes poles and
residues (including competing contributions) rather than this surrogate.
"""


def main():
    """Refuse to regenerate mathematically invalid comparison results."""
    raise NotImplementedError(
        "Withdrawn: the former service-rate surrogate is not a dominant-pole baseline. "
        "Use examples.bulk_sla_revision for the corrected experimental protocol."
    )


if __name__ == "__main__":
    main()
