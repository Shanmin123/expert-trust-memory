"""Conditioned, structured memory module: Source-1 self-discovered path.

Built now: four-table schema (evidence table present but validation inert), deterministic
point-in-time regime inference, a pluggable ReflectionBackend (deterministic default + inert
LLM seam), structured-only retrieval, decision-influence application, and a learning-curve
output. NOT built yet (registered as extensions): audit-counterfactual validation, decay,
provenance Sources 2/3/4, real LLM backbone, LLM regime inference, vector retrieval.

Hard invariant enforced here: the decision and (future) validation paths read ONLY
content_structured. content_annotation.rationale is human-only and never parsed by code.
"""

__all__ = []
