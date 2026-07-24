"""Central identity seam for validated doubt-solver requests."""

from __future__ import annotations

from schemas.doubt_solver import DoubtSolverRequest


def resolve_actor_id(request: DoubtSolverRequest) -> str:
    """Return the current compatibility actor from a validated request.

    Payload identity is not production-trusted. A future validated runtime
    principal can replace this implementation without changing downstream code.
    """
    return request.user_id
