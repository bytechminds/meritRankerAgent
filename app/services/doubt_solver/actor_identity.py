"""Central identity seam for validated doubt-solver requests."""

from __future__ import annotations

from schemas.doubt_solver import DoubtSolverRequest


def resolve_actor_id(request: DoubtSolverRequest) -> str:
    """Return the trusted actor inserted by the authenticated SSR proxy.

    ``DoubtSolverRequest`` rejects missing, blank, and malformed identifiers.
    Production callers must reach the IAM-authenticated Runtime through the
    Cognito-verifying Amplify SSR proxy; Python does not verify the JWT again.
    """
    return request.user_id
