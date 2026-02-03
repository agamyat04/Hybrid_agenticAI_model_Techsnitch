from dataclasses import dataclass
from typing import Literal, Dict
@dataclass(frozen=True)
class ModelRole:
    name: str #  (for logs, docs, audits)
    model_id: str # Hugging Face model ID or local path
    role: Literal["base", "fallback"]  
    responsibility: str # model responsiblity at runtime
    behavior_focus: str # What kind of behavior this model is allowed to learn

BASE_MODEL = ModelRole(
    name="Ministral 3 3B Base 2512",
    model_id="mistralai/Ministral-3B-Base-2512",
    role="base",
    responsibility=(
        "Primary reasoning model responsible for agent logic, "
        "step-by-step explanations, and most enterprise interactions."
    ),
    behavior_focus=(
        "Reasoning-heavy responses, structured explanations, calm and neutral tone, "
        "agent-style helpfulness without strict policy enforcement."
    ),
)

FALLBACK_MODEL = ModelRole(
    name="Gemma 3 4B IT",
    model_id="google/gemma-3-4b-it",
    role="fallback",
    responsibility=(
        "Safety-critical fallback model responsible for compliance enforcement, "
        "risk mitigation, and confident refusal of unsafe or ambiguous requests."
    ),
    behavior_focus=(
        "Conservative and professional tone, policy-compliant responses, "
        "high-confidence refusal, minimal verbosity, and no speculative guidance."
    ),
)
# Central registry (single source of truth)

MODEL_REGISTRY: Dict[str, ModelRole] = {
    "base": BASE_MODEL,
    "fallback": FALLBACK_MODEL,
}
def validate_model_registry() -> None:
    """
    Ensures the registry is correctly defined.
    Fails fast if misconfigured.
    """

    roles = {model.role for model in MODEL_REGISTRY.values()}
    if roles != {"base", "fallback"}:
        raise RuntimeError(
            "MODEL_REGISTRY must contain exactly one base and one fallback model."
        )
validate_model_registry()
