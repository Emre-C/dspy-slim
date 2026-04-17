from typing import Any

from pydantic.fields import FieldInfo

import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature


class ChainOfThought(Module):
    def __init__(
        self,
        signature: str | type[Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
        **config: dict[str, Any],
    ):
        """
        A lightweight wrapper around ``dspy.Predict`` that prepends a reasoning field.

        Args:
            signature: The signature of the module.
            rationale_field: Optional custom field definition for the reasoning output.
            rationale_field_type: The type of the reasoning field.
            **config: Configuration forwarded to ``dspy.Predict``.
        """
        super().__init__()
        signature = ensure_signature(signature)
        prefix = "Reasoning: Let's think step by step in order to"
        desc = "${reasoning}"
        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else dspy.OutputField(prefix=prefix, desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        self.predict = dspy.Predict(extended_signature, **config)

    def forward(self, **kwargs):
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        return await self.predict.acall(**kwargs)
