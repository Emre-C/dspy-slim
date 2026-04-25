from typing import Any

from dspy.primitives import Example, Module


class Teleprompter:
    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | list[Module] | None = None,
        valset: list[Example] | None = None,
        **kwargs,
    ) -> Module:
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        return self.__dict__
