from typing import TYPE_CHECKING, Any, Awaitable, Callable

from anyio import CapacityLimiter, to_thread

if TYPE_CHECKING:
    from dspy.primitives.module import Module

_limiter: CapacityLimiter | None = None


def get_async_max_workers() -> int:
    import dspy

    return dspy.settings.async_max_workers


def get_limiter() -> CapacityLimiter:
    async_max_workers = get_async_max_workers()

    global _limiter
    if _limiter is None:
        _limiter = CapacityLimiter(async_max_workers)
    elif _limiter.total_tokens != async_max_workers:
        _limiter.total_tokens = async_max_workers

    return _limiter


def asyncify(program: "Module") -> Callable[[Any, Any], Awaitable[Any]]:
    async def async_program(*args, **kwargs) -> Any:
        from dspy.dsp.utils.settings import thread_local_overrides

        parent_overrides = thread_local_overrides.get().copy()

        def wrapped_program() -> Any:
            original_overrides = thread_local_overrides.get()
            token = thread_local_overrides.set({**original_overrides, **parent_overrides.copy()})
            try:
                return program(*args, **kwargs)
            finally:
                thread_local_overrides.reset(token)

        return await to_thread.run_sync(
            wrapped_program,
            limiter=get_limiter(),
            abandon_on_cancel=True,
        )

    return async_program
