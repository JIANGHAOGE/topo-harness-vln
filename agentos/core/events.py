from __future__ import annotations

import inspect
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from agentos.schemas.events import Event
from agentos.schemas.enums import EventType, RuntimeState
from agentos.schemas.trace import TraceContext

EventSubscriber = Callable[[Event], Optional[Awaitable[None]]]


class InMemoryEventBus:
    def __init__(self) -> None:
        self._events: list[Event] = []
        self._subscribers: list[EventSubscriber] = []

    def subscribe(self, subscriber: EventSubscriber) -> None:
        self._subscribers.append(subscriber)

    async def publish(self, event: Event) -> None:
        self._events.append(event)
        for subscriber in list(self._subscribers):
            result = subscriber(event)
            if inspect.isawaitable(result):
                await result

    async def emit(
        self,
        *,
        event_type: EventType,
        trace: TraceContext,
        state: RuntimeState,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        event = Event(
            event_id=uuid.uuid4().hex,
            event_type=event_type,
            trace=trace,
            state=state,
            payload=payload or {},
        )
        await self.publish(event)
        return event

    def snapshot(self) -> list[Event]:
        return list(self._events)
