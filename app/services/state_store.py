from collections import OrderedDict, deque
from typing import Any

from pydantic import BaseModel


class MonitoringStateStore:
    """Small in-process record of recent monitoring outputs.

    This is intentionally lightweight for local/SIH use. It gives the backend
    one place to retain recent snapshots, events, and route evaluations without
    turning persistence into a separate infrastructure project.
    """

    def __init__(self, max_records: int = 64):
        self.max_records = max_records
        self._snapshots: deque[dict[str, Any]] = deque(maxlen=max_records)
        self._events: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._routes: deque[dict[str, Any]] = deque(maxlen=max_records)
        self._latest_scenario_stage: str | None = None

    def record_snapshot(
        self,
        snapshot: BaseModel,
        *,
        scenario_stage: str | None = None,
    ) -> None:
        self._snapshots.append(_jsonable(snapshot))
        if scenario_stage is not None:
            self._latest_scenario_stage = scenario_stage

    def record_events(self, events: list[BaseModel]) -> None:
        for event in events:
            payload = _jsonable(event)
            event_id = str(payload.get("event_id", ""))
            if not event_id:
                continue
            self._events[event_id] = payload
            self._events.move_to_end(event_id)
        while len(self._events) > self.max_records:
            self._events.popitem(last=False)

    def record_route_evaluation(self, route_result: BaseModel) -> None:
        self._routes.append(_jsonable(route_result))

    def clear(self) -> None:
        self._snapshots.clear()
        self._events.clear()
        self._routes.clear()
        self._latest_scenario_stage = None

    def recent_events(self) -> list[dict[str, Any]]:
        return list(self._events.values())

    def latest_snapshot(self) -> dict[str, Any] | None:
        if not self._snapshots:
            return None
        return self._snapshots[-1]

    def latest_scenario_stage(
        self,
        *,
        default: str,
    ) -> str:
        return self._latest_scenario_stage or default

    def summary(self) -> dict[str, Any]:
        latest_snapshot = self._snapshots[-1] if self._snapshots else None
        latest_route = self._routes[-1] if self._routes else None
        return {
            "snapshot_count": len(self._snapshots),
            "event_count": len(self._events),
            "route_evaluation_count": len(self._routes),
            "latest_snapshot_id": latest_snapshot.get("snapshot_id")
            if latest_snapshot
            else None,
            "latest_route_status": latest_route.get("status")
            if latest_route
            else None,
            "latest_scenario_stage": self._latest_scenario_stage,
        }


def _jsonable(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")
