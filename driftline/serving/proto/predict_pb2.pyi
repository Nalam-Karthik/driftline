from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TripRequest(_message.Message):
    __slots__ = ("pickup_unix_s", "pu_location_id", "do_location_id", "trip_distance", "passenger_count", "ratecode_id", "request_id")
    PICKUP_UNIX_S_FIELD_NUMBER: _ClassVar[int]
    PU_LOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    DO_LOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    TRIP_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    PASSENGER_COUNT_FIELD_NUMBER: _ClassVar[int]
    RATECODE_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    pickup_unix_s: int
    pu_location_id: int
    do_location_id: int
    trip_distance: float
    passenger_count: int
    ratecode_id: int
    request_id: str
    def __init__(self, pickup_unix_s: _Optional[int] = ..., pu_location_id: _Optional[int] = ..., do_location_id: _Optional[int] = ..., trip_distance: _Optional[float] = ..., passenger_count: _Optional[int] = ..., ratecode_id: _Optional[int] = ..., request_id: _Optional[str] = ...) -> None: ...

class TripPrediction(_message.Message):
    __slots__ = ("request_id", "duration_min", "model_version", "model_role")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    DURATION_MIN_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    MODEL_ROLE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    duration_min: float
    model_version: str
    model_role: str
    def __init__(self, request_id: _Optional[str] = ..., duration_min: _Optional[float] = ..., model_version: _Optional[str] = ..., model_role: _Optional[str] = ...) -> None: ...

class TripBatch(_message.Message):
    __slots__ = ("trips",)
    TRIPS_FIELD_NUMBER: _ClassVar[int]
    trips: _containers.RepeatedCompositeFieldContainer[TripRequest]
    def __init__(self, trips: _Optional[_Iterable[_Union[TripRequest, _Mapping]]] = ...) -> None: ...

class TripPredictions(_message.Message):
    __slots__ = ("predictions",)
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    predictions: _containers.RepeatedCompositeFieldContainer[TripPrediction]
    def __init__(self, predictions: _Optional[_Iterable[_Union[TripPrediction, _Mapping]]] = ...) -> None: ...
