from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Task(_message.Message):
    __slots__ = ("workerId", "taskId")
    WORKERID_FIELD_NUMBER: _ClassVar[int]
    TASKID_FIELD_NUMBER: _ClassVar[int]
    workerId: int
    taskId: int
    def __init__(self, workerId: _Optional[int] = ..., taskId: _Optional[int] = ...) -> None: ...

class Range(_message.Message):
    __slots__ = ("begin", "end")
    BEGIN_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    begin: int
    end: int
    def __init__(self, begin: _Optional[int] = ..., end: _Optional[int] = ...) -> None: ...

class KeyList(_message.Message):
    __slots__ = ("key",)
    KEY_FIELD_NUMBER: _ClassVar[int]
    key: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, key: _Optional[_Iterable[int]] = ...) -> None: ...

class Values(_message.Message):
    __slots__ = ("keys", "data", "width")
    KEYS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    keys: KeyList
    data: _containers.RepeatedScalarFieldContainer[float]
    width: int
    def __init__(self, keys: _Optional[_Union[KeyList, _Mapping]] = ..., data: _Optional[_Iterable[float]] = ..., width: _Optional[int] = ...) -> None: ...

class CallOptions(_message.Message):
    __slots__ = ("deps", "iteration", "consistency")
    DEPS_FIELD_NUMBER: _ClassVar[int]
    ITERATION_FIELD_NUMBER: _ClassVar[int]
    CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    deps: _containers.RepeatedCompositeFieldContainer[Task]
    iteration: int
    consistency: str
    def __init__(self, deps: _Optional[_Iterable[_Union[Task, _Mapping]]] = ..., iteration: _Optional[int] = ..., consistency: _Optional[str] = ...) -> None: ...

class PullRequest(_message.Message):
    __slots__ = ("workerId", "r", "k", "opts")
    WORKERID_FIELD_NUMBER: _ClassVar[int]
    R_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    OPTS_FIELD_NUMBER: _ClassVar[int]
    workerId: int
    r: Range
    k: KeyList
    opts: CallOptions
    def __init__(self, workerId: _Optional[int] = ..., r: _Optional[_Union[Range, _Mapping]] = ..., k: _Optional[_Union[KeyList, _Mapping]] = ..., opts: _Optional[_Union[CallOptions, _Mapping]] = ...) -> None: ...

class PushRequest(_message.Message):
    __slots__ = ("workerId", "r", "k", "values", "opts")
    WORKERID_FIELD_NUMBER: _ClassVar[int]
    R_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    OPTS_FIELD_NUMBER: _ClassVar[int]
    workerId: int
    r: Range
    k: KeyList
    values: _containers.RepeatedScalarFieldContainer[float]
    opts: CallOptions
    def __init__(self, workerId: _Optional[int] = ..., r: _Optional[_Union[Range, _Mapping]] = ..., k: _Optional[_Union[KeyList, _Mapping]] = ..., values: _Optional[_Iterable[float]] = ..., opts: _Optional[_Union[CallOptions, _Mapping]] = ...) -> None: ...

class Result(_message.Message):
    __slots__ = ("v", "ack")
    V_FIELD_NUMBER: _ClassVar[int]
    ACK_FIELD_NUMBER: _ClassVar[int]
    v: Values
    ack: bool
    def __init__(self, v: _Optional[_Union[Values, _Mapping]] = ..., ack: _Optional[bool] = ...) -> None: ...
