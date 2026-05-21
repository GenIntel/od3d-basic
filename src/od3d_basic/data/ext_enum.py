import enum
import sys
from enum import Enum


class ExtEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


if sys.version_info[1] >= 12:  # python3.12

    class StrEnum(ExtEnum, enum.StrEnum):
        pass

else:

    class StrEnum(str, ExtEnum):
        def __str__(self) -> str:
            return self.value
