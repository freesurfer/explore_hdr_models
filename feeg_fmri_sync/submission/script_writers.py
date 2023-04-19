from abc import abstractmethod, ABC
from typing import List, Generator


class ScriptWriter(ABC):
    @abstractmethod
    def get_lines(self) -> List[str]:
        raise NotImplementedError()


class IterativeScriptWriter(ABC):
    @abstractmethod
    def get_lines_for_identifier(self, identifier: int) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_identifiers(self) -> Generator[int]:
        raise NotImplementedError()