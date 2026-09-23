from __future__ import annotations
import typing
import time
import collections.abc as cabc


class TextWriter(typing.TextIO):
    __slots__ = ["__fp", "__previous_output_level"]

    @property
    def name(self) -> str:
        return self.__fp.name

    def __init__(self, fp: typing.TextIO) -> None:
        self.__fp: typing.TextIO = fp
        self.__previous_output_level: int = 0
        return

    def write(self, s: str = "") -> int:
        self.__fp.write(s)
        self.__previous_output_level = 0
        return len(s)

    def print(self, s: str = "") -> None:
        self.__fp.write(f"{s}\n")
        self.__previous_output_level = 0
        return

    def print_section(self, s: str) -> None:
        self.__fp.write("\n")
        self.__fp.write("********" * 9 + "\n")
        self.__fp.write(f"**  {s}\n")
        self.__fp.write("********" * 9 + "\n")
        self.__fp.write("\n")
        self.__previous_output_level = 3
        return

    def print_subsection(self, s: str) -> None:
        if self.__previous_output_level <= 2:
            self.__fp.write("\n")
        self.__fp.write(s + ":  " + "=" * (57 - len(s)) + "\n")
        self.__previous_output_level = 2
        return

    def print_subsubsection(self, s: str) -> None:
        if self.__previous_output_level <= 1:
            self.__fp.write("\n")
        self.__fp.write(s + ":  " + "-" * (45 - len(s)) + "\n")
        self.__previous_output_level = 1
        return


class Timer:
    __slots__ = ["__counter", "__times", "__labels", "__icounter"]

    def __init__(self):
        self.__counter: float = 0.0
        self.__icounter: int = -1
        self.__times: list[float] = []
        self.__labels: list[str] = []
        return

    def __lap(self) -> None:
        tsta: float = self.__counter
        self.__counter = time.perf_counter()
        if self.__icounter >= 0:
            self.__times[self.__icounter] += self.__counter - tsta
        return

    def measure(self, icounter: int, label: str | None = None) -> None:
        assert 1 <= icounter <= 100
        while len(self.__times) < icounter:
            self.__times.append(0.0)
            self.__labels.append("")
        self.__lap()
        self.__icounter = icounter - 1
        if label:
            self.__labels[self.__icounter] = label
        return

    def stop(self) -> None:
        self.__lap()
        self.__icounter = -1
        return

    def get_elapsed_time(self, icounter: int | None = None) -> float:
        if icounter is None:
            return sum(self.__times)
        assert 1 <= icounter <= len(self.__times)
        return self.__times[icounter - 1]

    def print_result(self, fp: typing.TextIO, header: str="") -> None:
        self.stop()
        fp.write("{:s}{:10s} {:>22s}\n". format(header, "Phase", "Elapsed time[s]"))
        for tm, label in zip(self.__times, self.__labels, strict=True):
            fp.write("{:s}{:20s} {:12.6f}\n".format(header, label, tm))
        fp.write("\n")
        fp.write("{:s}{:20s} {:12.6f}\n".format(header, "*** total ***", self.get_elapsed_time()))
        fp.write("\n")
        return


def get_all_subclasses[T_PARENT_TYPE: type](
        parent_class: T_PARENT_TYPE,
        ) -> cabc.Collection[T_PARENT_TYPE]:
    """指定したクラスの全サブクラスを再帰的に取得する"""
    all_subclasses: set[T_PARENT_TYPE] = set()

    # 直下の子クラスを取得
    direct_subclasses = parent_class.__subclasses__()

    for subclass in direct_subclasses:
        # 子クラス自身を追加
        all_subclasses.add(subclass)

        # さらにその子クラスを再帰的に取得して更新
        all_subclasses.update(get_all_subclasses(subclass))

    return all_subclasses
