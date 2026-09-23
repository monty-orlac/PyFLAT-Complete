import pyflat_numpy as np           # pyright: ignore[reportUnusedImport]
import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]

import sys
import pathlib
import textwrap
import argparse
import traceback

import fe
import linalg
import metadata
from pyflat_util import *
from card_reader import CardReader, ParseGeneralError


# **********************************************************
# ファイル読み込み
# **********************************************************
def main_read_model(reader: CardReader, writer: TextWriter) -> fe.Model:
    try:
        # ------------------------------------------
        #  NODE section
        # ------------------------------------------
        reader.enter_block("NODE")
        writer.write("NODE section...\n")
        nodes: fe.NodeArray = read_node(reader)

        # ------------------------------------------
        #  MATERIAL section
        # ------------------------------------------
        reader.enter_block("MATERIAL")
        writer.write("MATERIAL section...\n")
        lib_matrs: dict[int, fe.MaterialBase] = {it.label: it for it in
            read_material(reader)}

        # ------------------------------------------
        #  PROPERTY section
        # ------------------------------------------
        reader.enter_block("PROPERTY")
        writer.write("PROPERTY section...\n")
        lib_props: dict[int, fe.PropertyBase] = {it.label: it for it in
            read_property(reader, lib_matrs)}

        # ------------------------------------------
        #  ELEMENT section
        # ------------------------------------------
        reader.enter_block("ELEMENT")
        writer.write("ELEMENT section...\n")
        elems: fe.ElementArray = read_element(reader, nodes, lib_props)

        writer.write("Done.\n")
        writer.write("\n")

    except RuntimeError as ex:
        sys.stderr.write(f"ERROR in parsing an input file:\n")
        sys.stderr.write("  " + str(ex) + "\n")
        if hasattr(ex, "__notes__"):
            for sline in ex.__notes__:
                sys.stderr.write(f"    {sline:s}\n")
        sys.exit(1)

    # ------------------------------------------
    #   Create Model instance
    # ------------------------------------------
    model: fe.Model = fe.Model(nodes, elems)
    return model


# ファイル読み込み時にラベルが重複していないかどうかを
# 確認するための確認ツール。
class UniqueLabelChecker:
    __slots__ = ["__kind", "__labels"]

    def __init__(self, kind: str):
        self.__kind: str = kind
        self.__labels: set[int] = set()
        return

    def add_check(self, label: int) -> None:
        if label in self.__labels:
            raise RuntimeError(f"Duplicate {self.__kind:s} label {label:d} is found.")
        self.__labels.add(label)
        return


def read_node(reader: CardReader) -> fe.NodeArray:
    cardlib: dict[str, str] = {fe.Node.TYPE: fe.Node.get_card_initializer()}

    items: list[fe.Node] = []
    checker = UniqueLabelChecker("node")
    for card in reader.iter_cards(cardlib):
        try:
            label: int = card.integers[0]
            checker.add_check(label)
            fixed: tuple[bool, bool] = (bool(card.integers[1]), bool(card.integers[2]))
            coord: tuple[float, float] = (card.floats[0], card.floats[1])
            item: fe.Node = fe.Node(label, len(items), coord, fixed)
        except RuntimeError as ex:
            raise ParseGeneralError(card.lineno, f"<{card.label:s}> {ex!s}") from ex
        items.append(item)
    nodes: fe.NodeArray = fe.NodeArray(items)
    return nodes


def read_material(reader: CardReader) -> cabc.Sequence[fe.MaterialBase]:
    typelib: dict[str, type[fe.MaterialBase]] = {
        cls.TYPE: cls for cls in get_all_subclasses(fe.MaterialBase)
        if hasattr(cls, "TYPE")}
    cardlib: dict[str, str] = {k: v.get_card_initializer() for k, v in typelib.items()}

    items: list[fe.MaterialBase] = []
    checker = UniqueLabelChecker("material")
    for card in reader.iter_cards(cardlib):
        try:
            label: int = card.integers[0]
            checker.add_check(label)
            cls: type[fe.MaterialBase] = typelib[card.label]
            item: fe.MaterialBase = cls.get_instance(label, card.floats)
        except RuntimeError as ex:
            raise ParseGeneralError(card.lineno, f"<{card.label:s}> {ex!s}") from ex
        items.append(item)
    return items


def read_property(reader: CardReader, lib_matrs: dict[int, fe.MaterialBase]) -> cabc.Sequence[fe.PropertyBase]:
    typelib: dict[str, type[fe.PropertyBase]] = {
        cls.TYPE: cls for cls in get_all_subclasses(fe.PropertyBase)
        if hasattr(cls, "TYPE")}
    cardlib: dict[str, str] = {k: v.get_card_initializer() for k, v in typelib.items()}

    items: list[fe.PropertyBase] = []
    checker = UniqueLabelChecker("property")
    for card in reader.iter_cards(cardlib):
        try:
            label: int = card.integers[0]
            checker.add_check(label)
            cls: type[fe.PropertyBase] = typelib[card.label]
            matr: fe.MaterialBase | None = lib_matrs.get(card.integers[1], None)
            if matr is None:
                raise RuntimeError(f"Specified material label {card.integers[1]:d} not found.")
            item: fe.PropertyBase = cls.get_instance(label, matr, card.floats)
        except RuntimeError as ex:
            raise ParseGeneralError(card.lineno, f"<{card.label:s}> {ex!s}") from ex
        items.append(item)
    return items


def read_element(reader: CardReader, nodes: fe.NodeArray, lib_props: dict[int, fe.PropertyBase]) -> fe.ElementArray:
    typelib: dict[str, type[fe.ElementBase]] = {
        cls.TYPE: cls for cls in get_all_subclasses(fe.ElementBase)
        if hasattr(cls, "TYPE")}
    cardlib: dict[str, str] = {k: v.get_card_initializer() for k, v in typelib.items()}

    items: list[fe.ElementBase] = []
    checker = UniqueLabelChecker("element")
    for card in reader.iter_cards(cardlib):
        try:
            label: int = card.integers[0]
            checker.add_check(label)
            cls: type[fe.ElementBase] = typelib[card.label]
            prop: fe.PropertyBase | None = lib_props.get(card.integers[1], None)
            if prop is None:
                raise RuntimeError(f"Specified property label {card.integers[1]:d} not found.")
            elnodes: list[fe.Node] = [nodes.get_by_label(nlb) for nlb in card.integers[2:]]
            item: fe.ElementBase = cls.get_instance(label, prop, elnodes)
        except RuntimeError as ex:
            raise ParseGeneralError(card.lineno, f"<{card.label:s}> {ex!s}") from ex
        items.append(item)
    return fe.ElementArray(items)


def main_read_loadcase(reader: CardReader, model: fe.Model
        ) -> tuple[list[fe.LoadBase], dict[int, float]]:
    typelib: dict[str, type[fe.BaseCondition]] = {
        cls.TYPE: cls for cls in get_all_subclasses(fe.BaseCondition)
        if hasattr(cls, "TYPE")}
    cardlib: dict[str, str] = {k: v.get_card_initializer() for k, v in typelib.items()}

    loads: list[fe.LoadBase] = []
    bounds: list[fe.BoundarySPC] = []
    try:
        for card in reader.iter_cards(cardlib):
            try:
                cls: type[fe.BaseCondition] = typelib[card.label]
                item: fe.BaseCondition = cls.get_instance(model, card.integers, card.floats)
            except RuntimeError as ex:
                raise ParseGeneralError(card.lineno, f"<{card.label:s}> {ex!s}") from ex
            if isinstance(item, fe.BoundarySPC):
                bounds.append(item)
            elif isinstance(item, fe.LoadBase):
                loads.append(item)
            else:
                raise AssertionError("Unknown loading/boundary condition object is found.")
    except RuntimeError as ex:
        sys.stderr.write(f"ERROR in parsing an input file:\n")
        sys.stderr.write("  " + str(ex) + "\n")
        if hasattr(ex, "__notes__"):
            for s in ex.__notes__:
                sys.stderr.write("    " + s + "\n")
        sys.exit(1)
    spcs: dict[int, float] = fe.BoundarySPC.get_list_spc(bounds)
    return loads, spcs


def main_assemble_shape(model: fe.Model, matrix: linalg.IShape) -> None:
    for elem in model.elements:
        lm: cabc.Sequence[int] = elem.get_lm()
        matrix.assemble(lm)
    return


def main_assemble(model: fe.Model, matrix: linalg.IBuilder, spcs: cabc.Mapping[int, float], rhs: np.VectorType) -> None:
    for index, value in spcs.items():
        matrix.add_value(index, index, 1.0)
        rhs[index] = value

    # 要素剛性行列の計算に失敗した要素 (逆向き・つぶれた要素など) の一覧
    errmsg: list[str] = []

    for elem in model.elements:
        coords: np.MatrixType = elem.get_coord_matrix()
        try:
            matrix_local: np.MatrixType = elem.calc_stiffness_matrix(coords)
        except RuntimeError as ex:
            # エラーになった要素を記録し、残りの要素のチェックを続ける
            errmsg.append(f"Element {elem.label:d} ({elem.TYPE:s}): {ex!s}")
            continue

        lm: cabc.Sequence[int] = elem.get_lm()
        lm_spc: list[int] = [-1 if index in spcs else index for index in lm]
        for jl, jg in enumerate(lm):
            spc_jg: float | None = spcs.get(jg)
            if spc_jg is None:
                continue
            for il, ig in enumerate(lm_spc):
                if il == jl:
                    continue
                if ig < 0:
                    continue
                rhs[ig] += -matrix_local[(il, jl)] * spc_jg
        matrix.assemble(lm_spc, matrix_local)

    # 要素剛性行列の計算に失敗した要素があれば、まとめてエラーとする
    if errmsg:
        err: RuntimeError = RuntimeError("Failed to calculate the stiffness matrix of following element(s).")
        for msg in errmsg:
            err.add_note(msg)
        raise err
    return


def get_argument_parser(is_preparser: bool=False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=textwrap.dedent("""\
            2D simple structural analysis solver using the Finite Element Method
            for educational purposes.
        """), prog=pathlib.Path(sys.argv[0]).stem, add_help=(not is_preparser))

    parser.add_argument("-V", "--version", help="show version and exit.", action="store_true")
    parser.add_argument("--validate", help=textwrap.dedent("""\
            [[FOR DEBUGGING]] Runs the program in validate mode.
            Version, license, and execution time notation will be skipped.
        """), action="store_true")

    if not is_preparser:
        parser.add_argument("inppath", help=textwrap.dedent("""\
                Text input file containing the finite element model description.
            """), metavar="FILENAME", type=pathlib.Path)
        parser.add_argument("--solver", help=textwrap.dedent("""\
                matrix solver for the global stiffness matrix.
                (choices: %(choices)s) (default: %(default)s)
            """), metavar="TYPE", default="DIRECT", choices=linalg.SOLVER_TYPES.keys())
        parser.add_argument("-v", "--verbose", help=textwrap.dedent("""\
                enable verbose mode to provide detailed diagnostic output.
            """), action="store_true")
        parser.add_argument("--hourglass", help=textwrap.dedent("""\
                stabilization coefficient value for hourglass control
                via gamma-projection method (default: %(default)s)
            """), metavar="VALUE", type=float, default=fe.CommonConfig.HOURGLASS_STIFFNESS)
        parser.add_argument("--reorder", help=textwrap.dedent("""\
                specify the reordering algorithm for FEM nodal numbering. Note that
                this option is available when --solver=SKYLINE or --solver=DIRECT.
                For other types of solvers, --reorder is always SIMPLE
                (choices: %(choices)s) (default: %(default)s)
            """), metavar="METHOD", choices=[it.name for it in fe.AlgorithmReorder], default=fe.AlgorithmReorder.RCM.name)
        parser.add_argument("--cutoff", help=textwrap.dedent("""\
                relative tolerance to truncate small result values below the maximum peak (default: %(default)s)
            """), metavar="TOL", type=float, default=1.0e-8)
        parser.add_argument("--debug", help=textwrap.dedent("""\
            [[FOR DEBUGGING]] Runs the program in developer debug mode.
            This option enables full tracebacks on errors.
        """), action="store_true")

    return parser


def main_initialize(writer: TextWriter) -> argparse.Namespace:

    # ******************************************************
    #  コマンドライン引数の解析
    # ******************************************************
    parser: argparse.ArgumentParser = get_argument_parser()
    config: argparse.Namespace = parser.parse_args()

    if config.version:
        # --versionを指定していた場合、バージョン番号を表示してプログラムを終了
        metadata.show_version(writer)
        sys.exit(0)

    # ******************************************************
    #  タイトルの表示
    # ******************************************************
    metadata.show_title(writer, config.validate)

    # ******************************************************
    #  コマンドラインオプションの設定を適用
    # ******************************************************
    fe.CommonConfig.HOURGLASS_STIFFNESS = config.hourglass
    config.solver = linalg.SOLVER_TYPES[config.solver]()
    fe.CommonConfig.REORDER_ALGORITHM = fe.AlgorithmReorder[config.reorder]
    fe.CommonConfig.SMALL_STRESS_CUTOFF = config.cutoff
    fe.CommonConfig.SMALL_DISPLACEMENT_CUTOFF = config.cutoff

    # ******************************************************
    #  コマンドラインから入力された設定項目の表示
    # ******************************************************
    writer.print_subsection("Configuration via commandline arguments")
    writer.write("{:40s}:{:>12s}\n".format("Matrix solver", config.solver.TYPE))
    writer.write("  ... {:s}\n".format(config.solver.DESC))
    if not config.solver.NEEDS_REORDER:
        writer.write("       (Since reordering does not affect this solver's performance,\n")
        writer.write("        the reordering algorithm is set to SIMPLE.)\n")
        fe.CommonConfig.REORDER_ALGORITHM = fe.AlgorithmReorder.SIMPLE
    writer.write("{:40s}:{:>12s}\n".format("Reordering algorithm", fe.CommonConfig.REORDER_ALGORITHM.name))
    writer.write("  ... {:s}\n".format(fe.CommonConfig.REORDER_ALGORITHM.value))
    writer.write("{:40s}:{:>12s}\n".format("Verbose diagnostics output mode", str(config.verbose).upper()))
    writer.write("{:40s}:{:>12.4e}\n".format("Peak-relative truncation tolerance", config.cutoff))
    writer.write("{:40s}:{:>12.4f}\n".format("Hourglass stabilization coefficient", config.hourglass))
    writer.write("\n")

    return config


def main_routine(writer: TextWriter, config: argparse.Namespace) -> None:

    timer: Timer = Timer()

    # ******************************************************
    #  入力ファイルのオープン
    # ******************************************************
    fp: typing.TextIO
    try:
        fp = open(config.inppath, "r")
    except Exception as ex:
        sys.stderr.write(f"Failed to open the specified input file:\n")
        sys.stderr.write("  " + "    ".join(traceback.format_exception_only(type(ex), ex)).strip())
        sys.stderr.write("\n")
        sys.exit(1)

    # ******************************************************
    #  モデルデータ読み込み
    # ******************************************************
    writer.print_section("Pre-assembling the global stiffness matrix")

    timer.measure(1, "Model data input")
    writer.print_subsection("Reading model data")
    writer.write("Analysis input file:\n")
    writer.write(f"  {config.inppath!s}\n")
    writer.write("\n")

    reader: CardReader = CardReader(fp)
    writer.write("Start reading the file:\n")
    model: fe.Model = main_read_model(reader, writer)

    # ******************************************************
    #   全体剛性行列の形状の決定
    # ******************************************************
    timer.measure(2, "Symbolic Assembly")
    writer.print_subsection("Performing symbolic assembly")

    # ------------------------------------------
    # 節点の自由度を全体剛性行列に割り付ける
    # ------------------------------------------
    writer.write("Start element loop...\n")
    neq: int = model.reorder(writer)
    writer.write("  Done.\n")
    writer.write("\n")

    if config.verbose and neq <= 50:
        writer.write("<<DEBUG>> Nodal DOFs assignment\n")
        model.print_node_dof(writer)
    writer.write(f"Total {neq} DOFs assigned to the model.\n")
    writer.write("\n")

    solver_generator: linalg.IGenerator = config.solver
    matrix_shape: linalg.IShape = solver_generator.generate(neq)

    # ------------------------------------------
    # 自由度の割り付けをもとに全体剛性行列の形状を決定する
    # ------------------------------------------
    writer.print_subsubsection("Analyzing non-zero sparsity pattern")
    writer.write("Start element loop...\n")
    main_assemble_shape(model, matrix_shape)
    writer.write("  Done.\n")
    writer.write("\n")

    # ******************************************************
    #   全体剛性行列の作成
    # ******************************************************
    reader.enter_block("LOADCASE")
    ilcase: int = 0
    while reader.try_enter_block("CASE"):
        timer.measure(3, "Numeric Assembly")
        ilcase += 1
        writer.print_section(f"Load case #{ilcase:02d}")
        writer.print_subsection("Assembling global stiffness matrix")

        # 荷重境界条件の読み込み
        writer.print_subsubsection("Applying loading & boundary conditions")
        writer.write("Reading data...\n")
        loads: cabc.Sequence[fe.LoadBase]
        spcs: dict[int, float]
        loads, spcs = main_read_loadcase(reader, model)

        # 右辺値ベクトルのメモリ確保
        rhs: np.VectorType = np.zeros((matrix_shape.size, ))

        # 荷重境界条件の適用
        writer.write("Generating load vector...\n")
        for it in loads:
            it.apply(model, rhs)
        writer.write("Done.\n")
        writer.write("\n")

        if config.verbose and matrix_shape.size <= 50:
            writer.write("<<DEBUG>> Load vector\n")
            linalg.print_vector(writer, rhs)
            writer.write("\n")

        # --------------------------------------
        #   全体剛性行列の構築
        # --------------------------------------
        writer.print_subsubsection("Performing numeric assembly")

        # 全体剛性行列のメモリ確保
        matrix_builder: linalg.IBuilder = matrix_shape.allocate(spcs)

        # 要素剛性行列の和で全体剛性行列を生成
        writer.write("Start element loop...\n")
        main_assemble(model, matrix_builder, spcs, rhs)
        writer.write("  Done.\n")
        writer.write("\n")

        # 全体剛性行列が完成したのでfixする
        matrix_solver: linalg.ISolver = matrix_builder.complete()
        del matrix_builder

        if config.verbose and matrix_solver.size <= 15:
            writer.write("<<DEBUG>> Matrix components:\n")
            matrix_solver.print_matrix(writer)
            writer.write("\n")

        # ******************************************************
        #   行列を解いて解を求める
        # ******************************************************
        writer.print_subsection(f"Solving the matrix")

        # 行列の求解
        timer.measure(4, "Matrix solve prep.")
        matrix_factorized: linalg.IFactorized = matrix_solver.factorize(writer)
        writer.write("\n")

        timer.measure(5, "Matrix solve run")
        rhs = matrix_factorized.solve(rhs, writer)
        writer.write("\n")

        # ******************************************************
        #   結果表示処理
        # ******************************************************
        timer.measure(6, "Post process")
        writer.print_subsection(f"Post-processing")

        # 節点変位を出力
        writer.print_subsubsection("Nodal displacement result")
        model.print_node_result(writer, rhs)
        writer.write("\n")

        # 要素応力を出力
        writer.print_subsubsection("Element stress result")
        model.print_element_result(writer, rhs)
        writer.write("\n")

    reader.exit_block()

    # ******************************************************
    #   実行時間の表示
    # ******************************************************
    timer.stop()
    if not config.validate:
        writer.print_section("Solution time")
        timer.print_result(writer, header="")
    return


def main() -> None:
    writer: TextWriter = TextWriter(sys.stdout)
    config: argparse.Namespace = main_initialize(writer)
    try:
        main_routine(writer, config)
    except Exception as ex:
        sys.stderr.write("  ".join(traceback.format_exception_only(type(ex), ex)).strip())
        sys.stderr.write("\n")
        if config.debug:
            sys.stderr.write("\n")
            traceback.print_exc()
        # エラー終了したことを終了コードで呼び出し元に伝える
        sys.exit(1)
    except SystemExit:
        if config.debug:
            sys.stderr.write("\n")
            traceback.print_exc()
        # sys.exit() の終了コードをそのまま呼び出し元に伝える
        raise
