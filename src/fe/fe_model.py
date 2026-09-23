import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]

import sys
from collections import deque

from .fe_base import *


class Model(ModelData):
    __slots__ = []

    def print_node_dof(self, fp: typing.TextIO) -> None:
        fp.write("{:>8s}{:>8s}{:>8s}\n".format("nid", "dof_x", "dof_y"))
        for node in self.nodes:
            sdofs: list[str] = ["   FIXED" if dof < 0 else "{:>8d}".format(dof) for dof in node.dof]
            fp.write(f"{node.label:>8d}" + "".join(sdofs) + "\n")
        return

    def print_node_result(self, fp: typing.TextIO, rhs: np.VectorType) -> None:
        if len(self.nodes) == 0:
            return

        # ヘッダー行の出力
        fp.write(" {:>7s} {:>15s} {:>15s}\n".format("nid", "x-disp", "y-disp"))

        # アイテムごとに1行のデータ出力
        tol: float = max(abs(rhs)) * CommonConfig.SMALL_DISPLACEMENT_CUTOFF
        for node in self.nodes:
            fp.write(" {:7d}".format(node.label))
            for idof in node.dof:
                val: float = rhs[idof] if idof >= 0 and abs(rhs[idof]) > tol else 0.0
                fp.write(" {:15.6e}".format(val))
            fp.write("\n")
        return

    def print_element_result(self, fp: typing.TextIO, rhs: np.VectorType) -> None:
        if len(self.elements) == 0:
            return

        # ヘッダー行の出力
        fp.write(" {:>7s} {:>3s} {:>15s} {:>15s} {:>15s} {:>15s}\n".format("eid", "ip", "str11", "str22", "str33", "str12"))

        # アイテムごとに1行のデータ出力
        for it in self.elements:
            lm: cabc.Sequence[int] = it.get_lm()
            arr: list[float] = [rhs[idof] if idof >= 0 else 0.0 for idof in lm]
            coords: np.MatrixType = it.get_coord_matrix()
            stress_tensors: cabc.Sequence[np.VectorType] = it.calc_result(coords, np.array(arr))
            for ip, stress in enumerate(stress_tensors):
                tol: float = np.sqrt(np.sum(stress**2 * np.array([1.0, 1.0, 1.0, 2.0]))) * CommonConfig.SMALL_STRESS_CUTOFF
                fp.write(f" {it.label:7d} {ip+1:3d}")
                for i in range(len(stress)):
                    value: float = 0.0 if abs(stress[i]) < tol else stress[i]
                    fp.write(f" {value:15.6e}")
                fp.write("\n")
        return

    def reorder(self, writer: typing.TextIO) -> int:
        nsize: int
        match CommonConfig.REORDER_ALGORITHM:
            case AlgorithmReorder.SIMPLE:
                nsize = self.__reorder_simple()
            case AlgorithmReorder.RCM:
                nsize = self.__reorder_rcm(writer)
        return nsize
    
    def __reorder_simple(self) -> int:
        nodes: NodeArray = self.nodes

        nsize: int = 0
        for node in nodes:
            for i in range(len(node.dof)):
                if node.dof[i] != Node.DOF_VALUE_FIXED:
                    node.dof[i] = nsize
                    nsize += 1
        return nsize

    def __reorder_rcm(self, writer: typing.TextIO) -> int:
        nodes: NodeArray = self.nodes
        elements: ElementArray = self.elements

        nsize: int = len(nodes)

        # 各節点が登録されている要素の個数をカウント
        #   nodes[i] が含まれる要素の数を counts[i] に保存
        counts: list[int] = [0, ] * nsize
        for elem in elements:
            for node in elem.nodes:
                nid: int = nodes.get_index_by_label(node.label)
                counts[nid] += 1

        # 配列内の開始位置を記録するポインタ列 ptrs[] を生成
        # 同時に、次数が最小の節点について、その節点番号を minid に記録
        ptrs: list[int] = [0, ] * (nsize+1)
        minid: int = 0
        for nid, _ in enumerate(nodes):
            ptrs[nid+1] = ptrs[nid] + counts[nid]
            if 0 < counts[nid] <= counts[minid]:
                minid = nid

        # 孤立節点 (counts[i]==0) を検出
        # メッセージを表示しながら孤立節点を一覧表示し、1つでもあればエラー終了する
        nfix: int = 0
        for nid, node in enumerate(nodes):
            if counts[nid] != 0:
                continue
            if nfix == 0:
                sys.stderr.write("ERROR: The following nodes are not connected to any elements.\n")
            nfix += 1
            sys.stderr.write("{:7d}".format(node.label))
            sys.stderr.write("\n" if nfix % 8 == 0 else ", ")
        if nfix > 0:
            sys.stderr.write("\n" if nfix % 8 != 0 else "")
            sys.stderr.write("  Please remove the isolated nodes or check the element connectivity.\n")
            sys.exit(1)

        # 各節点が含まれる要素を conn[] に記録
        # nodes[i] が含まれる要素の列は下記の通り
        #   conn[ptr[i]:ptr[i+1]-1]
        conn: list[int] = [-1, ] * ptrs[nsize]
        pos: list[int] = list(ptrs)
        for eid, elem in enumerate(elements):
            for node in elem.nodes:
                nid: int = nodes.get_index_by_label(node.label)
                conn[pos[nid]] = eid
                pos[nid] += 1

        # posはもう使わないのでメモリを解放
        del pos

        # メッシュ節点を幅優先探索し、自由度番号を順番に付与
        #   次に探索する節点番号をキュー q に記録し、訪問済みの節点は
        #   countsを0にして対応
        MAXINT: int = nsize * 2
        que: deque[int] = deque()
        que.append(minid)
        counts[minid] = MAXINT

        nmsize: int = 0
        while que:
            # 次に訪問する節点を取り出し、自由度番号を付与
            nid: int = que.popleft()
            node: Node = nodes[nid]
            for idof in range(len(node.dof)):
                if node.dof[idof] != Node.DOF_VALUE_FIXED:
                    node.dof[idof] = nmsize
                    nmsize += 1

            # 未訪問の隣接節点をキューに登録
            # 本来RCMであれば次数 (隣接節点) が少ない順番に並べ替えて訪問するが
            # ここでは並べ替えまでは行わない
            for ptr in range(ptrs[nid], ptrs[nid+1]):
                eid: int = conn[ptr]
                for node in elements[eid].nodes:
                    nid: int = nodes.get_index_by_label(node.label)
                    if counts[nid] == MAXINT:
                        continue
                    que.append(nid)
                    counts[nid] = MAXINT

        # 自由度番号を逆順にし、同時に自由度番号を振っていない節点を確認
        #   (モデルに未接続の複数領域があった場合に起こりうる)
        unnumbered_nodes: int = 0
        for nid, node in enumerate(nodes):
            if counts[nid] != MAXINT:
                unnumbered_nodes += 1
            else:
                for idof in range(len(node.dof)):
                    if node.dof[idof] >= 0:
                        node.dof[idof] = nmsize - node.dof[idof] - 1
        if unnumbered_nodes > 0:
            sys.stderr.write("ERROR: A total of {:d} nodes are not numbered.\n".format(unnumbered_nodes))
            sys.stderr.write("  This error occurs when there are multiple unconnected regions in the model.\n")
            sys.stderr.write("  Please check the mesh connectivity.\n")
            sys.exit(1)

        return nmsize
