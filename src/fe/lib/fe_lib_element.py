import pyflat_numpy as np           # pyright: ignore[reportUnusedImport]
import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]

import math


def print_header(fp: typing.TextIO, name: str, label: int, stype: str, indent: str = "  ") -> None:
    fp.write(f"{indent:s}{name:s} #{label:03d}: {stype:s}\n")
    return


def print_body(fp: typing.TextIO, sline: str, indent: str = "    ") -> None:
    fp.write(f"{indent:s}{sline:s}\n")
    return


def print_parameter(fp: typing.TextIO, name: str, value: float | int | str | None, indent: str = "    ") -> None:
    assert isinstance(value, (float, int, str)) or value is None
    fp.write("{}{:20s} ".format(indent, name))
    if isinstance(value, float):
        fp.write("{:15.6e}".format(value))
    elif isinstance(value, int):
        fp.write("{:15d}".format(value))
    elif isinstance(value, str):
        assert len(value) <= 15
        fp.write("{:>15s}".format(value))
    fp.write("\n")
    return


def _gauss_info_1d(nedge: int) -> cabc.Iterator[tuple[float, float]]:
    match nedge:
        case 2:
            yield (-0.5773502692, 1.0)
            yield ( 0.5773502692, 1.0)
            return
        case 3:
            yield (-0.7745966692, 5.0/9.0)
            yield ( 0.0000000000, 8.0/9.0)
            yield ( 0.7745966692, 5.0/9.0)
            return
        case _:
            pass
    raise AssertionError(f"Unknown nedge={nedge} in {__file__}._gauss_info_1d().")


def _shape_func_1d(nedge: int, xi: float) -> np.VectorType:
    match nedge:
        case 2:
            return np.array([0.5 * (1.0 - xi), 0.5 * (1.0 + xi)])
        case 3:
            return np.array([
                0.5 * xi * (xi - 1.0),
                0.5 * xi * (xi + 1.0),
                1.0 - xi * xi, ])
        case _:
            pass
    raise AssertionError(f"Unknown nedge={nedge} in {__file__}._shape_func_1d().")


def _shape_derivs_1d(nedge: int, xi: float) -> np.VectorType:
    match nedge:
        case 2:
            return np.array([-0.5, 0.5])
        case 3:
            return np.array([xi - 0.5, xi + 0.5, -2.0 * xi])
        case _:
            pass
    raise AssertionError(f"Unknown nedge={nedge} in {__file__}._shape_derivs_1d().")


def _gauss_info_2d(nnode: int, nord: typing.Literal[0, -1]) -> cabc.Iterator[tuple[float, float, float]]:
    if nnode == 3:
        if nord == 0:
            yield (1.0/3.0, 1.0/3.0, 1.0/2.0)
            return
    elif nnode == 4:
        if nord == 0:
            yield (-0.5773502692, -0.5773502692, 1.0)
            yield ( 0.5773502692, -0.5773502692, 1.0)
            yield (-0.5773502692,  0.5773502692, 1.0)
            yield ( 0.5773502692,  0.5773502692, 1.0)
            return
        elif nord == -1:
            yield (0.0, 0.0, 4.0)
            return
    elif nnode == 6:
        if nord == 0:
            yield (1.0/6.0, 1.0/6.0, 1.0/6.0)
            yield (4.0/6.0, 1.0/6.0, 1.0/6.0)
            yield (1.0/6.0, 4.0/6.0, 1.0/6.0)
            return
    raise AssertionError(f"Unknown (nnode, nord)=({nnode}, {nord}) in {__file__}._gauss_info_2d().")
    return


def _shape_derivs_2d(nnode: int, xi: float, eta: float) -> np.MatrixType:
    match nnode:
        case 3:
            return np.array([
                [-1.0, +1.0,  0.0],
                [-1.0,  0.0, +1.0], ])
        case 4:
            return np.array([
                [-0.25 * (1.0 - eta), +0.25 * (1.0 - eta), +0.25 * (1.0 + eta), -0.25 * (1.0 + eta)],
                [-0.25 * (1.0 - xi),  -0.25 * (1.0 + xi),  +0.25 * (1.0 + xi),  +0.25 * (1.0 - xi)], ])
        case 6:
            fn1: float = 1.0 - xi - eta
            fn2: float = xi
            fn3: float = eta
            return np.array([
                [-4.0*fn1+1.0, 4.0*fn2-1.0, 0.0,         +4.0*(fn1-fn2), +4.0*fn3, -4.0*fn3,],
                [-4.0*fn1+1.0, 0.0,         4.0*fn3-1.0, -4.0*fn2,       +4.0*fn2, +4.0*(fn1-fn3),],
            ])
        case _:
            pass
    raise AssertionError(f"Unknown nnode={nnode} in {__file__}._shape_derivs_2d().")


def _shape_func_2d(nnode: int, xi: float, eta: float) -> np.VectorType:
    match nnode:
        case 3:
            return np.array([1.0 - xi - eta, xi, eta, ])
        case 4:
            return np.array([
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta),
            ])
        case 6:
            fn1: float = 1.0 - xi - eta
            fn2: float = xi
            fn3: float = eta
            return np.array([
                fn1 * (2.0 * fn1 - 1.0),
                fn2 * (2.0 * fn2 - 1.0),
                fn3 * (2.0 * fn3 - 1.0),
                4.0 * fn1 * fn2,
                4.0 * fn2 * fn3,
                4.0 * fn3 * fn1,
            ])
        case _:
            pass
    raise AssertionError(f"Unknown nnode={nnode} in {__file__}._shape_func_2d().")


_ELEMENT_EDGE_IDS: cabc.Mapping[int, cabc.Sequence[cabc.Sequence[int]]] = {
    3: [[0, 1], [1, 2], [2, 0], ],
    4: [[0, 1], [1, 2], [2, 3], [3, 0], ],
    6: [[0, 1, 3], [1, 2, 4], [2, 0, 5], ],
}


def get_element_edge_ids(nnode: int) -> cabc.Sequence[cabc.Sequence[int]]:
    edge_ids = _ELEMENT_EDGE_IDS.get(nnode, None)
    if edge_ids is None:
        raise AssertionError(f"Unknown nnode={nnode} in {__file__}.get_element_edge_ids().")
    return edge_ids


def calc_element_truss(
        coords: np.MatrixType, young: float, area: float,
        ans_elem: np.VectorType | None = None
        ) -> np.MatrixType | list[np.VectorType]:

    ldir: np.VectorType = coords[1, :] - coords[0, :]
    lenxy: float = float(np.linalg.norm(ldir))
    if not lenxy > 0.0:
        raise RuntimeError("Element length is zero. Check the nodal coordinates.")
    ldir /= lenxy
    bee: np.MatrixType = np.array([[-ldir[0], -ldir[1], +ldir[0], +ldir[1]], ]) / lenxy

    if ans_elem is None:
        # 要素剛性行列の計算
        matrix_local: np.MatrixType = bee.transpose() * young @ bee
        matrix_local *= area * lenxy
        return matrix_local

    # 結果の表示
    stress: float = float(bee.transpose()[:, 0].dot(ans_elem) * young)
    return [np.array([stress, ]), ]


def calc_edge_2d(
        coords_edge: np.MatrixType,
        thickness: float,
        value: float, direction: tuple[float, float] | None = None
        ) -> np.VectorType:
    # thicknessの値で要素タイプを区別する
    #   thickness > 0.0 なら平面要素
    #   thickness < 0.0 なら軸対称要素

    # 等価節点外力ベクトル
    nedge: int = coords_edge.shape[0]
    ndim: int = coords_edge.shape[1]
    ret: np.VectorType = np.zeros(nedge * 2)
    thick: float = thickness

    # Gauss積分点ごとにループを回す
    for xi, weight in _gauss_info_1d(nedge):

        # 1次元形状関数 N1, N2
        slocal: np.VectorType = _shape_func_1d(nedge, xi)

        # 形状関数の自然座標微分 dN/dxi
        dslocal: np.VectorType = _shape_derivs_1d(nedge, xi)

        # ヤコビアンの計算 (エッジの接線ベクトル)
        # dx/dxi = Σ (dNi/dxi * xi), dy/dxi = Σ (dNi/dxi * yi)
        jac: np.VectorType = dslocal @  coords_edge
        jdet: float = float(np.linalg.norm(jac, ord=2))

        # 法線ベクトルの算出 (エッジに垂直な方向)
        # 2Dの外向き法線ベクトル n = (dy/dxi, -dx/dxi) / detJ
        norm_vector: np.VectorType = np.zeros(2)
        if direction is None:
            norm_vector[0] = -jac[1] / jdet
            norm_vector[1] = jac[0] / jdet
        else:
            norm_vector[0] = direction[0]
            norm_vector[1] = direction[1]
        
        if thickness < 0.0:
            # 軸対称要素は板厚が円周の長さになる
            radius: float = slocal.dot( coords_edge[:, 0])
            thick = 2.0 * math.pi * radius

        # 外力の加算
        # 各節点jに対して f_x = N_j * p * nx, f_y = N_j * p * ny
        for iedge in range(nedge):
            ied: int = ndim * iedge
            ret[ied:ied+ndim] += weight * slocal[iedge] * (value * norm_vector[0:ndim]) * jdet * thick
    return ret


def calc_element_2d_full(
        coords: np.MatrixType, dee: np.MatrixType,
        thickness: float,
        ans_elem: np.VectorType | None = None
    ) -> np.MatrixType | list[np.VectorType]:

    nnode: int = coords.shape[0]
    kstiff: np.MatrixType = np.zeros((nnode*2, nnode*2))
    stress_ip: list[np.VectorType] = []

    thick: float = thickness
    radius: float = 0.0

    # Gauss積分点ごとにループを回す
    for (xi, eta, weight) in _gauss_info_2d(nnode, 0):

        # 形状関数 Ni の取得 (shape=(nnode, ))
        slocal: np.VectorType = _shape_func_2d(nnode, xi, eta)

        # 形状関数の自然座標微分 dN/dxi, dN/deta の取得 (shape=(2, nnode))
        #   dslocal[0,i] = dN_i/dxi
        #   dslocal[1,i] = dN_i/deta
        dslocal: np.MatrixType = _shape_derivs_2d(nnode, xi, eta)

        # ヤコビ行列 [J] (shape=(2, 2))
        jac: np.MatrixType = dslocal @ coords
        jdet: float = np.linalg.det(jac)
        # 行列式が 0 以下: 節点の並びが時計回り (逆向き) か、つぶれた・過度に歪んだ要素
        if not jdet > 0.0:
            raise RuntimeError("Non-positive Jacobian determinant (detJ = {:.4e}) is detected. ".format(jdet)
                + "Check that the element nodes are ordered counterclockwise and the element is not distorted.")
        jinv = np.linalg.inv(jac)

        # 形状関数の物理座標微分 dN/dx, dN/dy の計算 (size: 2 x NNODE)
        dsglobal: np.MatrixType = jinv @ dslocal

        if thickness < 0.0:
            # 積分点における半径 r の計算
            # 節点座標の r 成分を形状関数で補間することで計算できる
            radius = slocal.dot(coords[:, 0])
            if not radius > 0.0:
                raise RuntimeError("Non-positive radial coordinate (r = {:.4e}) is detected at an integration point. ".format(radius)
                    + "Axisymmetric elements must be placed in the region x > 0.")

            # thickness = (volume / area)
            # 軸対称要素の場合はドーナツ型であることを考慮する
            thick = 2.0 * math.pi * radius

        # Bマトリクスの組み立て (size: 3 x (NNODE*2))
        bee: np.MatrixType = np.zeros((4, nnode*2))
        for i in range(nnode):
            i0: int = 2 * i + 0
            i1: int = i0 + 1
            bee[0, i0] = dsglobal[0, i]
            bee[1, i1] = dsglobal[1, i]
            if thickness < 0.0:
                bee[2, i0] = slocal[i] / radius
            bee[3, i0] = dsglobal[1, i]
            bee[3, i1] = dsglobal[0, i]


        # 要素剛性行列が与えられている場合は計算
        if ans_elem is None:
            # 要素剛性行列の計算
            kstiff += bee.transpose() @ dee @ bee * (jdet * thick * weight)
        else:
            stress: np.VectorType = dee @ bee @ ans_elem
            stress_ip.append(stress)

    if ans_elem is None:
        return kstiff
    return stress_ip


# def calc_edge_axial2d(
#         coords_edge: np.MatrixType,
#         value: float, direction: tuple[float, float] | None = None,
#         ) -> np.VectorType:

#     # 等価節点外力ベクトル
#     nedge: int = coords_edge.shape[0]
#     ndim: int = coords_edge.shape[1]
#     ret: np.VectorType = np.zeros(nedge * 2)

#     # Gauss積分点ごとにループを回す
#     for xi, weight in _gauss_info_1d(nedge):

#         # 1次元形状関数 (shape=(nedge, ))
#         slocal: np.VectorType = _shape_func_1d(nedge, xi)

#         # 形状関数の自然座標微分 dN/dxi
#         dslocal: np.VectorType = _shape_derivs_1d(nedge, xi)

#         # ヤコビアンの計算 (エッジの接線ベクトル)
#         # dx/dxi = Σ (dNi/dxi * xi), dy/dxi = Σ (dNi/dxi * yi)
#         jac: np.VectorType = dslocal @  coords_edge
#         jdet: float = float(np.linalg.norm(jac, ord=2))

#         # 法線ベクトルの算出 (エッジに垂直な方向)
#         # 2Dの外向き法線ベクトル n = (dy/dxi, -dx/dxi) / detJ
#         norm_vector: np.VectorType = np.zeros(2)
#         if direction is None:
#             norm_vector[0] = -jac[1] / jdet
#             norm_vector[1] = jac[0] / jdet
#         else:
#             norm_vector[0] = direction[0]
#             norm_vector[1] = direction[1]

#         radius: float = slocal.dot( coords_edge[:, 0])
#         thickness = 2.0 * math.pi * radius

#         # 外力の加算
#         # 各節点jに対して f_x = N_j * p * nx, f_y = N_j * p * ny
#         for iedge in range(nedge):
#             ied: int = ndim * iedge
#             ret[ied:ied+ndim] += weight * slocal[iedge] * (value * norm_vector[0:ndim]) * jdet * thickness
#     return ret


def calc_element_plquad4r(
        coords: np.MatrixType, dee: np.MatrixType, thickness: float,
        epsilon_hg: float,
        ans_elem: np.VectorType | None = None
    ) -> np.MatrixType | list[np.VectorType]:

    NNODE: int = 4

    # 低減積分の積分点とウェイトの設定
    xi: float = 0.0
    eta: float = 0.0
    weight: float = 4.0

    # 形状関数の自然座標微分 dN/dxi, dN/deta の取得 (size: 2 x NNODE)
    #   dslocal[0,i] = dN_i/dxi
    #   dslocal[1,i] = dN_i/deta
    dslocal: np.MatrixType = _shape_derivs_2d(NNODE, xi, eta)

    # ヤコビ行列 [J] (size: 2 x 2)
    jac: np.MatrixType = dslocal @ coords
    jdet: float = np.linalg.det(jac)
    # 行列式が 0 以下: 節点の並びが時計回り (逆向き) か、つぶれた・過度に歪んだ要素
    if not jdet > 0.0:
        raise RuntimeError("Non-positive Jacobian determinant (detJ = {:.4e}) is detected. ".format(jdet)
            + "Check that the element nodes are ordered counterclockwise and the element is not distorted.")
    jinv = np.linalg.inv(jac)

    # 形状関数の物理座標微分 dN/dx, dN/dy の計算 (size: 2 x NNODE)
    dsglobal: np.MatrixType = jinv @ dslocal

    # Bマトリクスの組み立て (size: 4 x (NNODE*2))
    bee: np.MatrixType = np.zeros((4, NNODE*2))
    for i in range(NNODE):
        i0: int = 2 * i + 0
        i1: int = i0 + 1
        bee[0, i0] = dsglobal[0, i]
        bee[1, i1] = dsglobal[1, i]
        bee[3, i0] = dsglobal[1, i]
        bee[3, i1] = dsglobal[0, i]

    # 結果出力の場合はここで応力計算して完了
    if ans_elem is not None:
        stress: np.VectorType = dee @ bee @ ans_elem
        return [stress, ]

    # 以下、要素剛性行列の計算
    kstiff: np.MatrixType = bee.transpose() @ dee @ bee * (jdet * thickness * weight)

    # ------------------------------
    #  gamma-projection法による安定化行列の計算
    #     D. P. Flanagan and T. Belytschko,
    #     A uniform strain hexahedron and quadrilateral with orthogonal hourglass control,
    #     International Journal for Numerical Methods in Engineering, 17(5), 679–706, 1981.
    #  ------------------------------
    # アワーグラスの安定化係数 epsilon_hg
    if epsilon_hg > 1.0e-8:
        # アワーグラスベクトル hvec (shape = (nnode, ))
        # 安定化ベクトル gamma (shape = (nnode, ) の計算 (無次元)
        hvec: np.VectorType = np.array([1.0, -1.0, 1.0, -1.0])
        gamma: np.VectorType = hvec
        gamma -= hvec.dot(coords[:, 0]) * dsglobal[0, :]
        gamma -= hvec.dot(coords[:, 1]) * dsglobal[1, :]

        # アワーグラス剛性の係数
        # せん断弾性係数 G = E / 2(1+nu)
        shear_modulus: float = dee[3, 3]

        # kappa = epsilon_hg * G * (volume / area)
        kappa: float = epsilon_hg * shear_modulus * thickness

        # アワーグラス剛性行列 (8x8) を要素剛性行列に加算
        for i in range(NNODE):
            i0: int = 2 * i + 0
            i1: int = i0 + 1
            for j in range(NNODE):
                j0: int = 2 * j + 0
                j1: int = j0 + 1
                # x方向自由度(2i-1, 2j-1)とy方向自由度(2i, 2j)に加算
                term: float = kappa * gamma[i] * gamma[j]
                kstiff[i0, j0] += term
                kstiff[i1, j1] += term
    return kstiff


def _calc_element_plquad4i_raw(
        coords: np.MatrixType, dee: np.MatrixType, thickness: float,
        ans_elem: np.VectorType | None = None
    ) -> np.MatrixType | list[np.VectorType]:

    kstiff: np.MatrixType = np.zeros((12, 12))
    stress_ip: list[np.VectorType] = []

    # 要素中心におけるヤコビ行列 [J0] (サイズ: 2 x 2)
    dslocal_center: np.MatrixType = _shape_derivs_2d(4, 0.0, 0.0)
    jac0: np.MatrixType = dslocal_center @ coords
    jdet0: float = np.linalg.det(jac0)
    # 行列式が 0 以下: 節点の並びが時計回り (逆向き) か、つぶれた・過度に歪んだ要素
    if not jdet0 > 0.0:
        raise RuntimeError("Non-positive Jacobian determinant (detJ = {:.4e}) is detected. ".format(jdet0)
            + "Check that the element nodes are ordered counterclockwise and the element is not distorted.")
    jinv0 = np.linalg.inv(jac0)

    # (Gauss積分点ごとにループを回す)
    for (xi, eta, weight) in _gauss_info_2d(4, 0):

        # 形状関数の自然座標微分 dN/dxi, dN/deta の取得
        #   dslocal[1:2, 1:4] ... 形状関数の微分
        #   dslocal_nc[1:2, 1:2] ... 非適合部分の微分
        dslocal: np.MatrixType = _shape_derivs_2d(4, xi, eta)
        dslocal_nc: np.MatrixType = np.array([[-2.0 * xi, 0.0], [0.0, -2.0 * eta]])

        # ヤコビ行列 [J] (サイズ: 2 x 2)
        jac: np.MatrixType = dslocal @ coords
        jdet: float = np.linalg.det(jac)
        # 行列式が 0 以下: 節点の並びが時計回り (逆向き) か、つぶれた・過度に歪んだ要素
        if not jdet > 0.0:
            raise RuntimeError("Non-positive Jacobian determinant (detJ = {:.4e}) is detected. ".format(jdet)
                + "Check that the element nodes are ordered counterclockwise and the element is not distorted.")
        jinv = np.linalg.inv(jac)

        # 形状関数の物理座標微分 dN/dx, dN/dy の計算
        #   dNmat[1:2, 1:4] ... 形状関数の微分 (サイズ: 2 x NNODE)
        #   dNmat[1:2, 5:6] ... 非適合部分の微分 (サイズ: 2 x 2)
        # 非適合部分は要素中心のヤコビアンを使って計算 (Taylor's modification)
        # Bマトリクスの補正項 (detJ0/detJ) もここで先にかけておく
        dsglobal: np.MatrixType = jinv @ dslocal
        dsglobal_nc: np.MatrixType = (jdet0 / jdet) * jinv0 @ dslocal_nc

        # Bマトリクスの組み立て (shape=(3, (NNODE+2)*2))
        bee: np.MatrixType = np.zeros((4, 12))
        for i in range(4):
            i0: int = 2 * i + 0
            i1: int = i0 + 1
            bee[0, i0] = dsglobal[0, i]
            bee[1, i1] = dsglobal[1, i]
            bee[3, i0] = dsglobal[1, i]
            bee[3, i1] = dsglobal[0, i]
        for i in range(2):
            i0: int = 2 * i + 8
            i1: int = i0 + 1
            bee[0, i0] = dsglobal_nc[0, i]
            bee[1, i1] = dsglobal_nc[1, i]
            bee[3, i0] = dsglobal_nc[1, i]
            bee[3, i1] = dsglobal_nc[0, i]

        # 要素剛性行列が与えられている場合は計算
        if ans_elem is None:
            kstiff += bee.transpose() @ dee @ bee * (jdet * thickness * weight)
        else:
            stress: np.VectorType = dee @ bee @ ans_elem
            stress_ip.append(stress)

    if ans_elem is None:
        return kstiff
    return stress_ip


def calc_element_plquad4i(
        coords: np.MatrixType, dee: np.MatrixType,
        thickness: float,
        ans_elem: np.VectorType | None = None
    ) -> np.MatrixType | list[np.VectorType]:

    # 縮退前の12x12行列を構築する
    kstiff_raw = _calc_element_plquad4i_raw(coords, dee, thickness)
    assert isinstance(kstiff_raw, np.ndarray)

    # 内部自由度と外部自由度に行列を分割
    # kstiff = [kee kei][ue] = [fe]
    #          [kie kii][ui]   [ 0]
    kee: np.MatrixType = kstiff_raw[0:8, 0:8]
    kei: np.MatrixType = kstiff_raw[0:8, 8:12]
    kie: np.MatrixType = kstiff_raw[8:12, 0:8]
    kii_inv = np.linalg.inv(kstiff_raw[8:12, 8:12])

    # 要素剛性行列が与えられている場合は
    # 静的縮退 (Static Condensation) を計算して返す
    if ans_elem is None:
        kstiff: np.MatrixType = kee - kei @ kii_inv @ kie
        return kstiff

    # 右辺値ベクトルが与えられている場合は結果表示
    # 内部自由度を復元し、内部自由度を含めた変位ベクトルを構築
    ans_total: np.VectorType = np.zeros(12)
    ans_total[0:8] = ans_elem
    ans_total[8:12] = -kii_inv @ kie @ ans_elem

    # 応力値の出力を行う
    return _calc_element_plquad4i_raw(coords, dee, thickness, ans_total)


