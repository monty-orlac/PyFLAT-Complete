# **********************************************************
#  メタデータの定義
# **********************************************************

# ------------------------------
#  アプリケーションの基本情報
# ------------------------------
__title__: str       = "PyFLAT"
__version__: str     = "1.0.0"
__description__: str = "2D simple structural analysis solver using the Finite Element Method for educational purposes."

# ------------------------------
#  開発者・ライセンス情報
# ------------------------------
__author__: str      = "Monty Orlac"
__email__: str       = "289912596+monty-orlac@users.noreply.github.com"
__copyright__: str   = "Copyright (C) 2026 " + __author__
__license__: str     = "MIT"
__license_display__: str = """\
This program is distributed under the MIT license.
  https://opensource.org/licenses/mit-license.php
THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND.
"""

# ------------------------------
#  動作環境
# ------------------------------
# 実行に必要な Python の最小バージョン
__requires_python__: tuple[int, int] = (3, 12)

# 実行に必要な各種モジュールの情報を下記フォーマットで指定
#   (<モジュール名>, <必要バージョン>)
__dependencies__: list[tuple[str, tuple[int, int, int]]] = [
    ("numpy", (2,  0,  0)),
    ("scipy", (1, 13,  0)),
    ("numba", (0, 59,  0)),
]


# **********************************************************************
# **********************************************************************
# **
# **   メタデータに関連したメソッドの定義
# **
# **********************************************************************
# **********************************************************************
import sys
import typing
import importlib.metadata


# **********************************************************
#  Python 環境のバージョン確認
# **********************************************************
def check_python_version() -> None:
    """ 実行中の Python 環境のバージョンを確認
    
    実行中の Python 環境として sys.version_info でバージョン情報を
    取得し、上記メタデータで指定したバージョン __requires_python__ 以上で
    あることを確認する。
    """
    if sys.version_info < __requires_python__:
        req_version_str: str = ".".join([str(s) for s in __requires_python__])
        mod_version_str: str = ".".join([str(s) for s in sys.version_info[0:3]])
        sys.stderr.write(f"FATAL ERROR: Python {req_version_str:s}+ is required (running on {mod_version_str:s})\n")
        sys.exit(1)
    return


# **********************************************************
#  依存パッケージのバージョン確認
# **********************************************************
def check_package_versions() -> None:
    """ 各種 Python パッケージのバージョンを確認
    
    実行中の Python 環境にインストールされている各種パッケージのうち、
    本プログラムが必要としているもの (__requires_python__にて定義) について
    指定したバージョンを満たしていることを順番に確認する
    """

    # 検出されたエラーの数をカウント
    nerror = 0

    for name, req_version in __dependencies__:
        try:
            # パッケージ name のバージョン情報文字列を取得
            try:
                mod_version_str: str = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                # 実行中の Python 環境内に name がインストールされていない場合、
                # エラーメッセージを表示してプログラム終了
                req_version_str: str = ".".join([str(s) for s in req_version])
                sys.stderr.write(f"ERROR: Package {name:s} is required but not installed.\n")
                sys.stderr.write(f"  {name:s} ver.{req_version_str:s} or higher is required.\n")
                sys.stderr.write("\n")

                # 出力したエラーの数をインクリメントして、次のモジュールの確認へ進む
                nerror += 1
                continue

            # 取得したパッケージ name のバージョン情報文字列を解析し、
            # major/minor バージョン番号を取得
            vsmajor, _, vstr = mod_version_str.partition(".")
            vsminor, _, vstr = vstr.partition(".")
            vmajor = int(vsmajor if vsmajor else 0)
            vminor = int(vsminor if vsminor else 0)

            # 取得したパッケージが __dependencies__ に指定されたバージョン
            # 以上かどうかを判定し、満たしていなければエラー処理
            if (vmajor, vminor) < req_version:
                raise RuntimeError(f"Installed package <{name:s}> version ({mod_version_str:s}) is outdated.")

        except RuntimeError as ex:
            # エラーメッセージを表示
            req_version_str: str = ".".join([str(s) for s in req_version])
            sys.stderr.write(f"ERROR: {ex!s}\n")
            sys.stderr.write(f"  {name:s} ver.{req_version_str:s} or higher is required.\n")
            sys.stderr.write("\n")

            # 出力したエラーの数をインクリメントして、次のモジュールの確認へ進む
            nerror += 1

    # 1つ以上エラーが検出されていたらここでエラー終了する
    if nerror > 0:
        sys.stderr.write("FATAL ERROR: Failed to pass the required packages version checks.\n")
        sys.stderr.write("\n")
        sys.exit(1)
    return


# **********************************************************
#  タイトルブロックの表示
# **********************************************************
def show_title(fp: typing.TextIO, is_validate_mode: bool=False) -> None:

    # タイトルの表示
    fp.write("************************************************************\n")
    fp.write("************************************************************\n")
    fp.write("****                                                    ****\n")
    fp.write("**      PyFLAT - Python program for                       **\n")
    fp.write("**           Finite-element Linear Analysis Tutorial      **\n")
    fp.write("****                                                    ****\n")
    fp.write("************************************************************\n")
    fp.write("************************************************************\n")
    fp.write("\n")

    if not is_validate_mode:
        # 通常バージョンでの実行
        # バージョン情報とライセンス情報を表示
        version_str: str = ".".join([str(s) for s in sys.version_info[0:3]])
        fp.write(f"Version: {__version__:s}  (Running on Python.{version_str:s})\n")
        fp.write(f"{__copyright__}\n")
        fp.write("\n")
        for sline in __license_display__.splitlines():
            fp.write(sline + "\n")
    else:
        # validate モードの場合、バージョン番号やライセンス情報を表示しない
        fp.write("This program was started in validate mode.\n")
        fp.write("Skipping version, license, and execution time notation.\n")
    fp.write("\n")
    return


# **********************************************************
#  プログラムのバージョン情報の表示
# **********************************************************
def show_version(fp: typing.TextIO) -> None:
    fp.write(f"{__title__:s} ver.{__version__:s}\n")
    return
