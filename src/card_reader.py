"""構造化テキストデータの読み込みモジュール

本モジュールは構造化テキストファイルを読み込むためのパーサークラス
CardReader を提供する。構造化テキストファイルでは複数行から構成される
「ブロック」と、カンマ区切りの行データである「カード」を使うことができる。


# 構造化テキストデータ 入力フォーマット仕様

## 1. 基本的なファイル構成

ファイルは、複数の「ブロック」によって区切られ、
その中にデータ本体である「カード」が複数並ぶ階層構造を持っている。

下記は記述の例。コメント記号 (_COMMENT_MARK) を '//'、
ブロック終端ラベル (_LABEL_END_BLOCK) を 'END' であるとする。

[記述例]
  // ファイル全体に対するコメント

  NODE
    // ブロックの中のコメント
    NODE, 1, 10.0, 20.0, 30.0
    NODE, 2, 15.0, 25.0, 35.0   // 行の途中からのコメントも可
  END

  LOADCASE
    CASE
      PLOAD, 3, 2, 760.0
    END
    CASE
      DISP,  3, 2, 0.010
    END
  END


## 2. 共通ルール

* コメントと空行:
  コメント記号 (_COMMENT_MARK、例えば'//') 以降の記述はすべてコメントとして扱われ、
  パース時に自動的に無視される。コメントのみの行と空行は行自体が無視される。
  
* 大文字・小文字の区別:
  ブロックラベル、ブロック終端ラベル、カードラベルはいずれも大文字・小文字を区別しない。
  なお、カードのパラメータ (文字列型 `s` を含む) は変換されず、記述された通りに読み込まれる。
  ただし、プログラム側から渡すラベルは大文字で与えなければならない。
   (大文字であることを前提に、入力ファイル側だけを大文字化して比較している。)

* 空白のトリミング:
  カンマ（,）の前後にある不要な半角スペースやタブは自動的に除去される。


## 3. ブロックの仕様

特定の階層やデータグループを定義するために、ブロックでデータを囲むことができる。
* 開始宣言: ブロックの名前を表すラベルのみを単独で記述する。
* 終了宣言: ブロック終端ラベル (_LABEL_END_BLOCK、例えば'END') のみを単独で記述する。
* 入れ子: ブロックの中に別のブロック (子ブロック) を入れ子にできる
  (上記の記述例の LOADCASE と CASE)。

ブロックの開始宣言は特定のラベルが存在するかどうかのチェックしか行うことができない。
そのため、ブロックは決まった並びで並んでいることを前提とする。ただし、
親ブロックの中では、同じ名前の子ブロックが複数回 (回数は未定) 繰り返されることは
許容する (親ブロックの終端ラベルによって繰り返しの終わりを判定する)。


## 4. カードの仕様

ブロックの内部に記述するデータ本体。1行につき1データとして記述する。
* 基本フォーマット: 先頭の「カードラベル」から始まるカンマ区切り形式。
  [例] <カードラベル>, <パラメータ1>, <パラメータ2>, ...
* 行末カンマ: 行末がカンマで終わる場合（例: LABEL, 1, 2,）末尾のカンマは無視される。

カードを読み込む際にはカードのリストと各カードのパラメータ列の書式指定子を
与えて読み込む。そのため、リストにないカードが記述されている場合や、
カードのパラメータがフォーマットに従わない場合はエラーが送出される。

カードのデータ型は下記4種類のものがある。
* `s`: 文字列
* `i`: 符号付き整数
* `u`: 符号なし整数
* `f`: 浮動小数点
書式指定子ではこれを並べて表現する。また書式指定子に`/`を指定すると、
以降のパラメータは省略可能となる。例えばカード `LABEL` に対して
`if/uf/f` と指定した場合、`if`、`ifuf`、`ifuff` の3種類の指定に対応し、
* 成立例 (一部を省略): LABEL, 10, 1.5,
* 成立例 (すべて記述): LABEL, 10, 1.5, 20, 3.14, 250.0
* エラー (必須が不足): LABEL, 10
* エラー (グループの途中まで): LABEL, 10, 1.5, 20
* エラー (不正な型):   LABEL, 10, 1.5, 20.3, 3.14
`/` で区切られた各グループは「すべて記述する」か「すべて省略する」かのどちらかであり、
グループの途中までの記述はエラーとなる。省略されたパラメータに既定値は入らず、
CardData の該当するリストが短くなるだけである (何個記述されていたかは
len(card.integers) などで判定する)。


# エラー発生時の扱い

読み込み中のエラーはすべて例外として送出される。入力ファイルの誤りに起因する
例外は基本的に ParseGeneralError / ParseFormatError / ParseEOFError の
3種類でいずれも RuntimeError のサブクラスとなっているほか、
iter_cards() ではブロック内のエラーをまとめて RuntimeError として送出する。
例外が送出された後の CardReader の内部状態は規定しない。呼び出し側は例外を
受け取ったら、同じ CardReader を使って読み込みを続けないこと。
"""
import typing
import dataclasses
import collections.abc as cabc


_COMMENT_MARK: typing.Final[str] = "//"
"""行コメント記号。ファイル読み込みにおいてこの記号以降は無視される。"""

_LABEL_END_BLOCK: typing.Final[str] = "END"
"""ブロックの終端を示すラベル (大文字で定義すること。比較は大文字・小文字を区別しない)。"""

# **********************************************************
#  カスタムメッセージを出力するための RuntimeError の定義
# **********************************************************
class ParseBaseError(RuntimeError):
    """カスタムメッセージの基底クラス。中身は RuntimeError と同じ"""
    __slots__ = []


class ParseGeneralError(ParseBaseError):
    """一般的な読み込み中のエラーを表す例外 (ParseBaseErrorを継承)。

    定型メッセージを持った RuntimeError という位置づけであり、
    エラー送出先では RuntimeError として取り扱うことを想定している。
    エラー位置をプログラムから参照できるよう、引数を属性として保持する。

    Args:
        lineno (int): 読み込みエラーが起こった行の行番号
        message (str): エラーメッセージ本体

    Attributes:
        lineno (int): 読み込みエラーが起こった行の行番号 (1始まり)
    """
    __slots__ = ["lineno"]

    def __init__(self, lineno: int, message: str):
        msg: str = f"Parse error in line.{lineno:d}: {message:s}"
        super().__init__(msg)
        self.lineno: int = lineno
        return


class ParseEOFError(ParseBaseError):
    """読み込み中にファイル終端に到達したことを表す例外 (ParseBaseErrorを継承)。

    ブロックやカードを読み込もうとした時点でファイル終端に到達した場合に送出される。
    """
    __slots__ = []

    def __init__(self):
        super().__init__("Unexpected end of file is detected.")


class ParseFormatError(ParseBaseError):
    """カードの書式エラーに関する例外 (ParseBaseErrorを継承)

    定型メッセージを持った RuntimeError という位置づけであり、
    エラー送出先では RuntimeError として取り扱うことを想定している。
    エラー位置をプログラムから参照できるよう、引数を属性として保持する。

    Args:
        lineno (int): 読み込みエラーが起こった行の行番号
        label (str): 読み込み対象のカードのカードラベル
        message (str): エラーメッセージ本体

    Attributes:
        lineno (int): 読み込みエラーが起こった行の行番号 (1始まり)
        label (str): 読み込み対象のカードのカードラベル (大文字に正規化済み)
    """
    __slots__ = ["lineno", "label"]

    def __init__(self, lineno: int, label: str, message: str):
        msg: str = "Parse error while analyzing the line format " \
            f"of a card <{label:s}> on line.{lineno:d}: {message:s}"
        super().__init__(msg)
        self.lineno: int = lineno
        self.label: str = label
        return


# **********************************************************
#  カスタムファイル読み込みクラス
# **********************************************************
@dataclasses.dataclass(frozen=True)
class _LinePointer:
    """_LineReader の seek(), tell() でやり取りするファイルポインタ構造体"""
    lineno: int
    offset: int


class _LineReader:
    """行番号記録とコメントスキップを備えたファイル読み込みクラス。

    CardReader の内部で利用する、基本的な機能だけを備えたテキストリーダー。
    概ね Python のファイルオブジェクトと同じだが、
      - コメント記号以降を無視
      - 空行をスキップ (コメントだけの行もスキップ)
    という機能を備えている。また、CardReader でのデバッグ表示のため、
    1行を読み込むread()関数では行番号と行の内容を一緒に返す。
    ファイル終端に到達した場合、read() は ParseEOFError を送出する。

    Args:
        fp (TextIO): 読み込み対象のファイルオブジェクト
    """
    __slots__ = ["__fp", "__lineno"]

    def __init__(self, fp: typing.TextIO):
        self.__fp: typing.TextIO = fp
        self.__lineno: int = 1
        return

    def seek(self, pointer: _LinePointer | None = None):
        """ファイルポインタで指定した場所 (指定がなければファイル先頭) に移動する。"""
        if pointer:
            self.__fp.seek(pointer.offset)
            self.__lineno = pointer.lineno
        else:
            self.__fp.seek(0)
            self.__lineno = 1
        return

    def tell(self) -> _LinePointer:
        """現在の場所に対応するファイルポインタを返す。"""
        return _LinePointer(self.__lineno, self.__fp.tell())

    def read(self) -> tuple[str, int]:
        """ファイルから1行読み込み、行の内容と行番号の tuple を返す。

        改行やコメント記号以降をカットし、空行 (もしくはコメントだけの行) をスキップ。
        結果として改行コードなしで行の内容があるもの (空でない文字列) だけが返される。
        ファイル終端に到達した場合は ParseEOFError を送出する。
        このときエラーの行番号は「ファイルの最終行の行番号 + 1」となる。
        """
        sline: str = ""
        lineno: int = 0
        while True:
            lineno = self.__lineno
            sline = self.__fp.readline()
            if not sline:
                # EOFに達した場合 → 例外スロー
                raise ParseEOFError()

            # 実際に読み込んだので行番号をインクリメント
            self.__lineno += 1

            # 改行コード、コメント (_COMMENT_MARK 以降) を削除
            sline = sline.partition(_COMMENT_MARK)[0].rstrip()
            if sline:
                break
        return sline, lineno


@dataclasses.dataclass(frozen=True)
class CardData:
    """パースされたカード (行データ) の各トークンを型別に格納するデータクラス。

    Attributes:
        lineno (int): このカードが記述されていた入力ファイル上の行番号
        label (str): カードの識別ラベル名 (例: 'NODE', 'MTRUSS') 。
            入力ファイル上の表記によらず、大文字に正規化されている。
        integers (list[int]): 符号つき/なし整数のパラメータリスト。
        floats (list[float]): 浮動小数点数のパラメータリスト。
        strings (list[str]):  文字列パラメータのリスト。
    """
    lineno: int
    label: str
    integers: cabc.Sequence[int]
    floats: cabc.Sequence[float]
    strings: cabc.Sequence[str]

    def __len__(self) -> int:
        """int: カードに含まれる全パラメータ (型を問わない) の合計個数。"""
        return len(self.integers) + len(self.floats) + len(self.strings)

    def __str__(self) -> str:
        """デバッグおよび表示用のフォーマット済み1行文字列を生成。"""
        msg = f"{self.label:s}: ["
        msg += ", ".join([f"{v:s}" for v in self.strings]) + " / "
        msg += ", ".join([f"{v:d}" for v in self.integers]) + " / "
        msg += ", ".join([f"{v:f}" for v in self.floats]) + "]"
        return msg


class CardReader:
    """入れ子ブロックとカード書式指定により構造化されたファイル読み込みオブジェクト。

    ブロック読み込みは下記メソッドを利用できる。
    - try_enter_block(label):
        名前 label のブロックを検出する。ブロック終端子の場合は行を消費せず None を返す。
        それ以外はエラー。
    - enter_block(label):
        基本 try_enter_block() と一緒だが、ブロック終端子の場合もエラーを出す。
    - exit_block():
        ブロック終端子を検出し、現在のブロックを抜ける。
    - rewind():
        現在のブロックの先頭 (開始ラベル行の次の行) まで読み込み位置を戻す。
    開始ブロックの検出を行う enter_block(), try_enter_block() ではブロックラベルを
    指定して検出する必要があるため、ブロックの並びはあらかじめ決まっていなければならない。
    ただし、親ブロックの中では、try_enter_block() が親ブロックの終端子で None を
    返すことを利用して、同じ名前の子ブロックの不定回の繰り返しに対応できる
    (モジュールdocの使用例を参照)。

    カード読み込みは下記メソッドを利用できる。
    - try_read_card(cardlib):
        cardlibに登録されたカードの中の1つにマッチさせて読み込む。
        ブロック終端子の場合は行を消費せず None を返す。
        対応するカードがなければエラー。
    - read_card(cardlib):
        基本 try_read_card(cardlib) と一緒だが、ブロック終端子の場合もエラーを出す。
    - iter_cards(cardlib, rewind=False):
        ブロック終端が検出されるまでカードを読み込んで返し続け、
        最後にブロックを抜ける (rewind=True ならブロック先頭へ巻き戻す)。
    基本的にはブロック内に順不同に並ぶカードを iter_cards() で読み込むことを
    想定しているが、決まった並びのカードでも読み込み可能である。

    入力ファイル側のブロックラベル・カードラベルは大文字・小文字を区別しない。
    プログラム側から渡すブロックラベルや cardlib のキーは大文字であること
     (小文字を含んでいると、ラベルが一致しなかった時点で AssertionError になる)。

    エラーが起きた場合は例外を送出する。入力ファイルの誤りに起因するものは
    ParseGeneralError / ParseFormatError (いずれも RuntimeError のサブクラス)、
    iter_cards() の場合はそれらをまとめた RuntimeError である。例外の送出後は
    内部状態を規定しないので、同じ CardReader で読み込みを続けないこと。

    Args:
        fp (TextIO): 読み込み対象のファイルオブジェクト。基本は組み込み関数の
            open() で開かれたASCII文字のテキストファイルを想定しているが、
            seek()/tell()が可能な TextIO であれば何でもよく、
            おそらく io.StringIO にも対応するはずである。
    """
    __slots__ = ["__reader", "__blocks", "__pending_end"]

    def __init__(self, fp: typing.TextIO) -> None:
        self.__reader: _LineReader = _LineReader(fp)
        """読み込み対象のファイルオブジェクトに相当する _LineReader オブジェクト"""

        self.__blocks: list[tuple[str, _LinePointer]] = []
        """ 現在開いているブロックを管理するスタック。ブロックラベルと開始位置のポインタをまとめて管理。"""

        self.__pending_end: int = -1
        """ブロック終端を読み込んだかどうかを表すフラグ。

        値の符号により下記の意味を表す
          - 正の値 N    ... N 行目のブロック終端をすでに読み込み済み
          - -1 (0以下の任意の数) ... ブロック終端を読み込んでいない
        """
        return

    def __try_read_line(self, advance_end: bool=False) -> tuple[str | None, int]:
        """ブロック終端子でなければファイルから1行読み込む。

        処理は下記の4種類:
          (1) ブロック終端子でない場合、その行を消費して
              行の内容と行番号を返す
          (2) ブロック終端子が検出された場合、None と行番号を返す。
              引数 advance_end=True である場合のみ行を消費する
          (3) ファイル終端に到達した場合、_LineReader.read() が送出する
              ParseEOFError がそのまま伝わる
          (4) どのブロックにも入っていないのにブロック終端子が検出された場合、
              ParseGeneralError を送出する
        """
        sline: str | None
        lineno: int
        if self.__pending_end > 0:
            # 既にブロック終端に到達済みの場合は新規に行を読み込まない
            sline = None
            lineno = self.__pending_end
        else:
            # 次の行を読み込む
            # (ファイル終端に到達した場合は read() が例外を送出する → (3))
            sline, lineno = self.__reader.read()

            # ブロック終端子に一致する場合は sline=None とする
            sline = sline.strip()
            if sline.upper() == _LABEL_END_BLOCK:
                # ブロックの外でブロック終端子が見つかった → (4) 例外スロー
                if not self.__blocks:
                    raise ParseGeneralError(lineno,
                        f"Block end <{_LABEL_END_BLOCK:s}> found outside any block.")
                sline = None

        # 読み込んだ行がブロック終端子であった場合 → (2) (None, 行番号)
        if sline is None:
            self.__pending_end = -1 if advance_end else lineno
            return None, lineno

        # ブロック終端子ではない場合 → (1) (行の内容, 行番号)
        return sline, lineno

    def try_enter_block(self, label: str) -> int | None:
        """ラベルが label のブロック開始行を読み込む。

        label は大文字で与えること。入力ファイル側のブロック名は大文字・小文字を
        区別せずに比較する。ラベルが一致しなかった場合に label に小文字が
        含まれていると、プログラム側の誤りとして AssertionError を送出する。

        処理は下記の3種類:
          (1) 正常に読み込めた場合、その行を消費してブロックに入る。
              このとき行番号が int 型で返される。
          (2) ブロック終端子が検出された場合、その行を消費せず None を返す。
               (実際には行を消費した上で、終端子が検出されたことを
                self.__pending_end に記録する。)
          (3) 上記以外の場合は ParseGeneralError を送出する。
              次の行が別のラベルやカードだった場合 (その行は消費される) のほか、
              ファイル終端に到達した場合 (ParseEOFError) と、どのブロックにも
              入っていない位置でブロック終端子が見つかった場合もこれに含まれる。

        戻り値は行番号 (1以上) なので真偽値判定でも区別できるが、
        None との比較 (is not None) を推奨する。
        """
        # 次の行を読み込む (ブロック終端子の場合は行を読み進めない)
        sline_raw: str | None
        lineno: int
        sline_raw, lineno = self.__try_read_line(False)

        # ブロック終端子の場合 → (2) None
        if sline_raw is None:
            return None

        # 読み込んだ行がラベル label であった場合 → (1)行番号
        if sline_raw.upper() == label:
            # ブロックリストに読み込んだブロックを登録する
            self.__blocks.append((label, self.__reader.tell()))
            return lineno

        # 上記(1), (2)以外 → (3)例外スロー

        # label に小文字が含まれる場合はプログラム側のエラー
        #   → AssertionError が送出される
        assert label == label.upper(), f"Block label <{label:s}> has to be in upper case."

        # ブロックラベルが見つからなかったことを通常のエラー (ParseGeneralError) で送出
        slraw_trunc: str = sline_raw if len(sline_raw) < 16 else sline_raw[:13] + "..."
        raise ParseGeneralError(lineno,
            f"A block label <{label:s}> is expected, but not found."
            + f"(line image: '{slraw_trunc:s}'.)")

    def enter_block(self, label: str) -> int:
        """ラベルが label のブロック開始行を読み込む。

        label は大文字で与えること (try_enter_block() と同じ)。

        処理は下記の3種類:
          (1) 正常に読み込めた場合、その行を消費してブロックに入る。
              このとき行番号が int 型で返される。
          (2) ブロック終端子が検出された場合、ParseGeneralError を送出する
          (3) 上記以外の場合も try_enter_block() と同様に ParseGeneralError を送出する
        """
        # try_enter_block() と基本は同じ振る舞いだが、
        # (2) の場合だけ異なるので、(2)の例外スローをここで行う
        lineno: int | None = self.try_enter_block(label)
        if lineno is None:
            raise ParseGeneralError(self.__pending_end,
                f"Unexpected block end <{_LABEL_END_BLOCK:s}> found "
                + f"while parsing a block label <{label:s}>.")
        return lineno

    def exit_block(self) -> int:
        """ブロック終端子のラベル行を読み込み、現在のブロックを抜ける。

        処理は下記の2種類:
          (1) 正常に読み込めた場合、その行を消費してブロックを抜ける。
              このとき行番号が int 型で返される。
          (2) 上記以外の場合は ParseGeneralError を送出する。
              次の行が終端子でない場合 (その行は消費される) のほか、
              ファイル終端に到達した場合と、どのブロックにも入っていない場合も
              これに含まれる。
        """

        # 次の行を読み込む (ブロック終端子の場合も行を読み進める)
        sline_raw: str | None
        lineno: int
        sline_raw, lineno = self.__try_read_line(True)

        # 読み込んだ行がブロック終端子であった場合 → (1)行番号
        # (ブロックの外で終端子が見つかった場合は __try_read_line() が
        #  例外を送出するので、ここでは self.__blocks は空でない)
        if sline_raw is None:
            self.__blocks.pop()
            return lineno

        # 上記(1)以外 → (2)例外スロー
        slraw_trunc: str = sline_raw if len(sline_raw) < 16 else sline_raw[:13] + "..."
        raise ParseGeneralError(lineno,
            f"A block end <{_LABEL_END_BLOCK:s}> is expected, but not found."
            + f"(line image: '{slraw_trunc:s}'.)")

    def try_read_card(self, cardlib: dict[str, str]) -> CardData | None:
        """1行を読み込み、指定された定義に基づいて型変換を行う。

        処理は下記の3種類:
          (1) 指定されたカードのいずれかの場合、型変換済みのデータを
              CardData 型オブジェクトとして返す
          (2) ブロック終端子が検出された場合、その行を消費せず None を返す。
               (実際には行を消費した上で、終端子が検出されたことを
                self.__pending_end に記録する。)
          (3) 上記以外の場合、例外を送出する
              - 未知のカード名など: ParseGeneralError
              - ファイル終端への到達: ParseEOFError
              - パラメータの個数・型が書式指定子に合わない: ParseFormatError
              いずれの場合も、読み込んだ行は消費される。

        Args:
            cardlib (dict[str, str]): カードラベルをキー、型定義文字列 (例: 'usff', 'if/f') を値とする辞書。
                型文字の種類: 'i' -> int, 'u' -> unsigned int, 'f' -> float, 's' -> str。
                キーは大文字で与えること (入力ファイル側のカード名は大文字・小文字を
                区別せずに比較する)。一致するカードがなかった場合にキーに小文字が
                含まれていると、プログラム側の誤りとして AssertionError を送出する。
        """

        # 次の行を読み込む (ブロック終端子の場合は行を読み進めない)
        sline: str | None
        lineno: int
        sline, lineno = self.__try_read_line(False)

        # 読み込んだ行がブロック終端子であった場合 → (2)None
        # 実際の self.__reader 自体は行を消費するが、self.__pending_end に
        # 行番号を記録することで行を消費しなかったこととする
        if sline is None:
            return None

        # 読み込んだ行をパースする
        #   指定したフォーマットのカードがなかった場合 (3)
        #   返ってきた例外をそのまま上に伝える
        card: CardData = self.__parse_card(sline, cardlib, lineno)

        # 指定したフォーマットのカードがあった場合 → (1)カードを返す
        return card

    def read_card(self, cardlib: dict[str, str]) -> CardData:
        """1行を読み込み、指定された定義に基づいて型変換を行う。

        返り値は下記の3種類:
          (1) 指定されたカードのいずれかの場合、型変換済みのデータを
              CardData 型オブジェクトとして返す
          (2) ブロック終端子が検出された場合、ParseGeneralError を送出する
          (3) 上記以外の場合も try_read_card() と同様に
              ParseGeneralError / ParseFormatError を送出する

        Args:
            cardlib (dict[str, str]): try_read_card() と同じ。
        """
        # try_read_card() と基本は同じ振る舞いだが、
        # (2) の場合だけ異なるので、(2)の例外スローをここで行う
        card: CardData | None = self.try_read_card(cardlib)
        if card is None:
            raise ParseGeneralError(self.__pending_end,
                "Unexpected block end label found while parsing cards "
                + "in block <{:s}>.".format(self.__blocks[-1][0]))
        return card

    @classmethod
    def __parse_card(cls, sline: str, cardlib: dict[str, str], lineno: int) -> CardData:
        """与えられたカンマ区切り文字列を与えられた書式指定に沿って解析する。

        cardlib のキーは大文字であること。
        """

        # 行をカンマで分割してトークン列に変換
        # カンマで行が終わっていた場合、末尾の空文字列を削除
        tokens: list[str] = [s.strip() for s in sline.split(",")]
        if tokens and not tokens[-1]:
            tokens.pop()

        # カードラベル label を取得し (大文字に正規化)、cardlib に登録されたものと一致するか確認
        # 一致した場合、カードのフォーマットを initializer として取得
        label_raw: str = tokens[0]
        label: str = label_raw.upper()
        initializer: str | None = cardlib.get(label, None)
        if initializer is None:
            # 一致しなかった場合に限り、プログラム側の誤り (キーに小文字が含まれている) を確認
            assert all(key == key.upper() for key in cardlib), \
                "Card labels in cardlib have to be in upper case."

            # カードラベルが cardlib になかったので例外を返す
            label_trunc: str = label_raw if len(label_raw) < 16 else label_raw[:13] + "..."
            raise ParseGeneralError(lineno, f"Card label <{label_trunc:s}> is unknown.")

        # トークンごとに指定した型に変換する
        itoken: int = 1  # 次に消費するトークンの番号
        str_vals: list[str] = []
        int_vals: list[int] = []
        float_vals: list[float] = []
        for token_type in initializer:

            # オプションを表す境界線の場合、トークンが残っていなければパース終了
            # そうでない場合は解析続行
            if token_type == "/":
                if len(tokens) == itoken:
                    break
                continue

            # 書式指定子が残っているのにトークンを全て消費した → 例外を返す
            if itoken >= len(tokens):
                raise ParseFormatError(lineno, label,
                    f"Insufficient parameters for expected format <{initializer:s}>.")

            token: str = tokens[itoken]
            match token_type:
                case "s":  # トークンを文字列として解釈
                    str_vals.append(token)
                case "i" | "u":  # トークンを整数値に変換
                    try:
                        ival: int = int(token)
                    except ValueError:
                        raise ParseFormatError(lineno, label,
                            f"Failed to convert data #{itoken:03d} <{token:s}> to an integer value.")
                    if token_type == "u" and ival < 0:
                        raise ParseFormatError(lineno, label,
                            f"Data #{itoken:03d} <{token:s}> is negative while unsigned int type is specified.")
                    int_vals.append(ival)
                case "f":  # トークンを浮動小数点値に変換
                    try:
                        fval: float = float(token)
                    except ValueError:
                        raise ParseFormatError(lineno, label,
                            f"Failed to convert data #{itoken:03d} <{token:s}> to a floating-point value.")
                    float_vals.append(fval)
                case _:
                    # 不明な書式指定子が見つかった場合にデバッグエラーを返す
                    raise AssertionError(f"Unknown token type '{token_type:s}' for card <{label:s}>.")
            itoken += 1

        # 書式指定子を全て消費したのにトークンが残っている → 例外を返す
        if itoken != len(tokens):
            raise ParseFormatError(lineno, label,
                f"Too many parameters for expected format <{initializer:s}>.")

        # パース完了、CardData型オブジェクトして返す
        return CardData(lineno=lineno, label=label, integers=int_vals, floats=float_vals,
                        strings=str_vals)

    def rewind(self) -> None:
        """ファイルポインタを現在のブロック先頭 (orファイル先頭) まで巻き戻す。

        入れ子の場合は、一番内側の、いま開いているブロックの先頭
        (開始ラベル行の次の行) まで戻る。どのブロックにも入っていない場合は
        ファイル先頭まで戻る。ブロックのスタックは変化しない (ブロックは抜けない)。
        """

        # ブロック先頭 (orファイル先頭) まで巻き戻し
        if self.__blocks:
            self.__reader.seek(self.__blocks[-1][1])
        else:
            self.__reader.seek()

        # ENDブロックを保持している場合はその登録を削除
        self.__pending_end = -1
        return

    def iter_cards(self, cardlib: dict[str, str], rewind: bool=False) -> cabc.Iterator[CardData]:
        """ブロック終端子に到達するまでカードを解析し順番に返す (ジェネレータ)。

        ブロック終端子に到達すると、rewind=False (既定) の場合は exit_block() を
        呼んでブロックを抜け、rewind=True の場合は rewind() でブロック先頭へ巻き戻す
        (ブロックは抜けない)。したがって rewind=False で使った後に exit_block() を
        重ねて呼ばないこと。

        エラー処理:
          - カードの読み込みエラー (ParseGeneralError / ParseFormatError) が起きても
            その行を読み飛ばして読み込みを続け、ブロック終端子に到達した時点で、
            すべてのエラーメッセージを notes に持つ1つの RuntimeError をまとめて送出する。
            この場合はブロックを抜けず、巻き戻しも行わない。
          - ジェネレータなので、この RuntimeError が送出されるのは、正常に読めた
            カードをすべて yield し終わった後である。呼び出し側はエラーを受け取る前に
            一部のカードを処理済みになっている点に注意すること。
          - ブロックが閉じられないままファイル終端に到達した場合 (ParseEOFError) は、
            その時点で読み込みを打ち切り、そこまでのエラーとともに RuntimeError
            としてまとめて送出する。
          - RuntimeError 以外の例外 (書式指定子の誤りによる AssertionError など) は
            集約せず、そのまま送出する。

        注意:
          ブロックの終端・巻き戻しの処理はジェネレータを最後まで回したときに
          行われる。for ループを break などで途中で抜けると、読み込み位置が
          ブロックの途中に残ったままになるので、必ず最後まで回すこと。

        Args:
            cardlib (dict[str, str]): try_read_card() と同じ。
            rewind (bool): True の場合、ブロックを抜けずにブロック先頭へ巻き戻す。
                同じブロックを2回読む場合に使う。
        """
        errmsg: list[str] = []
        while True:
            try:
                card: CardData | None = self.try_read_card(cardlib)
            except ParseEOFError as ex:
                # ファイル終端に到達した場合はこれ以上読めないので終了
                # (読み込みを続けると同じ例外が繰り返され無限ループになる)
                errmsg.append(str(ex))
                break
            except RuntimeError as ex:
                errmsg.append(str(ex))
                continue
            if card is None:
                break
            yield card
        if errmsg:
            block_name: str = self.__blocks[-1][0] if self.__blocks else "(top level)"
            err: RuntimeError = RuntimeError(f"Following error(s) are detected in {block_name:s} block.")
            for msg in errmsg:
                err.add_note(msg)
            raise err
        if rewind:
            self.rewind()
        else:
            self.exit_block()
        return


if __name__ == "__main__":
    reader = CardReader(open("..\\tests\\model_patch_PEQUAD4.in", "r"))
    cardlib: dict[str, str] = {}

    print()
    print("NODE (1st parse):")
    reader.enter_block("NODE")
    cardlib = {"NODE": "iiiff"}
    for card in reader.iter_cards(cardlib, rewind=True):
        print(str(card))
    print()
    print("NODE (2nd parse):")
    for card in reader.iter_cards(cardlib):
        print(str(card))

    print()
    print("MATERIAL:")
    reader.enter_block("MATERIAL")
    cardlib = {"MTRUSS": "if", "M2DISO": "iff"}
    for card in reader.iter_cards(cardlib):
        print(str(card))

    print()
    print("PROPERTY:")
    reader.enter_block("PROPERTY")
    cardlib = {"PTRUSS": "iif", "P2D": "iif"}
    for card in reader.iter_cards(cardlib):
        print(str(card))

    print()
    print("ELEMENT:")
    reader.enter_block("ELEMENT")
    cardlib = {"TRUSS": "ii" + "i" * 2, "PEQUAD4": "ii" + "i" * 4}
    for card in reader.iter_cards(cardlib):
        print(str(card))

    print()
    print("LOADCASE:")
    reader.enter_block("LOADCASE")
    cardlib = {"PLOAD": "iif", "ELOAD": "iif/ff", "EPRESS": "iif/ff", "DISP": "iif"}
    while reader.try_enter_block("CASE") is not None:
        print()
        print("LOADCASE DATA:")
        for card in reader.iter_cards(cardlib):
            print(str(card))
    reader.exit_block()
