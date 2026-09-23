import metadata


def main() -> None:

    # Python環境のバージョン確認
    metadata.check_python_version()

    # 依存モジュールの存在/バージョン確認
    metadata.check_package_versions()

    # メインルーチンの実行開始
    import pyflat_main
    pyflat_main.main()

    return


if __name__ == "__main__":
    main()
