import llm_gateway


def test_package_imports() -> None:
    assert llm_gateway.__version__ == "0.1.0"
