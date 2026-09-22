def test_package_imports() -> None:
    """El paquete principal debe poder importarse."""
    import transaction_forecasting

    assert transaction_forecasting.__version__
