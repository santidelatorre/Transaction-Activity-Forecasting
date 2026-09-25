def test_stream_identity_package_imports() -> None:
    import ubs_recurrence
    from ubs_recurrence import data, official

    assert ubs_recurrence.__doc__
    assert official.LABELS == tuple(data.LABELS)
