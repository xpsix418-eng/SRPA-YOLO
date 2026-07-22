def test_public_imports() -> None:
    import srpa_yolo
    from srpa_yolo.modules import RepSharedPrivateResidualDetect

    assert srpa_yolo.ROOT.is_dir()
    assert RepSharedPrivateResidualDetect.__name__ == "RepSharedPrivateResidualDetect"

