from ninetoothed.targets import resolve_target_context


def test_gfx936_hygon_triton_profile_allows_aot():
    target = resolve_target_context(
        "triton", platform="hygon-bw1000", compute_arch="gfx936"
    )

    target.validate_materialization("aot")
    assert target.compute_arch == "gfx936"
    assert target.platform.name == "hygon-bw1000"
