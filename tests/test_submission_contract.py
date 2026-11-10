from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_formal_runner_uses_fixed_infinilm_configuration():
    source = (ROOT / "scripts" / "run_fl2va.sh").read_text(encoding="utf-8")

    assert '--text="A dog falls asleep"' in source
    assert "--gpus=0,1,2,3,4,5,6,7" in source
    assert "--num-frames=124" in source
    assert "--steps=50" in source
    assert "--seed=42" in source
    assert "--height=384 --width=576" in source
    assert "--legacy" not in source


def test_python_entry_keeps_legacy_reference_option():
    source = (ROOT / "InfiniLM" / "examples" / "minimax_h3.py").read_text(
        encoding="utf-8"
    )

    assert '"--legacy"' in source


def test_h3_legacy_swiglu_weights_are_reordered_from_gate_value_to_value_gate():
    legacy_source = (ROOT / "InfiniLM" / "examples" / "minimax_h3.py").read_text(
        encoding="utf-8"
    )
    native_source = (ROOT / "InfiniLM" / "python" / "infinilm" / "modeling_utils.py").read_text(
        encoding="utf-8"
    )

    assert "gate, value = tensor.chunk(2, dim=0)" in legacy_source
    assert "torch.cat((value, gate), dim=0).contiguous()" in legacy_source
    assert "gate, up = tensor.chunk(2, dim=0)" in native_source
    assert 'remapped[prefix + "up_proj.weight"] = up' in native_source
    assert 'remapped[prefix + "gate_proj.weight"] = gate' in native_source


def test_h3_native_swiglu_preserves_checkpoint_up_gate_order():
    source = (
        ROOT
        / "InfiniLM"
        / "csrc"
        / "models"
        / "minimax_h3"
        / "minimax_h3_transformer.cpp"
    ).read_text(encoding="utf-8")

    assert '"net.0.proj.up_proj",\n        "net.0.proj.gate_proj"' in source
    assert "auto [up, gate] = gate_up_proj_->forward_split(hidden_states_mutable);" in source
    assert "infinicore::op::swiglu(up, gate)" in source


def test_native_forward_does_not_repeat_a_full_transformer_pass():
    source = (
        ROOT
        / "InfiniLM"
        / "python"
        / "infinilm"
        / "multimodal"
        / "minimax_h3.py"
    ).read_text(encoding="utf-8")

    assert source.count("self.engine.forward_raw(None, **native_inputs)") == 1
    assert "_first_forward_warmup_complete" not in source


def test_h3_conditioner_waits_for_visual_handoff_before_embedding_lifetime_ends():
    source = (
        ROOT
        / "InfiniLM"
        / "csrc"
        / "models"
        / "qwen3_vl"
        / "qwen3_vl_for_conditional_generation.cpp"
    ).read_text(encoding="utf-8")

    assert "vision_output.pooler_output" in source
    assert "infinicore::context::syncStream();" in source
    assert source.index("infinicore::context::syncStream();") > source.index(
        "visual placeholder count does not match visual output"
    )


def test_hygon_vision_rope_position_ids_wait_for_async_host_copy():
    source = (
        ROOT
        / "InfiniLM"
        / "csrc"
        / "models"
        / "qwen3_5"
        / "qwen3_5_vision.cpp"
    ).read_text(encoding="utf-8")

    assert "position_ids_cpu->to(image_grid_thw->device())" in source
    assert "infinicore::context::syncStream();" in source
    assert source.index("infinicore::context::syncStream();") < source.index(
        "return position_ids_device;"
    )


def test_fl2va_rejects_pipeline_frame_count_mismatch():
    source = (ROOT / "InfiniLM" / "examples" / "minimax_h3.py").read_text(
        encoding="utf-8"
    )

    assert "FL2VA frame-count mismatch" in source
    assert 'actual_frames != expected_frames' in source


def test_gfx936_build_uses_swiglu_scoped_ninetoothed_option():
    source = (ROOT / "scripts" / "build_infinicore.sh").read_text(
        encoding="utf-8"
    )

    assert "--hygon-ninetoothed-swiglu=y" in source
    assert "--ninetoothed=y" not in source


def test_infinilm_build_configures_isolated_dir_and_existing_target():
    source = (ROOT / "scripts" / "build_infinilm.sh").read_text(
        encoding="utf-8"
    )

    assert 'source_env="${SETUP_ENV:-$ROOT/scripts/setup_env.sh}"' in source
    assert 'source "$source_env"' in source
    assert 'xmake f -y -cv -m release --builddir="$XMAKE_BUILDDIR"' in source
    assert 'xmake build -j "$XMAKE_BUILD_JOBS" _infinilm' in source
    assert "infinicore_infer" not in source


def test_batch_manifest_uses_relative_frame_names_and_configurable_data_root():
    manifest = (ROOT / "validation" / "gfx936-two-case-smoke.tsv").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "scripts" / "run_fl2va_batch.sh").read_text(
        encoding="utf-8"
    )

    assert ("/" + "code/") not in manifest
    assert 'DATA_ROOT="${DATA_ROOT:-$ROOT/data}"' in runner
    assert '*) resolved_first_frame="$DATA_ROOT/$first_frame" ;;' in runner


def test_hygon_environment_paths_are_discovered_or_configured():
    source = (ROOT / "scripts" / "setup_env.sh").read_text(encoding="utf-8")

    assert 'source "$DTK_ROOT/env.sh"' in source
    assert '"$DTK_ROOT/.hyhal/rocm_smi"' in source
    for path_fragment in ("opt/dtk", "usr/local/lib/python3.11", "dev/null"):
        assert ("/" + path_fragment) not in source


def test_flash_attention_extensions_use_environment_paths_not_python_install_paths():
    hygon_source = (
        ROOT
        / "InfiniCore"
        / "src"
        / "infinicore"
        / "adaptor"
        / "flash_attn"
        / "hygon"
        / "flash_attn_hygon.cc"
    ).read_text(encoding="utf-8")
    iluvatar_source = (
        ROOT
        / "InfiniCore"
        / "src"
        / "infinicore"
        / "adaptor"
        / "iluvatar_vendor_adaptor.cc"
    ).read_text(encoding="utf-8")

    assert 'std::getenv("FLASH_ATTN_2_CUDA_SO")' in hygon_source
    assert 'std::getenv("INFINICORE_ILUVATAR_VENDOR_SO")' in iluvatar_source
    assert 'std::getenv("INFINICORE_IXTRITURBO_OPS_SO")' in iluvatar_source
    assert ("/" + "usr/local/") not in hygon_source + iluvatar_source


def test_chunked_attention_helper_is_legacy_only():
    source = (ROOT / "scripts" / "run_legacy_math_reference.py").read_text(
        encoding="utf-8"
    )

    assert '"--legacy" not in generator_args' in source
    assert "F.scaled_dot_product_attention = torch_chunked_sdpa" in source


def test_ninetoothed_aot_keeps_generic_stride_and_overflow_fallbacks():
    source = (
        ROOT
        / "InfiniCore"
        / "src"
        / "infiniop"
        / "ninetoothed"
        / "build.py"
    ).read_text(encoding="utf-8")

    assert "def compact_aot_variants" in source
    assert "ninetoothed.dtype.int32" in source
    assert "ninetoothed.dtype.int64" in source
    assert 'line.lstrip().startswith("#include ")' in source
    assert '"hipCtxGetId(nullptr, &ctx_id);"' in source


def test_batch_runner_uses_formal_infinilm_path_and_preserves_case_logs():
    source = (ROOT / "scripts" / "run_fl2va_batch.sh").read_text(
        encoding="utf-8"
    )

    assert "--gpus=0,1,2,3,4,5,6,7" in source
    assert '--steps="$STEPS"' in source
    assert '--num-frames="$NUM_FRAMES"' in source
    assert "summary.tsv" in source
    assert "tee \"$log\"" in source
    assert "--legacy" not in source
