import concurrent.futures
import functools
import inspect
import itertools
import os
import pathlib

import ninetoothed
import ninetoothed.aot as ninetoothed_aot
from ninetoothed.aot import _HEADER_PATH
from ninetoothed.utils import calculate_default_configs

CURRENT_FILE_PATH = pathlib.Path(__file__)

BUILD_DIRECTORY_PATH = (
    CURRENT_FILE_PATH.parent.parent.parent.parent / "build" / "ninetoothed"
)


def build(premake, constexpr_param_grid, caller, op_name, output_dir):
    _install_cpp_aot_compat()
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    BUILD_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)
    headers = []
    all_param_names = []
    combinations = []
    launches = []

    max_workers = max(1, int(os.environ.get("NINETOOTHED_BUILD_JOBS", "2")))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for combination in tuple(
            _generate_param_value_combinations(constexpr_param_grid)
        ):
            future = executor.submit(
                _make, premake, combination, caller, op_name, output_dir
            )

            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            header, param_names, combination, launch = future.result()

            headers.append(header)
            all_param_names.append(param_names)
            combinations.append(combination)
            launches.append(launch)

    includes = "\n".join(f'#include "{header}"' for header in headers)

    param_names = list(
        functools.reduce(
            lambda x, y: dict.fromkeys(x) | dict.fromkeys(y),
            sorted(all_param_names, key=len, reverse=True),
            {},
        )
    )
    param_types = [
        "NineToothedStream",
    ] + ["NineToothedTensor" for _ in range(len(param_names) - 1)]

    for param_name in functools.reduce(lambda x, y: x | y, combinations, {}):
        param_names.append(param_name)
        param_types.append("int")

    param_decls = ", ".join(
        f"{type} {param}" for param, type in zip(param_names, param_types)
    )

    has_hip_kernels = any(pathlib.Path(output_dir).glob(f"{op_name}_*.cpp"))
    source_file_name = f"{op_name}.cpp" if has_hip_kernels else f"{op_name}.c"
    header_file_name = f"{op_name}.h"

    func_sig = f"NineToothedResult launch_{op_name}({param_decls})"

    joined_launches = "\n".join(launches)

    op_decl = f'#ifdef __cplusplus\nextern "C" {func_sig};\n#else\n{func_sig};\n#endif'
    op_def = f"""{func_sig} {{
{joined_launches}
    return INFINI_STATUS_NOT_IMPLEMENTED;
}}"""

    source_content = f"""#include "{header_file_name}"

#include "infinicore.h"

{includes}\n\n{op_def}\n"""
    header_content = f"""#include "{_HEADER_PATH}"
\n{op_decl}\n"""

    (BUILD_DIRECTORY_PATH / source_file_name).write_text(source_content)
    (BUILD_DIRECTORY_PATH / header_file_name).write_text(header_content)


def _make(premake, combination, caller, op_name, output_dir):
    arrangement, application, tensors = premake(**combination)

    for param_name, param_value in combination.items():
        if isinstance(param_value, str):
            combination[param_name] = (
                f"INFINI_DTYPE_{combination[param_name].replace('fp', 'F').upper()}"
            )

    combination = {f"{name}_": value for name, value in combination.items()}

    kernel_name = f"{op_name}_{_generate_suffix(combination.values())}"

    ninetoothed.make(
        arrangement,
        application,
        tensors,
        caller=caller,
        kernel_name=kernel_name,
        output_dir=output_dir,
    )

    header = output_dir / f"{kernel_name}.h"
    param_names = ("stream",) + tuple(inspect.signature(application).parameters.keys())
    launch = f"""    if ({_generate_condition(combination)})
        return launch_{kernel_name}({", ".join(param_names)});"""

    return header, param_names, combination, launch


def _generate_condition(combination):
    return " && ".join(f"{param} == {value}" for param, value in combination.items())


def _generate_suffix(values):
    return "_".join(f"{value}" for value in values)


def _generate_param_value_combinations(param_grid):
    keys = list(param_grid.keys())
    value_combinations = itertools.product(*param_grid.values())

    return tuple(dict(zip(keys, combination)) for combination in value_combinations)
def _install_cpp_aot_compat():
    """Support Triton ROCm releases that emit C++ AOT source files."""
    if getattr(ninetoothed, "_infini_cpp_aot_compat", False):
        return

    compile_original = ninetoothed_aot._compile

    def compile_with_cpp_alias(*args, **kwargs):
        source_path = pathlib.Path(args[0] if args else kwargs["path"])
        original_source = source_path.read_text()
        compatible_source = original_source.replace(
            "triton.bfloat16", "triton.language.bfloat16"
        )
        if compatible_source != original_source:
            source_path.write_text(compatible_source)
        try:
            signature_hash, outputs = compile_original(*args, **kwargs)
        finally:
            if compatible_source != original_source:
                source_path.write_text(original_source)
        for name, content in tuple(outputs.items()):
            if name.endswith(".cpp"):
                outputs.setdefault(name[:-4] + ".c", content)
        return signature_hash, outputs

    def aot_with_cpp_output(func, caller="cuda", kernel_name=None,
                            output_dir=None, num_warps=None, num_stages=None):
        default_warps, default_stages = calculate_default_configs()
        num_warps = default_warps if num_warps is None else num_warps
        num_stages = default_stages if num_stages is None else num_stages
        output_dir = pathlib.Path(output_dir)
        outputs = ninetoothed_aot._aot(
            func, caller, kernel_name, num_warps, num_stages
        )
        has_hip_kernel = any(name.endswith(".cpp") for name in outputs)
        for name, content in outputs.items():
            output_name = pathlib.Path(name).name
            suffix = pathlib.Path(output_name).suffix
            if suffix == ".cpp":
                lines = content.splitlines()
                includes = [
                    line for line in lines if line.lstrip().startswith("#include ")
                ]
                body = [
                    line for line in lines if not line.lstrip().startswith("#include ")
                ]
                content = (
                    "#include <hip/hip_runtime.h>\n"
                    + "\n".join(includes)
                    + f"\n#include \"{kernel_name}.h\"\n"
                    + "#define warpSize 64\n"
                    + "\n".join(body).replace(
                        "cuCtxGetId(NULL, &ctx_id);",
                        "hipCtxGetId(nullptr, &ctx_id);",
                    )
                    + "\n"
                )
            elif suffix == ".c" and has_hip_kernel:
                output_name = pathlib.Path(output_name).with_suffix(".cpp").name
                content = (
                    f"#include <hip/hip_runtime.h>\n#define warpSize 64\n#include \"{kernel_name}.h\"\n" + content
                )
            (output_dir / output_name).write_text(content)

    def compact_aot_variants(launch_arg_names, tensors, find_tensor):
        _, tensor_ndims, _ = ninetoothed_aot._per_tensor_dim_options(
            launch_arg_names, tensors, find_tensor
        )
        common = ((), (), launch_arg_names, tensor_ndims)
        int32_variant = ninetoothed_aot._variant_suffix(
            *common,
            size_type=ninetoothed.dtype.int32,
            stride_type=ninetoothed.dtype.int32,
        )
        int64_variant = ninetoothed_aot._variant_suffix(
            *common,
            size_type=ninetoothed.dtype.int64,
            stride_type=ninetoothed.dtype.int64,
        )
        return [
            (
                int32_variant,
                (),
                (),
                ninetoothed.dtype.int32,
                ninetoothed.dtype.int32,
            ),
            (
                int64_variant,
                (),
                (),
                ninetoothed.dtype.int64,
                ninetoothed.dtype.int64,
            ),
        ]

    ninetoothed_aot._compile = compile_with_cpp_alias
    ninetoothed_aot._enumerate_variant_specs = compact_aot_variants
    ninetoothed.make.__globals__["aot"] = aot_with_cpp_output
    ninetoothed._infini_cpp_aot_compat = True
