import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


SM = [70]  # Sm80 kernels support up to
HEAD_DIMENSIONS = [32, 64, 96, 128, 192, 256]
IS_CAUSAL = ["false", "true"]
IS_PREFILL = ["false", "true"]
NAMESPACE_INCLUDE = '#include "namespace_config.h"\n'

def get_fwd_template() -> str:
    return NAMESPACE_INCLUDE + """#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {{

template<>
void run_mha_fwd_<{HEAD_DIM}, {IS_CAUSAL}, {IS_PREFILL}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_hdim{HEAD_DIM}<{IS_CAUSAL}, {IS_PREFILL}>(params, stream);
}}

}} // namespace FLASH_NAMESPACE"""

def get_fwd_split_template() -> str:
    return NAMESPACE_INCLUDE + """#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {{

template void run_mha_fwd_splitkv_dispatch<{HEAD_DIM}, {IS_CAUSAL}, {IS_PREFILL}>(Flash_fwd_params &params, cudaStream_t stream);

}} // namespace FLASH_NAMESPACE"""


@dataclass
class Kernel:
    sm: int
    head_dim: int
    is_causal: bool
    is_prefill: bool
    direction: str

    @property
    def template(self) -> str:
        template_funcs = {
            "fwd": get_fwd_template,
            "fwd_split": get_fwd_split_template,
        }
        template_func = template_funcs[self.direction]
        return template_func().format(
            HEAD_DIM=self.head_dim,
            IS_CAUSAL=self.is_causal,
            IS_PREFILL=self.is_prefill
        )

    @property
    def filename(self) -> str:
        return f"flash_{self.direction}_hdim{self.head_dim}_{'causal_' if self.is_causal == 'true' else ''}{'prefill_' if self.is_prefill == 'true' else ''}sm{self.sm}.cu"

def get_all_kernels() -> List[Kernel]:
    for direction in ["fwd", "fwd_split"]:
        for head_dim, is_causal, is_prefill, sm in itertools.product(HEAD_DIMENSIONS, IS_CAUSAL, IS_PREFILL, SM):
            yield Kernel(sm=sm, head_dim=head_dim, is_causal=is_causal, is_prefill=is_prefill, direction=direction)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n"""
    content = prelude + kernel.template
    (autogen_dir / kernel.filename).write_text(content)

def main(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the kernels "
        " will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)
