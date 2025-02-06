# from https://github.com/chengzeyi/ParaAttention/blob/main/examples/run_hunyuan_video.py
import functools
from typing import Any, Dict, Union, Optional, Tuple
import logging
import time

import torch

from diffusers import DiffusionPipeline, HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.models.transformers.transformer_hunyuan_video import (
    HunyuanVideoTransformerBlock,
    HunyuanVideoSingleTransformerBlock,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import scale_lora_layers, unscale_lora_layers, USE_PEFT_BACKEND
from diffusers.utils import export_to_video

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_runtime_state,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    is_dp_last_group,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)

from xfuser.model_executor.layers.attention_processor import xFuserHunyuanVideoAttnProcessor2_0
from xfuser.model_executor.layers.hunyuan_isp import modify_transformer_block, modify_single_transformer_block

assert xFuserHunyuanVideoAttnProcessor2_0 is not None


def parallelize_transformer(pipe: DiffusionPipeline):
    transformer = pipe.transformer

    for block in transformer.transformer_blocks:
        modify_transformer_block(block)
    
    for block in transformer.single_transformer_blocks:
        modify_single_transformer_block(block)


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)

    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."
    assert engine_args.use_parallel_vae is False, "parallel VAE not implemented for HunyuanVideo"

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        revision="refs/pr/18",
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        transformer=transformer,
        torch_dtype=torch.float16,
        revision="refs/pr/18",
    )

    initialize_runtime_state(pipe, engine_config)
    get_runtime_state().set_video_input_parameters(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        batch_size=1,
        num_inference_steps=input_config.num_inference_steps,
        split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
    )

    parallelize_transformer(pipe)

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)

    if args.enable_tiling:
        pipe.vae.enable_tiling(
            # Make it runnable on GPUs with 48GB memory
            # tile_sample_min_height=128,
            # tile_sample_stride_height=96,
            # tile_sample_min_width=128,
            # tile_sample_stride_width=96,
            # tile_sample_min_num_frames=32,
            # tile_sample_stride_num_frames=24,
        )

    if args.enable_slicing:
        pipe.vae.enable_slicing()

    parameter_peak_memory = torch.cuda.max_memory_allocated(
        device=f"cuda:{local_rank}")

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer,
                                         mode="max-autotune-no-cudagraphs")

        # one step to warmup the torch compiler
        output = pipe(
            height=input_config.height,
            width=input_config.width,
            num_frames=input_config.num_frames,
            prompt=input_config.prompt,
            num_inference_steps=1,
            generator=torch.Generator(device="cuda").manual_seed(
                input_config.seed),
        ).frames[0]

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(
            input_config.seed),
    ).frames[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if is_dp_last_group():
        resolution = f"{input_config.width}x{input_config.height}"
        output_filename = f"results/hunyuan_video_{parallel_info}_{resolution}.mp4"
        export_to_video(output, output_filename, fps=15)
        print(f"output saved to {output_filename}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9} GB"
        )
    get_runtime_state().destory_distributed_env()


# mkdir -p results && torchrun --nproc_per_node=2 examples/hunyuan_video_usp_example.py --model tencent/HunyuanVideo --ulysses_degree 2 --num_inference_steps 30 --warmup_steps 0 --prompt "A cat walks on the grass, realistic" --height 320 --width 512 --num_frames 61 --enable_tiling --enable_model_cpu_offload
# mkdir -p results && torchrun --nproc_per_node=2 examples/hunyuan_video_usp_example.py --model tencent/HunyuanVideo --ulysses_degree 2 --num_inference_steps 30 --warmup_steps 0 --prompt "A cat walks on the grass, realistic" --height 544 --width 960 --num_frames 129 --enable_tiling --enable_model_cpu_offload
if __name__ == "__main__":
    main()
