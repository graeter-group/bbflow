# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

from bbflow.deployment.bbflow import BBFlow
from argparse import ArgumentParser
from pathlib import Path
from tqdm.auto import tqdm

def _sample():
    arg_parser = ArgumentParser(description="Sample conformations of protein backbone structures using BBFlow.")

    arg_parser.add_argument(
        "--input_path",
        "-pdb",
        type=str,
        nargs='+',
        required=False,
        default=None,
        help="Path to the input PDB file(s) containing the protein backbone structure used as initial conformation. If given, the output_path argument must also be given and contain the same number of paths."
    )
    arg_parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        nargs='+',
        required=False,
        default=None,
        help="Path to the output PDB/xtc file(s) where the sampled conformations will be saved. File ending can either be .pdb or .xtc. If given, the input_path argument must also be given and contain the same number of paths."
    )
    arg_parser.add_argument(
        "--input_dir",
        "-d",
        type=str,
        required=False,
        default=None,
        help="Path to the input directory containing the PDB files used as initial conformations. If given, the output_dir argument must also be given. input_dir and input_path are mutually exclusive."
    )
    arg_parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        required=False,
        default=None,
        help="Path to the output directory where the sampled conformations will be saved. If given, the input_dir argument must also be given."
    )
    arg_parser.add_argument(
        "--output_format",
        "-fmt",
        type=str,
        required=False,
        default="pdb",
        choices=["pdb", "xtc"],
        help="Output format of the sampled conformations if the output_path argument is not being used. Can be either pdb or xtc. Default is pdb."
    )
    arg_parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        required=False,
        default=100,
        help="Number of conformations to sample. Default is 100."
    )
    arg_parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        required=False,
        default=None,
        help="Batch size for sampling. If not given, the batch size will be estimated based on the GPU memory. Default is None."
    )
    arg_parser.add_argument(
        "--cuda_memory_GB",
        "-vram",
        type=float,
        required=False,
        default=40,
        help="Amount of GPU memory in GB to use for estimating the batch size. If not given, the batch size will be estimated based on the GPU memory. Default is None."
    )
    arg_parser.add_argument(
        "--tag",
        "-t",
        type=str,
        required=False,
        default="latest",
        help="Tag of the model to use. Default is latest."
    )
    arg_parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Whether to overwrite the output files if they already exist. Default is False, i.e. to overwrite the files."
    )
    arg_parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device to use for sampling. Default is cuda."
    )
    arg_parser.add_argument(
        "--ckpt_path",
        type=str,
        required=False,
        default=None,
        help="Path to the checkpoint file. If given, the tag argument will be ignored."
    )
    arg_parser.add_argument(
        "--timesteps",
        type=int,
        required=False,
        default=20,
        help="Number of timesteps to use for sampling. Default is 20."
    )
    arg_parser.add_argument(
        "--hide_progbar",
        action="store_true",
        help="Hide the progress bar during sampling. Default is False, i.e. to show the progress bar."
    )
    arg_parser.add_argument(
        "--gamma_trans",
        type=float,
        required=False,
        default=None,
        help="Gamma parameter for the translational part. If not given, the parameter will be chosen in accordance to the training config. Default is None."
    )
    arg_parser.add_argument(
        "--gamma_rots",
        type=float,
        required=False,
        default=None,
        help="Gamma parameter for the rotational part. If not given, the parameter will be chosen in accordance to the training config. Default is None."
    )

    args = arg_parser.parse_args()

    # Check if the arguments are valid
    _check_arg_validity(args)

    if args.input_path is not None:
        input_paths = args.input_path
        output_paths = [Path(p) for p in args.output_path]
    elif args.input_dir is not None:
        input_paths = list(Path(args.input_dir).glob('*.pdb'))
        output_paths = [Path(args.output_dir) / f'samples_{Path(p).stem}.{args.output_format}' for p in input_paths]
    else:
        raise ValueError("Either input_path or input_dir must be given.") # redundant, but for clarity

    # create the parent dirs of the output paths if they do not exist
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    # load the sampler class:
    init_kwargs = {
        "timesteps": args.timesteps,
        "gamma_rots": args.gamma_rots,
        "gamma_trans": args.gamma_trans,
        "progress_bar": not args.hide_progbar,
    }

    sample_kwargs = {
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "cuda_memory_GB": args.cuda_memory_GB,
        "overwrite": not args.no_overwrite,
        "device": args.device,
    }

    if not args.hide_progbar:
        pbar = lambda x: tqdm(x, total=len(input_paths), desc="Processing PDB files", unit="file", dynamic_ncols=True, position=0)
        p_bar_inner = {'position':1, 'leave':False}
        init_kwargs['_pbar_kwargs'] = p_bar_inner
    else:
        pbar = lambda x: x

    if args.ckpt_path is not None:
        bbflow_sampler = BBFlow(ckpt_path=args.ckpt_path, **init_kwargs)
    else:
        bbflow_sampler = BBFlow.from_tag(args.tag, **init_kwargs)

    # sample conformations:
    print(f"Sampling {args.num_samples} confs for {len(input_paths)} input PDB files...\n")

    for input_path, output_path in pbar(zip(input_paths, output_paths)):
        bbflow_sampler.sample(
            input_path=input_path,
            output_path=output_path,
            output_fmt=output_path.suffix[1:],
            **sample_kwargs
        )


def _check_arg_validity(args):
    # Check if input_path and input_dir are given
    if args.input_path is None and args.input_dir is None:
        raise ValueError("Either input_path or input_dir must be given.")
    
    # Check if output_path and output_dir are given
    if args.output_path is None and args.output_dir is None:
        raise ValueError("Either output_path or output_dir must be given.")
    
    # Check that not both input_path and input_dir are given
    if args.input_path is not None and args.input_dir is not None:
        raise ValueError("input_path and input_dir are mutually exclusive. Please provide only one of them.")
    if args.output_path is not None and args.output_dir is not None:
        raise ValueError("output_path and output_dir are mutually exclusive. Please provide only one of them.")

    
    # Check if input_path and output_path are given
    if args.input_path is not None:
        assert args.output_path is not None, "If input_path is given, output_path must also be given."
        assert len(args.input_path) == len(args.output_path), f"If input_path is given, output_path must also be given and contain the same number of paths, {len(args.input_path)} != {len(args.output_path)}."
        # check that all paths exist:
        for path in args.input_path:
            assert Path(path).exists(), f"Input path {path} does not exist."
            # check that all paths are .pdb
            assert Path(path).suffix == '.pdb', f"Found input path {path}. It must have .pdb ending."

        for path in args.output_path:
            assert Path(path).suffix in ['.pdb', '.xtc'], f"Output path {path} must have .pdb or .xtc ending."

            # check that all paths are not the same as input paths
            assert Path(path) not in [Path(p) for p in args.input_path], f"Output path {path} must not be the same as input path."

    # Check if input_dir and output_dir are given
    if args.input_dir is not None:
        assert args.output_dir is not None, "If input_dir is given, output_dir must also be given."
        assert Path(args.input_dir).exists(), f"Input directory {args.input_dir} does not exist."
        # check that at least one pdb file exists in the input directory
        assert len(list(Path(args.input_dir).glob('*.pdb'))) > 0, f"No PDB files found in {args.input_dir}."