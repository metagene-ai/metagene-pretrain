# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
import os
import pprint
import time
import copy
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, List

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from typing_extensions import Literal

from litgpt import Tokenizer
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import DataModule, TinyLlama, NAO
from litgpt.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from litgpt.utils import (
    CLI,
    CycleIterator,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    num_parameters,
    parse_devices,
    reset_parameters,
    save_config,
    save_hyperparameters,
)
from litgpt.utils_metrics import ActivationNormMetric, get_grad_norm
from litgpt.loss import cross_entropy_max_z_loss

from torch.distributed.fsdp import  MixedPrecision


# Sanity check code
# TODO: eventually remove this
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

# TODO: may set chunked_cross_entropy with chunk_size=0


def setup(
    model_name: Optional[str] = None,
    model_config: Optional[Config] = None,
    out_dir: Path = Path("out/pretrain"),
    initial_checkpoint_dir: Optional[Path] = None,
    resume: Union[bool, Path] = False,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e12),  # 3 trillion
        learning_rate=4e-4,
        weight_decay=1e-1,
        beta1=0.9,
        beta2=0.95,
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    devices: Union[int, str] = "auto",
    tokenizer_dir: Optional[Path] = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "wandb",
    seed: int = 42,
    fsdp_strategy: str = "HYBRID_SHARD",
    context_stuffing: bool = True,
    attention_impl: Literal["sdpa", "fa", "xformers"] = "sdpa",
    fake_data: bool = False,
    shuffle_block_size: int = 50_000_000,
):
    """Pretrain a model.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Mutually exclusive with
            ``model_config``.
        model_config: A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
            ``model_config``.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize the model from.
            Useful for continued pretraining. Mutually exclusive with ``resume``.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyLlama``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        devices: How many devices/GPUs to use. Uses all GPUs by default.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
            module require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
    """
    hparams = locals()

    if 'genomics' in hparams['model_name']:
        data = NAO(context_stuffing=context_stuffing, fake_data=fake_data)
    elif data is None:
        data = TinyLlama()

    if model_config is not None and model_name is not None:
        raise ValueError("Only one of `model_name` or `model_config` can be set.")
    elif model_config is None and model_name is None:
        model_name = "tiny-llama-1.1b"
    config = Config.from_name(model_name) if model_config is None else model_config
    config.attention_impl = attention_impl

    devices = int(os.environ.get("WORLD_SIZE", "1"))

    out_dir = init_out_dir(out_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    logger = choose_logger(
        logger_name, out_dir, name=f"pretrain-{config.name}", log_interval=train.log_interval
    )

    if devices > 1:
        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16)

        fsdp_args = dict(state_dict_type="full", sharding_strategy=fsdp_strategy, mixed_precision=mixed_precision)
        if not train.fsdp_full_wrap:
            fsdp_args["auto_wrap_policy"] = {Block}
        if train.activation_ckpt:
            fsdp_args["activation_checkpointing_policy"] = {Block}
        strategy = FSDPStrategy(**fsdp_args)
    else:
        strategy = "auto"
    fabric = L.Fabric(devices=devices, strategy=strategy, precision="bf16-mixed", loggers=[logger])
    fabric.launch()

    fabric.print(pprint.pformat(hparams))
    if logger_name in ("tensorboard", "wandb"):
        log_hparams = copy.deepcopy(hparams)
        log_hparams['out_dir'] = str(log_hparams['out_dir'])
        log_hparams['tokenizer_dir'] = str(log_hparams['tokenizer_dir'])
        log_hparams['data'].local_cache = str(log_hparams['data'].local_cache)
        log_hparams['data'].download_dir = str(log_hparams['data'].download_dir)
        fabric.logger.log_hyperparams(log_hparams)


    main(
        fabric,
        devices,
        seed,
        initial_checkpoint_dir,
        resume,
        config,
        data,
        out_dir,
        tokenizer_dir,
        tokenizer,
        train,
        eval,
        shuffle_block_size,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Optional[Path],
    resume: Union[bool, Path],
    config: Config,
    data: DataModule,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    tokenizer: Optional[Tokenizer],
    train: TrainArgs,
    eval: EvalArgs,
    shuffle_block_size: int,
) -> None:
    validate_args(train, eval, initial_checkpoint_dir, resume)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)

    initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)

    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight
    if train.max_seq_length:
        model.max_seq_length = train.max_seq_length

    assert train.seq_len_data <= model.max_seq_length
    if train.seq_len_data is None:
        train.seq_len_data = model.max_seq_length
    

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    if train.torch_compile:
        model = torch.compile(model)

    model = fabric.setup(module=model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train.learning_rate,
        weight_decay=train.weight_decay,
        betas=(train.beta1, train.beta2),
        fused=True,
    )
    optimizer = fabric.setup_optimizers(optimizer)

    train_dataloader, val_dataloaders = get_dataloaders(fabric, data, tokenizer, train, train.seq_len_data, shuffle_block_size)
    dataloaders = [train_dataloader] + val_dataloaders
    print(f"Train dataset: {len(data.train_dataset)} samples | {len(train_dataloader)} batches")
    for i, val_dataloader in enumerate(val_dataloaders):
        print(f"Valid dataset {i}: {len(val_dataloader)} batches")
    # dataloaders = fabric.setup_dataloaders(*dataloaders)
    # we don't use litgpt built it sampler for multi rank / multi node because mosaic streaming do it for us
    train_dataloader, val_dataloaders = dataloaders[0], dataloaders[1:]

    if initial_checkpoint_dir:
        fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

    state = {
        "model": model,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "iter_num": 0,
        "step_count": 0,
    }

    if resume is True:
        resume = max(out_dir.rglob("step-*/*.pth"), key=(lambda p: int(p.parent.name.split("-")[1])))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume / "lit_model.pth", state)

    train_time = time.perf_counter()
    fit(fabric, devices, state, train_dataloader, val_dataloaders, out_dir, tokenizer_dir, train, eval)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def fit(
    fabric: L.Fabric,
    devices: int,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloaders: List[DataLoader],
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    train: TrainArgs,
    eval: EvalArgs,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]


    # validate(fabric=fabric, model, val_dataloaders[0], max_iters=2, train=train)  # sanity check
    # we are removing because torch compile must be first run with training for better performance

    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        config_copy = copy.deepcopy(model.config)
        config_copy.attention_impl = "sdpa" # we force sdpa because fa2 cannot be used for meta model
        meta_model = GPT(config_copy)
        x = torch.randint(0, 1, (train.micro_batch_size, train.seq_len_data))
        model_fwd = lambda: meta_model(x)

        if train.z_loss:
            model_loss = lambda y: sum(cross_entropy_max_z_loss(y, x, train.z_loss_weight)) # z loss return two loss
        else:
            model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)

        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = train.max_tokens // fabric.world_size
    tokens_per_iter = train.micro_batch_size * train.seq_len_data
    max_iters = max_tokens_per_device // tokens_per_iter
    log_iter_interval = train.log_interval * train.gradient_accumulation_iters(devices)
    log_activation_interval = train.log_stability_interval * train.gradient_accumulation_iters(devices) if train.log_stability_interval else None
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices), sync_on_compute=False).to(
        fabric.device
    )

    if train.z_loss:
        running_z_loss = RunningMean(window=train.gradient_accumulation_iters(devices), sync_on_compute=False).to(
            fabric.device
        )
    fabric.barrier()
    total_t0 = time.perf_counter()
    val_loss = ["n/a"] * len(val_dataloaders)

    current_time = time.time()
    warmup_iters = train.lr_warmup_steps * train.gradient_accumulation_iters(devices)
    
    fabric.print(f"train.gradient_accumulation_iters(devices): {train.gradient_accumulation_iters(devices)}, micro_batch_size: {train.micro_batch_size}, global_batch_size: {train.global_batch_size}, devices: {devices}")

    for train_data in train_iterator:
        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(train.learning_rate, state["iter_num"], warmup_iters, max_iters, train.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1

        logging_stability_metrics = train.log_stability_interval is not None and state["iter_num"] % log_activation_interval == 0
        if logging_stability_metrics:
            activation_monitor = ActivationNormMetric(target_layers=train.stability_target_layers, gradient_accumulation_steps=train.gradient_accumulation_iters(devices))
            activation_monitor.register_metrics_hooks(model)

        iter_t0 = time.perf_counter()
        
        _, T = train_data["input_ids"].shape
        input_ids = train_data["input_ids"][:, 0 : T - 1].contiguous().long().to(fabric.device)
        targets = train_data["labels"][:, 1 : T].contiguous().long().to(fabric.device)
        seqlens = train_data.get("seqlens", None)
        if seqlens is not None:
            seqlens = seqlens.to(fabric.device)
            torch._dynamo.mark_dynamic(seqlens, 0)


        
        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices) != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, seqlens=seqlens)
            if train.z_loss:
                ce_loss, z_loss = cross_entropy_max_z_loss(logits, targets, train.z_loss_weight)
                loss = ce_loss + z_loss # todo(sami): check if it is more performant to backward through both loss instead of summing and doing one backward.
            else:
                ce_loss = chunked_cross_entropy(logits, targets)
                loss = ce_loss
            fabric.backward(loss / train.gradient_accumulation_iters(devices))

        running_loss.update(ce_loss.detach())
        if train.z_loss:
            running_z_loss.update(z_loss.detach())

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)
            if logging_stability_metrics:
                grads_norm_to_log = get_grad_norm(model, train.stability_target_layers)

            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

            if logging_stability_metrics:
                activation_monitor.remove_hooks()

        if state["iter_num"] % log_iter_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            if train.z_loss:
                z_loss = running_z_loss.compute().item()
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * train.micro_batch_size),
                lengths=(state["iter_num"] * train.micro_batch_size * train.seq_len_data),
            )
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"] * train.micro_batch_size * train.seq_len_data,
                "total_tokens": (state["iter_num"] * train.micro_batch_size * train.seq_len_data * fabric.world_size),
                "learning_rate": lr,
                "tokens_per_second": train.seq_len_data * train.global_batch_size / (time.time() - current_time),
            }
            current_time = time.time()
            if train.z_loss:
                metrics["z_loss"] = z_loss
            if isinstance(val_loss[0], float):
                val_loss = [f"{v:.3f}" for v in val_loss]
            val_loss_str = ", ".join(f"val {i}: {v}" for i, v in enumerate(val_loss))
            fabric.print(
                f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" {val_loss_str} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
                f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
            )

            throughput_metrics = throughput.compute()
            if "samples_per_sec" in throughput_metrics.keys():
                throughput_metrics["token_per_sec"] = throughput_metrics["samples_per_sec"] * train.seq_len_data
            metrics.update(throughput_metrics)
            if logging_stability_metrics:
                metrics.update(activation_monitor.log_activations)
                metrics.update(grads_norm_to_log)
            fabric.log_dict(metrics, step=state["iter_num"] - 1)

        for i, val_dataloader in enumerate(val_dataloaders):

            if val_dataloader is not None and not is_accumulating and state["step_count"] % eval.interval == 0:
                t0 = time.perf_counter()
                val_loss[i] = validate(fabric, model, val_dataloader, max_iters=eval.max_iters, train=train)
                val_loss[i] = val_loss[i].item() 
                td = time.perf_counter() - t0

                fabric.print(f"iter {state['iter_num']}: val set {i} loss {val_loss[i]:.4f}, val time: {td * 1000:.2f} ms")
                metrics = {f"val_{i}_loss": float(val_loss[i]), f"val_{i}_ppl": math.exp(val_loss[i])}
                fabric.log_dict(metrics, step=state["iter_num"] - 1)
                fabric.barrier()

        if train.save_interval is not None and not is_accumulating and state["step_count"] % train.save_interval == 0:
            checkpoint_file = out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth"
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
            # Use buffer to fix issue with serializing state['train_dataloader']
            buffer_train_dataloader = state['train_dataloader']
            state['train_dataloader'] = buffer_train_dataloader.state_dict()
            fabric.save(checkpoint_file, state)
            state['train_dataloader'] = buffer_train_dataloader
            if fabric.global_rank == 0:
                save_hyperparameters(setup, checkpoint_file.parent)
                if tokenizer_dir is not None:
                    copy_config_files(tokenizer_dir, checkpoint_file.parent)
                save_config(model.config, checkpoint_file.parent)


@torch.no_grad()
def validate(fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int, train: TrainArgs) -> torch.Tensor:
    fabric.barrier()
    fabric.print("Validating ...")
    model.eval()

    losses = []
    for k, batch in enumerate(val_dataloader):
        if k >= max_iters:
            break

        bs, T = batch["input_ids"].shape
        if bs != train.micro_batch_size:
            break # the bs that micro batch size being smaller happened only for the last batch of the val stream but it breaks torch.compile
        input_ids = batch["input_ids"][:, 0 : T - 1].contiguous().long().to(fabric.device)
        targets = batch["labels"][:, 1 : T].contiguous().long().to(fabric.device)
        seqlens = batch.get("seqlens", None)
        if seqlens is not None:
            seqlens = seqlens.to(fabric.device)
            torch._dynamo.mark_dynamic(seqlens, 0)
            # https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html
            # seqlens has a dynamic shape but one dimension, this allow to still torch compile

        logits = model(input_ids, seqlens=seqlens)


        loss = chunked_cross_entropy(logits, targets)
        losses.append(loss)

    val_loss = torch.stack(losses).mean()
    model.train()
    fabric.barrier()
    return val_loss


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs, max_seq_length: int, shuffle_block_size
) -> Tuple[DataLoader, List[DataLoader]]:
    """
    here max_seq_length rules the dataloader but does not impact the model.
    """
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=max_seq_length, shuffle_block_size=shuffle_block_size)
    data.setup(rank=fabric.local_rank)
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    if isinstance(val_dataloader, DataLoader):
        val_dataloader = [val_dataloader]
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def initialize_weights(fabric: L.Fabric, model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd))

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer))

    if not isinstance(fabric.strategy, FSDPStrategy):
        reset_parameters(model)


def init_out_dir(out_dir: Path) -> Path:
    if not out_dir.is_absolute() and "LIGHTNING_ARTIFACTS_DIR" in os.environ:
        return Path(os.getenv("LIGHTNING_ARTIFACTS_DIR")) / out_dir
    return out_dir


def validate_args(train: TrainArgs, eval: EvalArgs, initial_checkpoint_dir, resume) -> None:
    issues = []
    unsupported = [(train, ["max_steps", "epochs"]), (eval, ["max_new_tokens"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if initial_checkpoint_dir and resume:
        issues.append("Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one.")
    if issues:
        raise ValueError("\n".join(issues))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = "GENOMIC_DEBUG" not in os.environ 
    # allow to continue when compiling failed
    # can be disable by setting GENOMIC_DEBUG=1 as env var
    CLI(setup)
