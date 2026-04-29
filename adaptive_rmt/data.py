from __future__ import annotations

import glob
import io
import random
from dataclasses import dataclass

from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


@dataclass
class ProbeSnapshot:
    top1: float
    loss: float
    mean_margin: float
    logits: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class ProbeLoaderFactory:
    files: list[str]
    transform: object
    batch_size: int
    num_workers: int
    pin_memory: bool
    seed: int
    shuffle_buffer: int
    length: int

    def make_loader(self, probe_seed: int) -> DataLoader:
        # Build a fresh randomized probe loader each time so controller decisions do
        # not overfit to one frozen slice of the training set.
        probe_ds = DirectParquetIterableDataset(
            self.files,
            self.transform,
            shuffle=True,
            shuffle_buffer=self.shuffle_buffer,
            seed=self.seed + probe_seed,
            length=self.length,
        )
        loader_kwargs = dict(
            dataset=probe_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = False
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(**loader_kwargs)


def decode_image_field(image_field):
    if isinstance(image_field, Image.Image):
        image = image_field
    elif isinstance(image_field, dict):
        image_bytes = image_field.get("bytes")
        image_path = image_field.get("path")
        if image_bytes is not None:
            image = Image.open(io.BytesIO(image_bytes))
        elif image_path:
            image = Image.open(image_path)
        else:
            raise ValueError("Image field dict missing both bytes and path")
    elif isinstance(image_field, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_field))
    elif isinstance(image_field, str):
        image = Image.open(image_field)
    else:
        raise TypeError(f"Unsupported image field type: {type(image_field)!r}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def parquet_num_rows(files: list[str]) -> int:
    total = 0
    for path in files:
        total += pq.ParquetFile(path).metadata.num_rows
    return total


class DirectParquetIterableDataset(IterableDataset):
    def __init__(
        self,
        files: list[str],
        transform,
        *,
        shuffle: bool,
        shuffle_buffer: int,
        seed: int,
        length: int,
    ) -> None:
        self.files = list(files)
        self.transform = transform
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.length = length
        self._epoch = 0

    def __len__(self) -> int:
        return self.length

    def _worker_files(self) -> tuple[list[str], int]:
        worker = get_worker_info()
        if worker is None:
            return list(self.files), self.seed + self._epoch
        return list(self.files[worker.id::worker.num_workers]), self.seed + self._epoch + worker.id

    def _iter_rows(self, files: list[str]):
        for path in files:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(
                batch_size=256,
                columns=["image", "label"],
                use_threads=False,
            ):
                for row in batch.to_pylist():
                    yield row

    def _shuffle_rows(self, rows, rng: random.Random):
        buffer = []
        for row in rows:
            buffer.append(row)
            if len(buffer) >= self.shuffle_buffer:
                idx = rng.randrange(len(buffer))
                yield buffer.pop(idx)
        while buffer:
            idx = rng.randrange(len(buffer))
            yield buffer.pop(idx)

    def __iter__(self):
        files, iter_seed = self._worker_files()
        rng = random.Random(iter_seed)
        if self.shuffle:
            rng.shuffle(files)
        rows = self._iter_rows(files)
        if self.shuffle and self.shuffle_buffer > 1:
            rows = self._shuffle_rows(rows, rng)
        for item in rows:
            image = decode_image_field(item["image"])
            label = int(item["label"])
            if self.transform is not None:
                image = self.transform(image)
            yield image, label
        self._epoch += 1


def build_loaders(
    *,
    train_glob: str,
    val_glob: str,
    preprocess_train,
    preprocess_val,
    batch_size_train: int,
    batch_size_val: int,
    train_probe_batch_size: int,
    num_workers: int,
    shuffle_buffer: int,
    seed: int,
    log_fn,
):
    train_files = sorted(glob.glob(train_glob))
    val_files = sorted(glob.glob(val_glob))
    log_fn(f"Train parquet files: {len(train_files)}, val parquet files: {len(val_files)}")
    if not train_files:
        raise FileNotFoundError(f"Train parquet shards not found for glob: {train_glob}")
    if not val_files:
        raise FileNotFoundError(f"Val parquet shards not found for glob: {val_glob}")

    train_rows = parquet_num_rows(train_files)
    val_rows = parquet_num_rows(val_files)
    log_fn(f"Direct parquet row counts: train={train_rows}, val={val_rows}")

    train_ds = DirectParquetIterableDataset(
        train_files,
        preprocess_train,
        shuffle=True,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        length=train_rows,
    )
    val_ds = DirectParquetIterableDataset(
        val_files,
        preprocess_val,
        shuffle=False,
        shuffle_buffer=0,
        seed=seed,
        length=val_rows,
    )
    train_loader_kwargs = dict(
        dataset=train_ds,
        batch_size=batch_size_train,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader_kwargs = dict(
        dataset=val_ds,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2
        val_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["prefetch_factor"] = 2

    train_eval_ds = DirectParquetIterableDataset(
        train_files,
        preprocess_val,
        shuffle=False,
        shuffle_buffer=0,
        seed=seed,
        length=train_rows,
    )
    train_eval_loader_kwargs = dict(
        dataset=train_eval_ds,
        batch_size=train_probe_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if num_workers > 0:
        train_eval_loader_kwargs["persistent_workers"] = True
        train_eval_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)
    train_eval_loader = DataLoader(**train_eval_loader_kwargs)
    train_probe_factory = ProbeLoaderFactory(
        files=train_files,
        transform=preprocess_val,
        batch_size=train_probe_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        seed=seed,
        shuffle_buffer=max(shuffle_buffer, train_probe_batch_size * 8),
        length=train_rows,
    )
    return train_loader, val_loader, train_eval_loader, train_probe_factory


@torch.no_grad()
def capture_probe_snapshot(
    model,
    loader,
    device: torch.device,
    *,
    log_fn,
    label: str = "",
    max_batches: int | None = None,
    store_logits: bool = True,
) -> ProbeSnapshot:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    margin_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        autocast_kwargs = {"device_type": device.type, "enabled": device.type == "cuda"}
        if device.type == "cuda":
            autocast_kwargs["dtype"] = torch.bfloat16
        with torch.amp.autocast(**autocast_kwargs):
            output = model(images)
            loss = criterion(output, targets)
        output_fp32 = output.detach().float()
        prediction = output.argmax(dim=1)
        correct += int((prediction == targets).sum().item())
        total += int(targets.size(0))
        loss_sum += float(loss.item()) * int(targets.size(0))
        true_logits = output_fp32.gather(1, targets.unsqueeze(1)).squeeze(1)
        masked = output_fp32.clone()
        masked.scatter_(1, targets.unsqueeze(1), float("-inf"))
        other_logits = masked.max(dim=1).values
        margin_sum += float((true_logits - other_logits).sum().item())
        if store_logits:
            all_logits.append(output_fp32.cpu())
            all_labels.append(targets.detach().cpu())
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    top1 = 100.0 * correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    mean_margin = margin_sum / max(total, 1)
    log_fn(
        f"  eval {label}: top1={top1:.2f}% loss={avg_loss:.4f} "
        f"margin={mean_margin:.4f} on {total} images"
    )
    return ProbeSnapshot(
        top1=top1,
        loss=avg_loss,
        mean_margin=mean_margin,
        logits=torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, 0), dtype=torch.float32),
        labels=torch.cat(all_labels, dim=0) if all_labels else torch.empty((0,), dtype=torch.long),
    )


def compare_probe_snapshots(reference: ProbeSnapshot, current: ProbeSnapshot) -> dict[str, float]:
    if reference.logits.shape != current.logits.shape:
        raise ValueError(
            f"Probe snapshot shape mismatch: {reference.logits.shape} vs {current.logits.shape}"
        )
    if reference.labels.shape != current.labels.shape:
        raise ValueError(
            f"Probe label shape mismatch: {reference.labels.shape} vs {current.labels.shape}"
        )

    ref_logits = reference.logits
    cur_logits = current.logits
    labels = reference.labels
    mean_abs_logit_drift = float((cur_logits - ref_logits).abs().mean().item())

    ref_true = ref_logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    cur_true = cur_logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    ref_masked = ref_logits.clone()
    cur_masked = cur_logits.clone()
    ref_masked.scatter_(1, labels.unsqueeze(1), float("-inf"))
    cur_masked.scatter_(1, labels.unsqueeze(1), float("-inf"))
    ref_margin = ref_true - ref_masked.max(dim=1).values
    cur_margin = cur_true - cur_masked.max(dim=1).values
    margin_flip_rate = float(((ref_margin > 0) != (cur_margin > 0)).float().mean().item())
    mean_margin_change = float((cur_margin - ref_margin).mean().item())
    return {
        "mean_abs_logit_drift": mean_abs_logit_drift,
        "margin_flip_rate": margin_flip_rate,
        "mean_margin_change": mean_margin_change,
    }


@torch.no_grad()
def evaluate_model_metrics(
    model,
    loader,
    device: torch.device,
    *,
    log_fn,
    label: str = "",
    max_batches: int | None = None,
) -> tuple[float, float]:
    snapshot = capture_probe_snapshot(
        model,
        loader,
        device,
        log_fn=log_fn,
        label=label,
        max_batches=max_batches,
    )
    return snapshot.top1, snapshot.loss


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device: torch.device,
    *,
    log_fn,
    label: str = "",
    max_batches: int | None = None,
) -> float:
    top1, _ = evaluate_model_metrics(
        model,
        loader,
        device,
        log_fn=log_fn,
        label=label,
        max_batches=max_batches,
    )
    return top1
