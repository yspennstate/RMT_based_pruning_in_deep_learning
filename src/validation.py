import time
import torch
from torch.utils.data import Subset
from enum import Enum
import torch.distributed as dist
from random import shuffle
import torchvision.datasets as datasets


def validate(val_loader, model, criterion, args):
    """
    Validates the model on the given validation data loader.

    Args:
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        model (torch.nn.Module): The model to validate.
        criterion (torch.nn.Module): Loss function.
        args (argparse.Namespace): Arguments containing configuration.

    Returns:
        tuple: Average top-1 and top-5 accuracy.
    """

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=False)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=False)

                # Compute output and loss
                output = model(images)
                loss = criterion(output, target)

                # Measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

                # Free per-batch tensors and periodically reclaim allocator slabs.
                # Required to prevent fragmentation-driven OOM on the 8GB GPU
                # that took the trading bots down on 2026-04-06.
                del images, target, output, loss, acc1, acc5
                if torch.cuda.is_available() and (i % 50 == 0):
                    torch.cuda.empty_cache()

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader)
        + (
            args.distributed
            and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))
        ),
        [batch_time, losses, top1, top5],
        prefix="Test: ",
    )

    # Switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (
        len(val_loader.sampler) * args.world_size < len(val_loader.dataset)
    ):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg, top5.avg


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class arg_input:
    def __init__(
        self,
        gpu=None,
        print_freq=10,
        world_size=-1,
        workers=4,
        multiprocessing_distributed=False,
    ) -> None:
        self.gpu = gpu
        self.print_freq = print_freq
        self.world_size = world_size
        self.workers = workers
        self.distributed = world_size > 1 or multiprocessing_distributed


def evaluate(loader, resnet, device):
    """
    Evaluates the model on the given data loader.

    Args:
        loader (torch.utils.data.DataLoader): Data loader.
        resnet (torch.nn.Module): The model to evaluate.
        device (torch.device): The device to run the evaluation on.

    Returns:
        tuple: Average top-1 and top-5 accuracy.
    """
    return validate(
        loader, resnet, torch.nn.CrossEntropyLoss().to(device), arg_input(gpu=device)
    )


class HFImageNetDataset(torch.utils.data.Dataset):
    """Wraps HuggingFace ImageNet dataset for use with PyTorch DataLoader."""
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

_VAL_DATASET_CACHE = None

def get_val_dataset(preprocess, batch_size=8):
    """
    Gets the validation dataset from HuggingFace with the specified preprocessing.
    The HF dataset object is cached in module state so changing batch_size between
    calls is cheap (only the DataLoader is rebuilt).

    Args:
        preprocess: Preprocessing transformations.
        batch_size: DataLoader batch size. Smaller = shorter GPU kernels = nicer
                    to other apps (Chrome, games) when the user is active.
    """
    global _VAL_DATASET_CACHE
    if _VAL_DATASET_CACHE is None:
        from datasets import load_dataset
        import os, glob
        # HF token must be set externally: `export HF_TOKEN=...` (HuggingFace)
if "HF_TOKEN" not in os.environ:
    raise RuntimeError("Set HF_TOKEN env var to download HuggingFace assets.")
        # Linux/Runpod path: load only the locally-downloaded validation parquet shards
        # (avoids triggering a full-repo re-download via load_dataset).
        _LOCAL_VAL_PARQUETS = sorted(glob.glob(
            "/workspace/hf_cache/hub/datasets--ILSVRC--imagenet-1k/snapshots/*/data/validation-*.parquet"
        ))
        if _LOCAL_VAL_PARQUETS:
            hf_ds = load_dataset("parquet", data_files=_LOCAL_VAL_PARQUETS, split="train")
        else:
            hf_ds = load_dataset("ILSVRC/imagenet-1k", split="validation",
                                 cache_dir="/workspace/hf_cache",
                                 num_proc=4)
        val_dataset = HFImageNetDataset(hf_ds, preprocess)
        # Use full 50K val set (no subset) for accurate top-1 reporting
        _VAL_DATASET_CACHE = val_dataset

    # num_workers=8 + pin_memory=True: avoids Windows DataLoader-worker leaks
    # that contributed to the 2026-04-06 hard crash. Slower but stable.
    return torch.utils.data.DataLoader(_VAL_DATASET_CACHE, batch_size=batch_size,
                                        shuffle=False, num_workers=8,
                                        pin_memory=True)
