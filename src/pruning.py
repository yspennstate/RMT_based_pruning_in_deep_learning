import torch
from SplittableLayers import (
    SplittableConv,
    SplittableLinear,
)


def naive_prune(layer, threshold):
    """
    Prune the weights of a layer by setting values below a threshold to zero.

    Parameters:
    layer (torch.nn.Module): The layer to prune.
    threshold (float): The threshold below which weights are set to zero.
    """
    with torch.no_grad():
        weight_mask = torch.abs(layer.weight) > threshold
        layer.weight *= weight_mask.float()


def count_nonzero_params(model):
    """
    Count the number of non-zero parameters in a model.

    Parameters:
    model (torch.nn.Module): The model to count parameters for.

    Returns:
    int: The number of non-zero parameters.
    """
    count = 0
    for param in model.parameters():
        count += torch.count_nonzero(param).item()
    return count


def count_total_params(models):
    """
    Count the total number of parameters in a model or list of models.

    Parameters:
    models (torch.nn.Module or list): The model or list of models to count parameters for.

    Returns:
    int: The total number of parameters.
    """
    if isinstance(models, list):
        return sum(p.numel() for model in models for p in model.parameters())
    else:
        return sum(p.numel() for p in models.parameters())


def replace_layers(m, alpha, beta, goodnessOfFitCutoff, depth=0):
    """
    Replace layers in a model with splittable layers.

    Parameters:
    m (torch.nn.Module): The model to replace layers in.
    alpha (float): Alpha parameter for Splittable layers.
    beta (float): Beta parameter for Splittable layers.
    goodnessOfFitCutoff (float): Goodness of fit cutoff for Splittable layers.
    depth (int): Current depth of recursion, default is 0.
    """
    replacable_layers = {}
    for name, module in m.named_children():
        if name == "":
            continue
        try:
            module.goodnessOfFitCutoff
            continue
        except:
            pass

        if isinstance(module, torch.nn.MultiheadAttention):
            replacable_layers[name] = (module, "attention")
            continue

        if len(list(module.named_modules())) > 1:
            replace_layers(module, alpha, beta, goodnessOfFitCutoff, depth=depth + 1)
            continue

        if isinstance(module, torch.nn.Conv2d):
            replacable_layers[name] = (module, "conv")
            continue

        if isinstance(module, torch.nn.Linear):
            replacable_layers[name] = (module, "linear")
            continue
    for name, pair in replacable_layers.items():
        module, type_str = pair
        if type_str == "conv":
            setattr(
                m,
                name,
                SplittableConv.from_layer(
                    module,
                    alpha=alpha,
                    beta=beta,
                    goodnessOfFitCutoff=goodnessOfFitCutoff,
                ),
            )
        if type_str == "linear":
            setattr(
                m,
                name,
                SplittableLinear.from_layer(
                    module,
                    alpha=alpha,
                    beta=beta,
                    goodnessOfFitCutoff=goodnessOfFitCutoff,
                ),
            )


# Global cache for layer metrics — computed once, reused across all trials and cycles
_layer_metrics_cache = None

def compute_layer_metrics_once(model):
    """
    Compute LinfError and percentage_less_than_splus for each splittable layer
    on the unpruned model. Called once at startup, cached forever.
    """
    global _layer_metrics_cache
    if _layer_metrics_cache is not None:
        return _layer_metrics_cache

    print("Computing RMT metrics for all layers (one-time)...")
    _layer_metrics_cache = {}
    splittable_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (SplittableConv, SplittableLinear))
    ]
    for name, layer in splittable_layers:
        # Call split with threshold=0 to get metrics without modifying weights
        result, splus, LinfError, percentage_less_than_splus = layer.split(1, 0)
        _layer_metrics_cache[name] = {
            'LinfError': LinfError,
            'percentage_less_than_splus': percentage_less_than_splus,
            'splus': splus,
        }
        print(f"  {name}: LinfError={LinfError:.4f} spike%={percentage_less_than_splus:.1f}%")

    print(f"Cached metrics for {len(_layer_metrics_cache)} layers.\n")
    return _layer_metrics_cache


def prune_model(model, target_reduction, i, n_prune_cycles, device,
                hp_a=4.0, hp_b=1.5, enable_sv_pruning=True, cached_metrics=None):
    """
    Prune the model to achieve a target reduction in parameters.

    Parameters:
    model (torch.nn.Module): The model to prune.
    target_reduction (float): The target reduction in parameters (rho, tunable).
    i (int): Current pruning cycle (1-indexed).
    n_prune_cycles (int): Total number of pruning cycles.
    device (torch.device): The device to move parameters to.
    hp_a (float): Linear coefficient for pruning threshold (original: 4.0).
    hp_b (float): Exponent numerator: exponent = hp_b / i (original: 1.5).
    enable_sv_pruning (bool): Whether to run SV pruning on even cycles (default True).
    cached_metrics (dict or None): Pre-computed layer metrics. If provided, skip layer.split().

    Returns:
    dict: Dictionary of Linf errors for each layer.
    """
    linf_errors = {}
    splittable_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (SplittableConv, SplittableLinear))
    ]
    for name, layer in splittable_layers:
        current_reduction = 0
        num_params_unpruned = 0
        initial_pruning_factor = 0.00001 * i  # Starting pruning factor
        step_size = 0.00002  # Incremental step for adjusting pruning factor
        num_params_unpruned_now = count_nonzero_params(layer)
        pruning_factor = initial_pruning_factor
        W = layer.get_matrix()
        M, N = W.shape
        end_scale = 1
        scale = 1 + i * (end_scale - 1) / n_prune_cycles

        if cached_metrics is not None and name in cached_metrics:
            # Use pre-computed metrics — no SVD needed
            LinfError = cached_metrics[name]['LinfError']
            percentage_less_than_splus = cached_metrics[name]['percentage_less_than_splus']
        else:
            # Compute fresh (original behavior)
            if enable_sv_pruning and i % 2 == 0:
                result, splus, LinfError, percentage_less_than_splus = layer.split(
                    1, scale * 750 * 0.000000015 * target_reduction * N * M
                )  # type: ignore
            else:
                result, splus, LinfError, percentage_less_than_splus = layer.split(
                    1, 0
                )  # type: ignore

        linf_errors[name] = LinfError

        # Compute target number of params to prune for this layer
        randomness_score = (
            (1 - LinfError) ** (hp_b / i)
            * (percentage_less_than_splus / 100) ** (hp_b / i)
        )
        target_prune_count = int(randomness_score * target_reduction * num_params_unpruned_now)

        if target_prune_count > 0:
            # Collect all nonzero weight values across sublayers, find threshold directly
            all_abs_weights = []
            for submodule in layer.modules():
                if isinstance(submodule, torch.nn.Conv2d) or isinstance(
                    submodule, torch.nn.Linear
                ):
                    w = submodule.weight.data.abs()
                    nonzero_vals = w[w > 0]
                    if len(nonzero_vals) > 0:
                        all_abs_weights.append(nonzero_vals)

            if all_abs_weights:
                all_abs_weights = torch.cat(all_abs_weights)
                # Clamp target to not exceed available nonzero weights
                target_prune_count = min(target_prune_count, len(all_abs_weights))
                if target_prune_count > 0:
                    # Sort and pick the threshold at the target index
                    sorted_vals, _ = torch.sort(all_abs_weights)
                    threshold = sorted_vals[target_prune_count - 1].item()
                    # Apply the threshold to all sublayers
                    for submodule in layer.modules():
                        if isinstance(submodule, torch.nn.Conv2d) or isinstance(
                            submodule, torch.nn.Linear
                        ):
                            naive_prune(submodule, threshold)

    num_nonzero = count_nonzero_params(model)
    for name, param in model.named_parameters():
        if param.device.type == "cpu":
            param.data = param.data.to(device)
    return linf_errors


def freeze_pruned_weights(model):
    """
    Freeze the pruned weights of a model by setting their gradients to zero.

    Parameters:
    model (torch.nn.Module): The model to freeze pruned weights for.
    """
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Set the gradients of the pruned weights to zero where the parameter data is zero
            param.grad.data.mul_(param.data.ne(0).float())


def calculate_lasso_strength(linf_error, i):
    """
    Calculate the Lasso strength based on Linf error and pruning cycle.

    Parameters:
    linf_error (float): The Linf error.
    i (int): The current pruning cycle.

    Returns:
    float: The Lasso strength.
    """
    return 1 / (1 + 10000 * linf_error / (i))


def get_base_name(name):
    """
    Get the base name of a layer, removing specific suffixes.

    Parameters:
    name (str): The name of the layer.

    Returns:
    str: The base name of the layer.
    """
    parts = name.split(".")
    if parts[-1] in ["layer_1", "layer_2"]:
        return ".".join(parts[:-1])
    return name
