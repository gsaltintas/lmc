from .activation_alignment import *
from .alignment import *
from .perm_stability import *
from .utils import *
from .weight_alignment import *


def get_cost(
        perm_spec: PermSpec,
        A: Union[Dict[str, torch.Tensor], nn.Module],
        B: Union[Dict[str, torch.Tensor], nn.Module] = None,
        cost: Union[str, callable] = "linear",
        align_obj=None,
        align_type: str = "weight",
        dataloader: Union[torch.utils.data.DataLoader, None] = None,
        normalize_m: bool = True,
        return_minimizing_permutation: bool = False,
) -> Dict[str, np.ndarray]:
    B = A if B is None else B
    kernel = get_kernel_function(cost)

    # align and get similarity matrices
    if align_obj is None:
        if align_type == "activation":
            assert isinstance(A, nn.Module) and isinstance(B, nn.Module)
            assert dataloader is not None
            cost = get_activations_cost_matrix(perm_spec, A, B, dataloader, kernel_func=kernel, num_samples=-1)
        else:
            cost = weight_matching_cost(perm_spec, A, B, kernel_func=kernel)
    return cost
    params_A = A.state_dict() if isinstance(A, nn.Module) else A
    params_B = B.state_dict() if isinstance(B, nn.Module) else B

    # normalize, then align (this will change the alignment somewhat vs not normalizing)
    permutations, similarity_matrices = align_obj.fit(params_A, params_B)

    costs = {}


def create_permspec_from_model(model: nn.Module, model_name: Optional[str] = None) -> PermSpec:
    names_to_perms = OrderedDict()
    perm_counts = 0
    prev_layer_perm = None
    layer_outs = []
    norm_outs = []
    
    for name, module in model.named_modules():
        p_out = f"P_{perm_counts}"
        p_in = prev_layer_perm if prev_layer_perm is not None else None
        if isinstance(module, nn.Linear):
            names_to_perms[name] = (p_out, p_in)
            prev_layer_perm = p_out
            layer_outs.append(p_out)
            perm_counts += 1
        elif "norm" in name: #or is_norm_layer(module):
            p_out = layer_outs[-1]
            names_to_perms[name] = (p_out, )
            prev_layer_perm = p_out
            norm_outs.append(p_out)
        elif isinstance(module, nn.Conv2d):
            names_to_perms[name] = (p_out, p_in)
            prev_layer_perm = p_out
            layer_outs.append(p_out)
    
    # Set the output permutation of the last layer to None
    if layer_outs:
        last_layer = list(names_to_perms.keys())[-1]
        names_to_perms[last_layer] = (None, *names_to_perms[last_layer][1:])
   
    return PermSpec(names_to_perms, model_name=model_name)
    