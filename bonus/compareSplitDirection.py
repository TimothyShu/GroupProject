import torch
from sklearn.model_selection import train_test_split


def to_real_vector(v):
    v = torch.as_tensor(v).detach().cpu().flatten()
    if torch.is_complex(v):
        if torch.max(v.imag.abs()) > 1e-6:
            print(f"Warning: split direction has non-trivial imaginary component; max imag={torch.max(v.imag.abs()).item():.3e}")
        v = v.real
    return v.float()


def normalize_vector(v):
    return v / torch.linalg.norm(v)


def similarity_label(abs_cosine):
    if abs_cosine >= 0.90:
        return "similar"
    else:
        return "different"


def prominent_axis_info(direction):
    axis_idx = int(torch.argmax(direction.abs()).item())
    axis_value = float(direction[axis_idx].item())
    axis_abs_value = float(direction.abs()[axis_idx].item())
    return axis_idx, axis_value, axis_abs_value

def compare_split_directions(dir1, dir2):

    # error with different devices to using whatever dir1 is on, since the split directions are derived from the data and should be on the same device
    device = dir1.device
    dir1 = to_real_vector(dir1).to(device)
    dir2 = to_real_vector(dir2).to(device)

    cosine_similarity = torch.dot(dir1, dir2).item()
    abs_cosine_similarity = abs(cosine_similarity)

    angle_degrees = torch.rad2deg(
        torch.arccos(torch.tensor(min(max(abs_cosine_similarity, -1.0), 1.0)))
    ).item()

    my_axis, my_axis_value, my_axis_abs = prominent_axis_info(dir1)
    xrfm_axis, xrfm_axis_value, xrfm_axis_abs = prominent_axis_info(dir2)


    print("\nFirst-split comparison: my model vs xRFM")
    print(f"  cosine similarity           = {cosine_similarity:.6f}")
    print(f"  abs cosine similarity       = {abs_cosine_similarity:.6f}")
    print(f"  angle between directions    = {angle_degrees:.4f} degrees")
    print(f"  prominent axis (my model)   = feature[{my_axis}] value={my_axis_value:.6f} abs={my_axis_abs:.6f}")
    print(f"  prominent axis (xRFM model) = feature[{xrfm_axis}] value={xrfm_axis_value:.6f} abs={xrfm_axis_abs:.6f}")
    print(f"  interpretation              = {similarity_label(abs_cosine_similarity)}")

def get_xrfm_split_directions(xrfm_model):
    directions = []
    for node in preorder_traverse_xrfm_tree(xrfm_model.trees[0]):
        if node["type"] == "split":
            directions.append(to_real_vector(node["split_direction"]))
    return directions

def get_xrfm_split_points(xrfm_model):
    points = []
    for node in preorder_traverse_xrfm_tree(xrfm_model.trees[0]):
        if node["type"] == "split":
            points.append(node["split_point"])
    return points

def preorder_traverse_xrfm_tree(node):
    """
    Preorder traversal for a single xRFM tree node dictionary.
    Visit order: node -> left subtree -> right subtree.
    Returns a flat list of node dicts (both split and leaf).
    """
    if node is None:
        return []

    nodes = [node]
    if node["type"] == "split":
        nodes.extend(preorder_traverse_xrfm_tree(node["left"]))
        nodes.extend(preorder_traverse_xrfm_tree(node["right"]))
    return nodes

def compare_split_direction(model1, model2):

    my_split_directions = get_xrfm_split_directions(model1)
    xrfm_split_directions = get_xrfm_split_directions(model2)
    my_split_points = get_xrfm_split_points(model1)
    xrfm_split_points = get_xrfm_split_points(model2)

    print(f"Number of splits in my model: {len(my_split_directions)}")
    print(f"Number of splits in xRFM model: {len(xrfm_split_directions)}")

    # should verify that all arrays are the same length, since we should have the same number of splits
    assert len(my_split_directions) == len(xrfm_split_directions), "Number of splits in my model and xRFM model do not match"
    assert len(my_split_points) == len(xrfm_split_points), "Number of splits in my model and xRFM model do not match"

    for i in range(len(my_split_directions)):
        print(f"\nComparing split {i}...")
        compare_split_directions(
            to_real_vector(my_split_directions[i]),
            to_real_vector(xrfm_split_directions[i])  # get the first split direction from xRFM
        )
