import torch
from transformers import AutoModelForCausalLM
import csv
from tqdm import tqdm
import random
import os
import fire
import numpy as np
import setproctitle

setproctitle.setproctitle("junkim100 extract_code_region")


def jaccard_similarity(tensor1, tensor2):
    # Compute the intersection (common elements)
    intersection = torch.logical_and(tensor1, tensor2).sum()
    # Compute the union (all unique elements)
    union = torch.logical_or(tensor1, tensor2).sum()
    # Calculate the Jaccard Index (IoU)
    iou = intersection.float() / union.float()

    return iou.item()


def compare_bool_matrix(bool_dict1, bool_dict2):
    params_diff_similarity = {}
    for (name1, bool_matrix1), (name2, bool_matrix2) in zip(
        bool_dict1.items(), bool_dict2.items()
    ):
        assert name1 == name2
        params_diff_similarity[name1] = jaccard_similarity(bool_matrix1, bool_matrix2)
    return params_diff_similarity


def logical_and_bool_matrix(bool_dict1, bool_dict2):
    params_diff_similarity = {}
    for (name1, bool_matrix1), (name2, bool_matrix2) in zip(
        bool_dict1.items(), bool_dict2.items()
    ):
        assert name1 == name2
        params_diff_similarity[name1] = torch.logical_and(bool_matrix1, bool_matrix2)
    return params_diff_similarity


def logical_or_bool_matrix(bool_dict1, bool_dict2):
    params_diff_similarity = {}
    for (name1, bool_matrix1), (name2, bool_matrix2) in zip(
        bool_dict1.items(), bool_dict2.items()
    ):
        assert name1 == name2
        params_diff_similarity[name1] = torch.logical_or(bool_matrix1, bool_matrix2)
    return params_diff_similarity


def calculate_row_bool_matrix(bool_dict1, num):
    params_top_row = {}
    params_top_col = {}
    params_bottom_row = {}
    params_bottom_col = {}

    for name1, bool_matrix1 in bool_dict1.items():
        row_sums = torch.sum(bool_matrix1, dim=1)
        col_sums = torch.sum(bool_matrix1, dim=0)
        top_rows_indices = torch.topk(row_sums, k=num).indices.tolist()
        top_cols_indices = torch.topk(col_sums, k=num).indices.tolist()
        min_rows_indices = torch.topk(row_sums, k=num, largest=False).indices.tolist()
        min_cols_indices = torch.topk(col_sums, k=num, largest=False).indices.tolist()
        params_top_row[name1] = top_rows_indices
        params_top_col[name1] = top_cols_indices
        params_bottom_row[name1] = min_rows_indices
        params_bottom_col[name1] = min_cols_indices
    return params_top_row, params_top_col, params_bottom_row, params_bottom_col


def get_diff_tensor(tensor_base, tensor_change):
    # Compute the difference of two tensors
    assert tensor_change.shape == tensor_base.shape
    tensor_diff = torch.abs(tensor_change - tensor_base)
    tensor_diff = tensor_diff / torch.abs(tensor_base)
    return tensor_diff


def get_top_bottom_tensor(tensor_diff, k):
    # Calculate the number of points that need to be recorded (maximum/minimum/middle 3%)
    num_points = int(k * tensor_diff.numel())
    # Find the top 3% points with the largest difference
    max_points = tensor_diff.view(-1).topk(num_points).indices
    bool_sensor_max = torch.zeros(tensor_diff.shape, dtype=torch.bool)
    bool_sensor_max.view(-1)[max_points] = True
    # Find the top 3% points with the smallest difference
    min_points = tensor_diff.view(-1).topk(num_points, largest=False).indices
    bool_sensor_min = torch.zeros(tensor_diff.shape, dtype=torch.bool)
    bool_sensor_min.view(-1)[min_points] = True
    # Randomly find 3% points
    bool_sensor_random = torch.zeros(tensor_diff.shape, dtype=torch.bool)
    # Randomly select the index of the element to be set to True
    random_points = random.sample(range(bool_sensor_random.numel()), num_points)
    # Set selected index position to True
    bool_sensor_random.view(-1)[random_points] = True
    return bool_sensor_max, bool_sensor_min, bool_sensor_random


def accumulate_matrix(param_dict1, param_dict2):
    params_diff_accumulate = {}
    for (name1, param_matrix1), (name2, param_matrix2) in zip(
        param_dict1.items(), param_dict2.items()
    ):
        assert name1 == name2
        params_diff_accumulate[name1] = param_matrix1 + param_matrix2
    return params_diff_accumulate


# Compare the average value of the parameter change amplitude of each layer for each model
def compare_parameters(model1, model2):
    params_diff = {}
    with tqdm(total=400) as pbar:
        for (name1, params1), (name2, params2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            pbar.update(1)
            if "layers." not in name1:
                continue
            assert name1 == name2  # Make sure parameter names are consistent
            params_diff[name1] = get_diff_tensor(params1, params2)
    return params_diff


def extract(
    model_name: str = "llama-3.1",
    original_model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    language_list: list = [
        "go",
        "java",
    ],
    sample_list: list = [10000],
    k: float = 0.01,
    input_dir: str = "/data_x/junkim100/code-spot/training/further_training/Llama-3.1-8B-Instruct",
):
    original_model = AutoModelForCausalLM.from_pretrained(original_model_path)

    top_k_params_dict = {}
    bottom_k_params_dict = {}
    random_k_params_dict = {}

    for samples in sample_list:
        with tqdm(total=291) as pbar:

            for name, params in original_model.named_parameters():
                if "layers." not in name:
                    continue
                grad_tensor = torch.zeros_like(params).cpu()
                pbar.update(1)
                for language in language_list:
                    # Take the abs from the grad-mul-params of the six countries and add them together
                    file_name = "{}/grad-mul-param_checkpoint_{}".format(
                        language, samples
                    )
                    file_dir = os.path.join(input_dir, file_name)
                    save_path = os.path.join(
                        file_dir, "{}.pt".format(name.replace("module.", ""))
                    )
                    grad_tensor += torch.load(save_path, map_location="cpu").abs().cpu()

                bool_sensor_max, bool_sensor_min, bool_sensor_random = (
                    get_top_bottom_tensor(grad_tensor, k)
                )
                top_k_params_dict[name] = bool_sensor_max
                bottom_k_params_dict[name] = bool_sensor_min
                random_k_params_dict[name] = bool_sensor_random

            # Save output to CSV file
            output_file = "code-region/{}/top{}.csv".format(model_name, k)

            # Make the ouput_file directory if it does not exist
            os.makedirs(
                os.path.dirname(output_file),
                exist_ok=True,
            )

            with open(output_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Parameter Name",
                        f"Parameters Difference top {k} Similarity",
                        # f"Parameters Difference bottom {k} Similarity",
                        # f"Parameters Difference random {k} Similarity",
                    ]
                )
                for name, diff_top in top_k_params_dict.items():
                    writer.writerow([name, (diff_top.sum() / diff_top.numel()).item()])

            os.makedirs(
                "code-region/{}/top{}".format(model_name, k),
                exist_ok=True,
            )
            for key, values in top_k_params_dict.items():
                # Convert boolean matrix to tensor of type byte
                save_path = os.path.join(
                    "code-region/{}/top{}".format(model_name, k),
                    "{}.pt".format(key),
                )
                # Save the tensor to a file using torch.save()
                torch.save(values, save_path)

            # os.makedirs(
            #     "core-accumulated-{}-grad-mul-param/{}/bottom{}".format(
            #         model_name, samples, k
            #     ),
            #     exist_ok=True,
            # )
            # for key, values in bottom_k_params_dict.items():
            #     save_path = os.path.join(
            #         "core-accumulated-{}-grad-mul-param/{}/bottom{}".format(
            #             model_name, samples, k
            #         ),
            #         "{}.pt".format(key),
            #     )
            #     torch.save(values, save_path)

            # if (
            #     samples == 100000
            # ):  # If random k has been saved, there is no need to filter again.
            #     continue

            # os.makedirs(
            #     "core-accumulated-{}-grad-mul-param/{}/random{}".format(
            #         model_name, samples, k
            #     ),
            #     exist_ok=True,
            # )
            # for key, values in random_k_params_dict.items():
            #     save_path = os.path.join(
            #         "core-accumulated-{}-grad-mul-param/{}/random{}".format(
            #             model_name, samples, k
            #         ),
            #         "{}.pt".format(key),
            #     )
            #     torch.save(values, save_path)


if __name__ == "__main__":
    fire.Fire(extract)
