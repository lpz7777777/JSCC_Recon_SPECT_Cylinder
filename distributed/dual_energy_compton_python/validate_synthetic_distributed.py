"""Two-rank CPU/GLOO or CUDA/NCCL MLEM validation against a serial reference."""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from reconstruction import run_compton_and_joint_mlem_dist, run_single_mlem_dist


def reference_mlem(sysmat, projection, rotmat, rotmat_inv, sensitivity, iterations, background):
    image = torch.ones((sysmat.size(1), 1), dtype=torch.float32)
    rotate_num = rotmat.size(1)
    for _ in range(iterations):
        weight = torch.zeros_like(image)
        for view in range(rotate_num):
            rotated = torch.index_select(image, 0, rotmat[:, view] - 1)
            forward = torch.matmul(sysmat, rotated) + background[:, view:view + 1] * rotate_num
            local = torch.matmul(
                sysmat.transpose(0, 1),
                projection[:, view:view + 1] / forward.clamp_min(1.0e-12),
            )
            weight += torch.index_select(local, 0, rotmat_inv[:, view] - 1)
        image = image * weight / sensitivity
    return image


def reference_compton_joint(
    sysmat, projection, rotmat, rotmat_inv, event_blocks, sensi_s, sensi_d, iterations
):
    image_d = torch.ones((sysmat.size(1), 1), dtype=torch.float32)
    image_j = torch.ones_like(image_d)
    for _ in range(iterations):
        weight_d = torch.zeros_like(image_d)
        weight_j = torch.zeros_like(image_j)
        for view in range(rotmat.size(1)):
            ids = rotmat[:, view] - 1
            inverse_ids = rotmat_inv[:, view] - 1
            rotated_d = torch.index_select(image_d, 0, ids)
            rotated_j = torch.index_select(image_j, 0, ids)
            single = torch.matmul(
                sysmat.transpose(0, 1),
                projection[:, view:view + 1]
                / torch.matmul(sysmat, rotated_j).clamp_min(1.0e-12),
            )
            weight_j += torch.index_select(single, 0, inverse_ids)
            for response in event_blocks[view]:
                local_d = torch.matmul(
                    response.transpose(0, 1),
                    1.0 / torch.matmul(response, rotated_d).clamp_min(1.0e-12),
                )
                local_j = torch.matmul(
                    response.transpose(0, 1),
                    1.0 / torch.matmul(response, rotated_j).clamp_min(1.0e-12),
                )
                weight_d += torch.index_select(local_d, 0, inverse_ids)
                weight_j += torch.index_select(local_j, 0, inverse_ids)
        image_d = image_d * weight_d / sensi_d
        image_j = image_j * weight_j / (sensi_s + sensi_d)
    return image_d, image_j


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = os.environ.get("JSCC_TEST_BACKEND", "gloo")
    if backend == "nccl":
        device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device)
        torch.set_default_device(f"cuda:{device}")
    elif backend != "gloo":
        raise ValueError("JSCC_TEST_BACKEND must be gloo or nccl")
    dist.init_process_group(backend, init_method=os.environ.get("JSCC_TEST_INIT_METHOD", "env://"),
                            rank=rank, world_size=world_size)
    if world_size != 2:
        raise ValueError("Run this validation with torchrun --nproc_per_node=2")

    sysmat = torch.tensor(
        [[1.0, 0.2, 0.1], [0.1, 0.8, 0.2], [0.3, 0.1, 0.9], [0.5, 0.4, 0.2]],
        dtype=torch.float32,
    )
    rotmat = torch.tensor([[1, 2], [2, 3], [3, 1]], dtype=torch.long)
    rotmat_inv = torch.argsort(rotmat - 1, dim=0) + 1
    true_image = torch.tensor([[2.0], [4.0], [7.0]])
    background = torch.tensor(
        [[0.02, 0.03], [0.01, 0.04], [0.05, 0.02], [0.03, 0.01]],
        dtype=torch.float32,
    )
    projection = torch.empty((4, 2), dtype=torch.float32)
    for view in range(2):
        rotated = torch.index_select(true_image, 0, rotmat[:, view] - 1)
        projection[:, view] = (
            torch.matmul(sysmat, rotated).squeeze(1) / 2 + background[:, view]
        )

    base_sensitivity = sysmat.sum(dim=0, keepdim=True).transpose(0, 1)
    sensitivity = torch.zeros((3, 1), dtype=torch.float32)
    for view in range(2):
        sensitivity += torch.index_select(base_sensitivity, 0, rotmat_inv[:, view] - 1)
    sensitivity /= 2

    start = rank * 2
    end = start + 2
    result = run_single_mlem_dist(
        "synthetic",
        sysmat[start:end],
        projection[start:end],
        rotmat,
        rotmat_inv,
        sensitivity,
        iterations=4,
        save_step=1,
        rank=rank,
        additive_background=background[start:end],
    )
    expected = reference_mlem(
        sysmat, projection, rotmat, rotmat_inv, sensitivity, 4, background
    )
    torch.testing.assert_close(result.image, expected, rtol=2.0e-6, atol=2.0e-6)
    images = [torch.empty_like(result.image) for _ in range(world_size)]
    dist.all_gather(images, result.image)
    torch.testing.assert_close(images[0], images[1], rtol=0.0, atol=0.0)

    event_rows = [
        torch.tensor(
            [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.4, 0.3, 0.3]]
        ),
        torch.tensor(
            [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]
        ),
    ]
    local_event_blocks = [
        [rows[rank * 2:(rank + 1) * 2].contiguous()] for rows in event_rows
    ]
    sensi_d = torch.tensor([[1.2], [1.1], [1.3]], dtype=torch.float32)
    result_d, result_j = run_compton_and_joint_mlem_dist(
        sysmat[start:end],
        projection[start:end] - background[start:end],
        rotmat,
        rotmat_inv,
        local_event_blocks,
        sensitivity,
        sensi_d,
        iterations=3,
        save_step=1,
        rank=rank,
    )
    expected_d, expected_j = reference_compton_joint(
        sysmat,
        projection - background,
        rotmat,
        rotmat_inv,
        [[rows] for rows in event_rows],
        sensitivity,
        sensi_d,
        iterations=3,
    )
    torch.testing.assert_close(result_d.image, expected_d, rtol=2.0e-6, atol=2.0e-6)
    torch.testing.assert_close(result_j.image, expected_j, rtol=2.0e-6, atol=2.0e-6)
    if rank == 0:
        print(f"SYNTHETIC DISTRIBUTED VALIDATION PASSED ({backend})", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
