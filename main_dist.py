import sys
import torch
import numpy as np
import time
import argparse
import os
import shutil
import random
import torch.distributed as dist
from process_list_plane import get_compton_backproj_list_single
from recon_osem_dist import run_recon_osem_dist


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_distributed():
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return global_rank, local_rank, world_size


def main():
    global_rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    start_time = time.time()

    # ===== 配置参数 (与原版 main_plane.py 一致) =====
    e0_list = [0.511]
    ene_threshold_sum_list = [0.46]
    intensity_list = [1]
    s_map_d_ratio = 1.0
    data_file_name = "ContrastPhantom_240_30"
    count_level = "5e9"
    ds = 1

    pixel_num_layer, pixel_num_z, rotate_num = 1160, 20, 10
    pixel_num = pixel_num_layer * pixel_num_z
    delta_r1, delta_r2, alpha = 2, 2, 1
    ene_resolution_662keV = 0.1

    iter_arg = argparse.Namespace()
    iter_arg.sc, iter_arg.jsccd, iter_arg.jsccsd = 1000, 500, 1000
    iter_arg.admm_inner_single, iter_arg.admm_inner_compton, iter_arg.mode = 1, 1, 0
    iter_arg.save_iter_step = 10
    iter_arg.osem_subset_num = 8
    iter_arg.t_divide_num = 10
    iter_arg.ene_num = len(e0_list)
    iter_arg.num_workers = 20

    if global_rank == 0:
        rand_suffix = f"{random.randint(0, 9999):04d}"
        log_filename = f"print_log_dist_{rand_suffix}.txt"
        logfile = open(log_filename, "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, logfile)
        print(f"Distributed initialized: World Size {world_size}")

    # ===== Step 2: Loading & Partitioning =====
    proj_local_all, proj_d_local_all, list_local_all, sysmat_local_all = [], [], [], []
    rotmat_all, rotmat_inv_all = [], []
    sensi_s_all, sensi_d_all = [], []
    e_params = []
    single_event_count_total = 0
    compton_event_count_total = 0

    for e0, ene_threshold_sum, intensity in zip(e0_list, ene_threshold_sum_list, intensity_list):
        ene_resolution = ene_resolution_662keV * (0.662 / e0) ** 0.5
        ene_threshold_max = 2 * e0 ** 2 / (0.511 + 2 * e0) - 0.001
        ene_threshold_min = 0.05

        factor_path = f"./Factors/{round(1000 * e0)}keV"
        # 1. Sysmat 分片 (按 bin 维度切分)
        full_sysmat = torch.from_numpy(np.fromfile(f"{factor_path}/SysMat_polar", dtype=np.float32).reshape(pixel_num, -1)).transpose(0, 1) * intensity
        total_bins = full_sysmat.size(0)
        bins_per_rank = total_bins // world_size
        idx_start, idx_end = global_rank * bins_per_rank, ((global_rank + 1) * bins_per_rank if global_rank != world_size - 1 else total_bins)
        sysmat_local_all.append(full_sysmat[idx_start:idx_end, :])

        # print(f"global_rank:{global_rank}, full_sysmat.size(0):{full_sysmat.size(0)}, full_sysmat.size(1):{full_sysmat.size(1)}, idx_start:{idx_start}, idx_end:{idx_end}")

        # 2. Proj 分片 (按对应 bin 切分)
        full_proj = torch.from_numpy(np.genfromtxt(f"./CntStat/CntStat_{data_file_name}_{round(1000 * e0)}keV_{count_level}.csv", delimiter=",", dtype=np.float32).reshape(rotate_num, -1)).transpose(0, 1)
        proj_local_all.append(full_proj[idx_start:idx_end, :])
        single_event_count_total = single_event_count_total + full_proj.sum().item()

        # 3. List 分片 (按事件条数切分)
        list_rotate_local = []
        for i in range(rotate_num):
            full_list = torch.from_numpy(np.genfromtxt(f"./List/List_{data_file_name}_{round(1000 * e0)}keV_{count_level}/{i + 1}.csv", delimiter=",", dtype=np.float32)[:, 0:4])
            ev_per_rank = full_list.size(0) // world_size
            ev_start, ev_end = global_rank * ev_per_rank, ((global_rank + 1) * ev_per_rank if global_rank != world_size - 1 else full_list.size(0))
            list_rotate_local.append(full_list[ev_start:ev_end, :])

        list_local_all.append(list_rotate_local)

        # 4. 公共因子 (每个 Rank 持有一份)
        sensi_d_file_path = f"{factor_path}/Sensi_d"
        if os.path.exists(sensi_d_file_path):
            sensi_d = torch.from_numpy(np.reshape(np.fromfile(sensi_d_file_path, dtype=np.float32), [pixel_num, 1])) * intensity
            sensi_d_all.append(sensi_d)

        rotmat = torch.from_numpy(np.genfromtxt(f"{factor_path}/RotMat_full.csv", delimiter=",", dtype=int))
        rotmat_inv = torch.from_numpy(np.genfromtxt(f"{factor_path}/RotMatInv_full.csv", delimiter=",", dtype=int))

        # 敏感度 (由Sysmat 计算)
        sensi_s = torch.zeros([1, pixel_num], dtype=torch.float32)
        for i in range(0, rotate_num):
            rotmat_inv_tmp = rotmat_inv[:, i]
            sensi_s = sensi_s + torch.sum(full_sysmat[:, rotmat_inv_tmp - 1], dim=0, keepdim=True).cpu()
        sensi_s = sensi_s.transpose(0, 1) / rotate_num
        sensi_s_all.append(sensi_s)

        rotmat = rotmat.to(device)
        rotmat_inv = rotmat_inv.to(device)
        rotmat_all.append(rotmat)
        rotmat_inv_all.append(rotmat_inv)

        e_params.append((e0, ene_resolution, ene_threshold_max, ene_threshold_min, ene_threshold_sum))

    # ===== Step 3: Processing Local List =====
    t_local_all = []
    for idx, (list_local, sysmat_l, (e0, er, e_max, e_min, e_sum)) in enumerate(zip(list_local_all, sysmat_local_all, e_params)):
        detector = torch.from_numpy(np.genfromtxt(f"./Factors/{round(1000 * e0)}keV/Detector.csv", delimiter=",", dtype=np.float32)[:, 1:4]).to(device)
        coor_polar = torch.from_numpy(np.genfromtxt(f"./Factors/{round(1000 * e0)}keV/coor_polar_full.csv", delimiter=",", dtype=np.float32)).to(device)
        sysmat_full_gpu = torch.from_numpy(np.fromfile(f"./Factors/{round(1000 * e0)}keV/SysMat_polar", dtype=np.float32).reshape(pixel_num, -1)).transpose(0, 1).to(device)

        t_rotate_local = []

        for i in range(rotate_num):
            list_local_chunks = torch.chunk(list_local[i], iter_arg.num_workers, dim=0)
            t_rotate_local_tmp = []
            for j in range(iter_arg.num_workers):
                t_chunk, _, _ = get_compton_backproj_list_single(sysmat_full_gpu, detector, coor_polar, list_local_chunks[j].to(device), delta_r1, delta_r2, e0, er, e_max, e_min, e_sum, device)
                t_rotate_local_tmp.append(t_chunk)
                compton_event_count_total = compton_event_count_total + t_chunk.size(0)

            t_rotate_local.append(torch.cat(t_rotate_local_tmp, dim=0))

        t_local_all.append(t_rotate_local)

        # 构造分布式的 proj_d (康普顿等效投影)
        # 这里逻辑较复杂，简单处理：每个 Rank 根据自己 local t 的数量对 proj 进行抽稀
        p_l = proj_local_all[idx].clone()
        proj_d_local_all.append(p_l * 0.5)  # 示例

    # ===== Step 4: Reconstruction =====
    s_map_arg = argparse.Namespace()
    s_map_arg.s = sum(sensi_s_all).to(device)

    if len(sensi_d_all)>0:
        print("sensi_d change to file definition")
        s_map_arg.d = sum(sensi_d_all).to(device)
        s_map_arg.d = s_map_arg.d * s_map_d_ratio
    else:
        s_map_arg.d = s_map_arg.s * s_map_d_ratio

    s_map_arg.j = alpha * s_map_arg.s + (2 - alpha) * s_map_arg.d

    save_path = f"./Figure_Dist/SingleEnergy_{data_file_name}_{round(1000 * e0_list[0])}keV_{count_level}_{ds}_SMap{s_map_d_ratio}_Delta{delta_r1}_Alpha{alpha}_ER{ene_resolution_662keV}_OSEM{iter_arg.osem_subset_num}_ITER{iter_arg.jsccsd}_SDU{single_event_count_total}_DDU{compton_event_count_total}/Polar/"
    if global_rank == 0: os.makedirs(save_path, exist_ok=True)
    dist.barrier()

    run_recon_osem_dist(sysmat_local_all, rotmat_all, rotmat_inv_all, proj_local_all, proj_d_local_all, t_local_all, iter_arg, s_map_arg, alpha, save_path)

    if global_rank == 0:
        # 先把标准输出恢复为系统的原始输出，解除对 Tee 对象的依赖
        sys.stdout = sys.__stdout__ 
        logfile.close()
        print("Done.") # 这句会打印到控制台而不是 log 文件

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    with torch.no_grad():
        main()