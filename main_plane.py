import sys
import torch
import numpy as np
from process_list_plane import get_coor_plane, get_compton_backproj_list_mp
from recon_osem_plane import run_recon_osem
import time
import argparse
import os
import shutil
import torch.multiprocessing as mp
import random
from torch.multiprocessing import Manager

# ===== Import model_denoiser Models =====
# from Models.dncnn import DnCNN
# from Models.tv import TV

# ===== Import ComptonGenerator Models =====
# from Models.ComptonGenerator.attention_unet import AttentionUNet


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


def main():
    global start_time
    start_time = time.time()

    # ===== Parse arguments =====
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    args = parser.parse_args()

    # ===== Multi Energy Input =====
    # e0_list = [0.218, 0.440]
    # ene_threshold_sum_list = [0.18, 0.40]
    # intensity_list = [0.1142, 0.261]

    # e0_list = [0.440]
    # ene_threshold_sum_list = [0.40]
    # intensity_list = [0.261]
    # intensity_list = [1]
    # s_map_d_ratio = 0.75
    # s_map_d_ratio = 0.6

    # e0_list = [0.218]
    # ene_threshold_sum_list = [0.18]
    # intensity_list = [0.1142]
    # intensity_list = [1]
    # s_map_d_ratio = 0.5

    # e0_list = [0.662]
    # ene_threshold_sum_list = [0.60]
    # intensity_list = [1]
    # s_map_d_ratio = 1

    e0_list = [0.511]
    ene_threshold_sum_list = [0.46]
    intensity_list = [1]
    s_map_d_ratio = 0.6
    s_map_d_ratio = 1

    data_file_name = "ContrastPhantom_240_30"
    count_level = "1e9"
    ds = 1

    # ===== System & FOV Parameters =====
    ene_resolution_662keV = 0.1
    pixel_num_layer = 1160
    pixel_num_z = 20
    rotate_num = 10
    pixel_num = pixel_num_layer * pixel_num_z

    delta_r1 = 2
    delta_r2 = 2
    alpha = 1

    # ===== Reconstruction Parameters =====
    iter_arg = argparse.ArgumentParser().parse_args()
    iter_arg.sc = 1000
    iter_arg.jsccd = 500
    iter_arg.jsccsd = 1000

    iter_arg.admm_inner_single = 1
    iter_arg.admm_inner_compton = 1
    iter_arg.mode = 0

    iter_arg.save_iter_step = 10
    iter_arg.osem_subset_num = 8
    iter_arg.t_divide_num = 1
    iter_arg.ene_num = len(e0_list)

    iter_arg.event_level = 2
    iter_arg.event_level = 2
    iter_arg.num_workers = 10

    # ===== Down Sampling Parameters =====
    flag_save_t = 0
    flag_save_s = 0
    flag_save_d = 0

    # ===== Denoise Net =====
    model_denoiser = None
    # model_denoiser = TV(weight=0.001, iter_num=1)
    # model_denoiser.eval()
    # model_denoiser = DnCNN(in_nc=1, out_nc=1, nc=64, nb=17, act_mode="R")
    # model_denoiser.load_state_dict(torch.load('./Models/DnCNN/dncnn_25.pth'))
    # model_denoiser.eval()

    # ===== ComptonGenerator Net =====
    model_compton_generator = None
    # model_compton_generator = AttentionUNet(in_channels=1, out_channels=1)

    # ===== Setup logging =====
    rand_suffix = f"{random.randint(0, 9999):04d}"  # 生成四位随机数
    log_filename = f"print_log_{rand_suffix}.txt"
    logfile = open(log_filename, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, logfile)

    print("=======================================")
    print("--------Step1: Checking Devices--------")
    if torch.cuda.is_available():
        print(f"CUDA is available, found {args.num_gpus} GPUs")
    else:
        print("CUDA is not available, running on CPU")
        args.num_gpus = 0

    # ====== Data List ======
    proj_all = []
    list_all = []
    sysmat_all = []
    detector_all = []
    coor_polar_all = []
    rotmat_all = []
    rotmat_inv_all = []

    sensi_s_all = []
    sensi_d_all = []
    e_params = []
    print("====================================")
    print("--------Step2: Loading Files--------")
    for e0, ene_threshold_sum, intensity in zip(
            e0_list, ene_threshold_sum_list, intensity_list
    ):
        # Ene Parameters
        ene_resolution = ene_resolution_662keV * (0.662 / e0) ** 0.5
        ene_threshold_max = 2 * e0 ** 2 / (0.511 + 2 * e0) - 0.001
        ene_threshold_min = 0.05

        # Path
        factor_file_path = f"{round(1000 * e0)}keV"
        data_file_path = f"{data_file_name}_{round(1000 * e0)}keV_{count_level}"

        # SysMat + Detector
        sysmat_file_path = f"./Factors/{factor_file_path}/SysMat_polar"
        detector_file_path = f"./Factors/{factor_file_path}/Detector.csv"
        sensi_s_file_path = f"./Factors/{factor_file_path}/Sensi_s"
        sensi_d_file_path = f"./Factors/{factor_file_path}/Sensi_d"
        coor_polar_file_path = f"./Factors/{factor_file_path}/coor_polar_full.csv"
        rotmat_file_path = f"./Factors/{factor_file_path}/RotMat_full.csv"
        rotmat_inv_file_path = f"./Factors/{factor_file_path}/RotMatInv_full.csv"

        sysmat = torch.from_numpy(np.reshape(np.fromfile(sysmat_file_path, dtype=np.float32), [pixel_num, -1])).transpose(0, 1) * intensity
        detector = torch.from_numpy(np.genfromtxt(detector_file_path, delimiter=",", dtype=np.float32)[:, 1:4])
        coor_polar = torch.from_numpy(np.genfromtxt(coor_polar_file_path, delimiter=",", dtype=np.float32))
        rotmat = torch.from_numpy(np.genfromtxt(rotmat_file_path, delimiter=",", dtype=int))
        rotmat_inv = torch.from_numpy(np.genfromtxt(rotmat_inv_file_path, delimiter=",", dtype=int))

        if os.path.exists(sensi_s_file_path):
            sensi_s = torch.from_numpy(np.reshape(np.fromfile(sensi_s_file_path, dtype=np.float32), [pixel_num, 1])) * intensity
            sensi_s_all.append(sensi_s)

        if os.path.exists(sensi_d_file_path):
            sensi_d = torch.from_numpy(np.reshape(np.fromfile(sensi_d_file_path, dtype=np.float32), [pixel_num, 1])) * intensity
            sensi_d_all.append(sensi_d)

        # Data files
        proj_file_path = f"./CntStat/CntStat_{data_file_path}.csv"
        list_file_path = f"./List/List_{data_file_path}/"
        proj = torch.from_numpy(np.reshape(np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32), [rotate_num, -1])).transpose(0,1)
        list_origin = []
        for i in range(0, rotate_num):
            list_file_path_tmp = list_file_path + str(i+1) + ".csv"
            list_origin_tmp = torch.from_numpy(np.genfromtxt(list_file_path_tmp, delimiter=",", dtype=np.float32)[:, 0:4])
            list_origin.append(list_origin_tmp)

        print(f"Loaded energy {e0:.3f} MeV")

        # Downsampling
        if ds * intensity < 0.9999:
            print("--------Data Downsampling--------")
            for i in range(0, rotate_num):
                # porj
                proj_tmp = proj[:, i]
                proj_ds_tmp = proj[:, i] * 0
                proj_s_index_tmp = torch.tensor([i for i in range(proj_tmp.size(0)) for _ in range(round(proj_tmp[i].item()))])
                indices_tmp = torch.randperm(proj_s_index_tmp.size(0))
                selected_indices_tmp = indices_tmp[0:int(torch.round(proj_tmp.sum() * ds).item())]
                proj_s_index_ds_tmp = proj_s_index_tmp[selected_indices_tmp]
                for j in range(0, proj_ds_tmp.size(dim=0)):
                    proj_ds_tmp[j] = (proj_s_index_ds_tmp == j).sum()
                proj[:, i] = proj_ds_tmp

                # list
                list_origin_tmp = list_origin[i]
                indices_tmp = torch.randperm(list_origin_tmp.size(0))
                selected_indices_tmp = indices_tmp[0:int(list_origin_tmp.size(0) * ds)]
                list_origin_tmp = list_origin_tmp[selected_indices_tmp, :]
                list_origin[i] = list_origin_tmp

        # 保存
        proj_all.append(proj)
        list_all.append(list_origin)
        sysmat_all.append(sysmat)
        detector_all.append(detector)
        coor_polar_all.append(coor_polar)
        rotmat_all.append(rotmat)
        rotmat_inv_all.append(rotmat_inv)
        e_params.append((e0, ene_resolution, ene_threshold_max, ene_threshold_min, ene_threshold_sum))

    # ===== Step3: Processing List =====
    print("==================================================")
    print("--------Step3: Processing List (Multi-GPU)--------")
    t_all = []
    proj_d_all = []
    single_event_count_total = 0
    compton_event_count_total = 0

    s_map_arg = argparse.ArgumentParser().parse_args()
    s_map_arg.s = torch.zeros([1, pixel_num], dtype=torch.float32)
    s_map_arg.d = torch.zeros([1, pixel_num], dtype=torch.float32)

    for idx, (proj, list_origin, sysmat, detector, coor_polar, rotmat_inv, (e0, ene_resolution, ene_threshold_max, ene_threshold_min, ene_threshold_sum)) in enumerate(
        zip(proj_all, list_all, sysmat_all, detector_all, coor_polar_all, rotmat_inv_all, e_params)
    ):
        print(f"Processing energy {e0:.3f} MeV ...")

        # load model
        if model_compton_generator is not None:
            model_compton_generator.load_state_dict(torch.load(f"./Models/ComptonGenerator/AttentionUNet/attention_unet_model_params_{round(1000 * e0)}keV.pth"))
            model_compton_generator.eval()

        t = []
        size_t = 0
        compton_event_count_list = torch.zeros(size=[rotate_num, 1], dtype=torch.int)
        for i in range(0, rotate_num):
            # Split list data
            if args.num_gpus > 1:
                chunks = torch.chunk(list_origin[i], args.num_gpus, dim=0)
            else:
                chunks = [list_origin[i]]

            with Manager() as manager:
                result_dict = manager.dict()
                processes = []

                for rank in range(args.num_gpus):
                    p = mp.Process(
                        target=get_compton_backproj_list_mp,
                        args=(rank, args.num_gpus, sysmat, detector, coor_polar, chunks[rank],
                                delta_r1, delta_r2, e0, ene_resolution,
                                ene_threshold_max, ene_threshold_min, ene_threshold_sum,
                                result_dict, iter_arg.num_workers, start_time, flag_save_t, model_compton_generator)
                    )
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

                t_results = []
                for rank in range(args.num_gpus):
                    if rank in result_dict:
                        t_results.append(result_dict[rank])
                        print(f"Collected result from rank {rank}")
                    else:
                        print(f"Warning: No result found for rank {rank}")

                if t_results:
                    t_tmp = torch.cat(t_results, dim=0)
                    compton_event_count_list[i] = t_tmp.size(0)
                    size_t = size_t + t_tmp.element_size() * t_tmp.nelement()
                    t.append(t_tmp)
                    print("Rotate Num", str(i + 1), "ends, time used:", time.time() - start_time, "s")
                else:
                    print("Error: No results collected from any GPU")
                    return

        # create a proj that has an equal count
        proj_d = torch.zeros(size=proj.size(), dtype=torch.float32)
        for i in range(0, rotate_num):
            proj_tmp = proj[:, i]
            proj_s_index_tmp = torch.tensor(
                [i for i in range(proj_tmp.size(0)) for _ in range(round(proj_tmp[i].item()))])
            indices_tmp = torch.randperm(proj_s_index_tmp.size(0))
            selected_indices_tmp = indices_tmp[0:compton_event_count_list[i]]
            proj_d_index_tmp = proj_s_index_tmp[selected_indices_tmp]
            for j in range(0, proj_d.size(dim=0)):
                proj_d[j, i] = (proj_d_index_tmp == j).sum()

        single_event_count = round(proj.sum().item())
        compton_event_count = round(proj_d.sum().item())
        print(f"[Energy {e0:.3f}] Single events = {single_event_count}, Compton events = {compton_event_count}")
        print(f"[Energy {e0:.3f}] The size of t is {size_t / (1024 ** 3):.2f} GB")

        t_all.append(t)
        proj_d_all.append(proj_d)

        for i in range(0, rotate_num):
            rotmat_inv_tmp = rotmat_inv[:, i]
            s_map_arg.s = s_map_arg.s + torch.sum(sysmat[:, rotmat_inv_tmp - 1], dim=0, keepdim=True).cpu()
        s_map_arg.s = s_map_arg.s.transpose(0, 1) / rotate_num
        s_map_arg.d = s_map_arg.s * compton_event_count / single_event_count

        single_event_count_total += single_event_count
        compton_event_count_total += compton_event_count

    if len(sensi_s_all) > 0:
        print("sensi_s change to file definition")
        s_map_arg.s = sum(sensi_s_all)

    if len(sensi_d_all)>0:
        print("sensi_d change to file definition")
        s_map_arg.d = sum(sensi_d_all)
        s_map_arg.d = s_map_arg.d * s_map_d_ratio
        # s_map_arg.d = s_map_arg.d * torch.sum(s_map_arg.s) / torch.sum(s_map_arg.d) * compton_event_count_total / single_event_count_total

    # ===== Step4: Reconstruction =====
    print("===========================================")
    print("--------Step4: Image Reconstruction--------")

    if flag_save_s == 1:
        with open("sensitivity_s", "w") as file:
            s_map_arg.s.cpu().numpy().astype('float32').tofile(file)

    # calculate sensi_d (only for square)
    if flag_save_d == 1:
        sensi_d = 0 * s_map_arg.s
        for t in t_all:
            sensi_d_tmp = torch.sum(t, dim=0, keepdim=True).transpose(0, 1)
            sensi_d_tmp = sensi_d_tmp * torch.sum(s_map_arg.s) / torch.sum(sensi_d_tmp) * compton_event_count_total / single_event_count_total
            sensi_d = sensi_d + sensi_d_tmp
        with open("Sensi_d", "w") as file:
            sensi_d.cpu().numpy().astype('float32').tofile(file)

    torch.cuda.empty_cache()

    if len(e0_list) == 1:
        save_path = f"./Figure/SingleEnergy_{data_file_name}_{round(1000 * e0_list[0])}keV_{count_level}_{ds}_SMap{s_map_d_ratio}_Delta{delta_r1}_Alpha{alpha}_ER{ene_resolution_662keV}_OSEM{iter_arg.osem_subset_num}_ITER{iter_arg.jsccsd}_SDU{single_event_count_total}_DDU{compton_event_count_total}/Polar/"
    else:
        e0_list_str = " ".join(str(round(e0 * 1000)) for e0 in e0_list)
        save_path = f"./Figure/MultiEnergy_{data_file_name}_({e0_list_str})keV_{count_level}_{ds}_SMap{s_map_d_ratio}_Delta{delta_r1}_Alpha{alpha}_ER{ene_resolution_662keV}_OSEM{iter_arg.osem_subset_num}_ITER{iter_arg.jsccsd}_SDU{single_event_count_total}_DDU{compton_event_count_total}/Polar/"
    os.makedirs(save_path, exist_ok=True)

    run_recon_osem(sysmat_all, rotmat_all, rotmat_inv_all, proj_all, proj_d_all, t_all, iter_arg, s_map_arg, alpha, save_path, args.num_gpus, model_denoiser)

    print(f"\nTotal time used: {time.time() - start_time:.2f}s")

    logfile.close()
    sys.stdout = sys.__stdout__
    final_log_name = "print_log.txt"
    shutil.move(log_filename, os.path.join(save_path, final_log_name))


if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    with torch.no_grad():
        main()
