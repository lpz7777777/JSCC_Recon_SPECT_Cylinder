import torch
import numpy as np
import time
import os
import torch.multiprocessing as mp

def get_weight_single(sysmat_list_tmp, proj_list_tmp, img_tmp):
    # get the weight of single events
    return torch.matmul(sysmat_list_tmp.transpose(0, 1), proj_list_tmp / torch.matmul(sysmat_list_tmp, img_tmp))


def get_weight_compton(t_block, img):
    # get the weight of Compton events
    device = img.device
    t_gpu = t_block.to(device, non_blocking=True)

    weight = torch.matmul(t_gpu.transpose(0, 1), 1 / (torch.matmul(t_gpu, img)))
    if t_block.device.type == 'cpu':
        del t_gpu

    return weight


def osem_bin_mode(sysmat_list_all, proj_list_all, rotmat_all, rotmat_inv_all, img, s_map, rotate_num, model_denoiser=None):
    n = int(img.shape[0] ** 0.5)
    for sysmat_list, proj_list in zip(sysmat_list_all, proj_list_all):
        weight = 0 * img
        for i in range(0, rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_list, proj_list, rotmat_all, rotmat_inv_all):
                img_rotate = torch.index_select(img, 0, rotmat[:,i] - 1)
                weight_tmp = get_weight_single(sysmat, proj[:, i].unsqueeze(1) , img_rotate)
                weight = weight + torch.index_select(weight_tmp, 0, rotmat_inv[:, i] - 1)
        img = img * weight / s_map

        if model_denoiser is not None:
            img_2d = img.view(1, 1, n, n)
            img = model_denoiser(img_2d).view(n*n, 1)
            img = torch.clamp(img, min=0)

    return img

def osem_list_mode(t_list_all, rotmat_all, rotmat_inv_all, img, s_map, rotate_num, model_denoiser=None):
    n = int(img.shape[0] ** 0.5)
    for t_list in t_list_all:
        weight_compton = 0 * img
        for i in range(0, rotate_num):
            for t in t_list[i]:
                for t_tmp, rotmat, rotmat_inv in zip(t, rotmat_all, rotmat_inv_all):
                    img_rotate = torch.index_select(img, 0, rotmat[:, i] - 1)
                    weight_tmp = get_weight_compton(t_tmp, img_rotate)
                    weight_compton = weight_compton + torch.index_select(weight_tmp, 0, rotmat_inv[:, i] - 1)
        img = img * weight_compton / s_map

        if model_denoiser is not None:
            img_2d = img.view(1, 1, n, n)
            img = model_denoiser(img_2d).view(n*n, 1)
            img = torch.clamp(img, min=0)

    return img

def osem_list_mode_mp(t_list_all, rotmat_all, rotmat_inv_all, iter_num, rank, img_queue, weight_queue, rotate_num, model_denoiser=None):
    with torch.no_grad():
        print(f"List Mode OSEM Rank{rank} Starts")
        for id_iter in range(iter_num):
            for t_list in t_list_all:
                img = img_queue.get()
                weight_compton = 0 * img
                img = img.to(f"cuda:{rank}")

                for i in range(0, rotate_num):
                    for t in t_list[i]:
                        for t_tmp, rotmat, rotmat_inv in zip(t, rotmat_all, rotmat_inv_all):
                            img_rotate = torch.index_select(img, 0, (rotmat[:, i] - 1).to(f"cuda:{rank}"))
                            weight_tmp = get_weight_compton(t_tmp, img_rotate).to("cuda:0")
                            weight_compton = weight_compton + torch.index_select(weight_tmp, 0, rotmat_inv[:, i] - 1)

                weight_queue.put(weight_compton)

def osem_joint_mode(sysmat_list_all, proj_list_all, t_list_all, rotmat_all, rotmat_inv_all, img, s_map, alpha, rotate_num, model_denoiser=None):
    n = int(img.shape[0] ** 0.5)
    for sysmat_list, proj_list, t_list in zip(sysmat_list_all, proj_list_all, t_list_all):
        weight_compton = 0 * img
        weight_single = 0 * img
        for i in range(0, rotate_num):
            for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_list, proj_list, rotmat_all, rotmat_inv_all):
                img_rotate = torch.index_select(img, 0, rotmat[:,i] - 1)
                weight_single_tmp = get_weight_single(sysmat, proj[:, i].unsqueeze(1) , img_rotate)
                weight_single = weight_single + torch.index_select(weight_single_tmp, 0, rotmat_inv[:, i] - 1)

            for t in t_list[i]:
                for t_tmp, rotmat, rotmat_inv in zip(t, rotmat_all, rotmat_inv_all):
                    img_rotate = torch.index_select(img, 0, rotmat[:, i] - 1)
                    weight_compton_tmp = get_weight_compton(t_tmp, img_rotate)
                    weight_compton = weight_compton + torch.index_select(weight_compton_tmp, 0, rotmat_inv[:, i] - 1)

        weight = (2 - alpha) * weight_compton + alpha * weight_single
        img = img * weight / s_map

        if model_denoiser is not None:
            img_2d = img.view(1, 1, n, n)
            img = model_denoiser(img_2d).view(n*n, 1)
            img = torch.clamp(img, min=0)

    return img


def save_img(img_sc, img_scd, img_jsccd, img_jsccsd, img_sc_iter, img_scd_iter, img_jsccd_iter, img_jsccsd_iter, iter_arg, save_path):
    with open(save_path + "Image_SC", "wb") as file:
        img_sc.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_SCD", "wb") as file:
        img_scd.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_JSCCD", "wb") as file:
        img_jsccd.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_JSCCSD", "wb") as file:
        img_jsccsd.cpu().numpy().astype('float32').tofile(file)

    with open(save_path + "Image_SC_Iter_%d_%d" % (iter_arg.sc, iter_arg.sc / iter_arg.save_iter_step), "wb") as file:
        img_sc_iter.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_SCD_Iter_%d_%d" % (iter_arg.jsccd, iter_arg.jsccd / iter_arg.save_iter_step), "wb") as file:
        img_scd_iter.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_JSCCD_Iter_%d_%d" % (iter_arg.jsccd, iter_arg.jsccd / iter_arg.save_iter_step), "wb") as file:
        img_jsccd_iter.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_JSCCSD_Iter_%d_%d" % (iter_arg.jsccsd, iter_arg.jsccsd / iter_arg.save_iter_step), "wb") as file:
        img_jsccsd_iter.cpu().numpy().astype('float32').tofile(file)

    file.close()


def get_gpu_memory_usage(num_gpus):
    # return every gpu memory condition
    usage = {}
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"cuda:{i}")
        print(f"allocated_GB:{round(allocated, 2)}")
        print(f"allocated_GB:{round(allocated, 2)}")


def run_recon_osem(sysmat_all, rotmat_all, rotmat_inv_all, proj_all, proj_d_all, t_all, iter_arg, s_map_arg, alpha, save_path, num_gpus, model_denoiser=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pixel_num = sysmat_all[0].size(1)
    rotate_num = rotmat_all[0].size(1)

    # ===== Init Images =====
    img_sc = torch.ones([pixel_num, 1], dtype=torch.float32).to("cuda:0", non_blocking=True)
    img_scd = torch.ones([pixel_num, 1], dtype=torch.float32).to("cuda:0", non_blocking=True)
    img_jsccd = torch.ones([pixel_num, 1], dtype=torch.float32).to("cuda:0", non_blocking=True)
    img_jsccsd = torch.ones([pixel_num, 1], dtype=torch.float32).to("cuda:0", non_blocking=True)

    img_sc_iter = torch.ones([round(iter_arg.sc / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_scd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_jsccd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_jsccsd_iter = torch.ones([round(iter_arg.jsccsd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)

    # ===== Process t/proj/sysmat =====
    sysmat_list_all = [[[] for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    proj_list_all = [[[] for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    proj_d_list_all = [[[] for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    proj_d_list_all = [[[] for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.osem_subset_num)]
    # rotmat_all_mp = [[[] for _ in range(iter_arg.ene_num)] for _ in range(num_gpus)]
    # rotmat_inv_all_mp = [[[] for _ in range(iter_arg.ene_num)] for _ in range(num_gpus)]
    if num_gpus == 1:
        t_list_all = [[[[[] for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.t_divide_num)] for _ in range(rotate_num)] for _ in range(iter_arg.osem_subset_num)]
    else:
        t_list_all = [[[[[[] for _ in range(iter_arg.ene_num)] for _ in range(iter_arg.t_divide_num)] for _ in range(rotate_num)] for _ in range(iter_arg.osem_subset_num)] for _ in range(num_gpus)]

    cpnum_list = torch.arange(0, proj_all[0].size(dim=0))
    random_id = torch.randperm(proj_all[0].size(dim=0))
    cpnum_list = cpnum_list[random_id]
    cpnum_list = list(torch.chunk(cpnum_list, iter_arg.osem_subset_num, dim=0))

    GPU_T_BUDGET_BYTES = 40 * (1024 ** 3)

    # 记录每个 GPU 当前已使用的 t 显存量
    gpu_vram_used = [0] * num_gpus

    for e, (sysmat, proj, proj_d, t) in enumerate(zip(sysmat_all, proj_all, proj_d_all, t_all)):
        if num_gpus == 1:
            # 单 GPU 逻辑
            for i in range(0, rotate_num):
                t_chunks = list(torch.chunk(t[i], iter_arg.osem_subset_num, dim=0))
                for j in range(0, iter_arg.osem_subset_num):
                    t_sub_chunks = list(torch.chunk(t_chunks[j], iter_arg.t_divide_num, dim=0))
                    for k in range(0, iter_arg.t_divide_num):
                        chunk = t_sub_chunks[k]
                        size_bytes = chunk.nelement() * chunk.element_size()
                        if gpu_vram_used[0] + size_bytes < GPU_T_BUDGET_BYTES:
                            # 预算内：直接存入 GPU
                            t_list_all[j][i][k][e] = chunk.to("cuda:0", non_blocking=True)
                            gpu_vram_used[0] += size_bytes
                        else:
                            # 预算外：存入 CPU pin_memory
                            t_list_all[j][i][k][e] = chunk.pin_memory()

            rotmat_all[e] = rotmat_all[e].to("cuda:0", non_blocking=True)
            rotmat_inv_all[e] = rotmat_inv_all[e].to("cuda:0", non_blocking=True)

        else:
            for i in range(0, rotate_num):
                t_chunks = list(torch.chunk(t[i], num_gpus, dim=0))
                for gpu_id in range(0, num_gpus):
                    t_gpu_chunks = list(torch.chunk(t_chunks[gpu_id], iter_arg.osem_subset_num, dim=0))
                    for j in range(0, iter_arg.osem_subset_num):
                        t_sub_chunks = list(torch.chunk(t_gpu_chunks[j], iter_arg.t_divide_num, dim=0))
                        for k in range(0, iter_arg.t_divide_num):
                            chunk = t_sub_chunks[k]
                            size_bytes = chunk.nelement() * chunk.element_size()
                            if gpu_vram_used[gpu_id] + size_bytes < GPU_T_BUDGET_BYTES:
                                t_list_all[gpu_id][j][i][k][e] = chunk.to(f"cuda:{gpu_id}", non_blocking=True)
                                gpu_vram_used[gpu_id] += size_bytes
                            else:
                                t_list_all[gpu_id][j][i][k][e] = chunk.pin_memory()

            rotmat_all[e] = rotmat_all[e].to("cuda:0", non_blocking=True)
            rotmat_inv_all[e] = rotmat_inv_all[e].to("cuda:0", non_blocking=True)

        print(f"显存加载完成。各 GPU 常驻显存百分比: {[round(u / GPU_T_BUDGET_BYTES * 100, 2) for u in gpu_vram_used]} %")

        # ===== proj =====
        for i in range(0, iter_arg.osem_subset_num):
            sysmat_list_all[i][e] = sysmat[cpnum_list[i], :].to("cuda:0", non_blocking=True)
            proj_list_all[i][e] = proj[cpnum_list[i], :].to("cuda:0", non_blocking=True)
            proj_d_list_all[i][e] = proj_d[cpnum_list[i], :].to("cuda:0", non_blocking=True)

    del sysmat_all, t_all

    # ===== Sensitivity to GPU =====

    s_map_arg.s = s_map_arg.s.to("cuda:0", non_blocking=True)
    s_map_arg.d = s_map_arg.d.to("cuda:0", non_blocking=True)
    s_map_arg.j = alpha * s_map_arg.s + (2 - alpha) * s_map_arg.d

    # ===== model_denoiser to GPU =====
    if model_denoiser is not None:
        model_denoiser.to("cuda:0")

    # ===== Start Iteration =====
    get_gpu_memory_usage(num_gpus)
    time_start = time.time()

    # self-collimation
    print("Self-Collimation OSEM starts")
    id_save = 0
    for id_iter_sc in range(iter_arg.sc):
        img_sc = osem_bin_mode(sysmat_list_all, proj_list_all, rotmat_all, rotmat_inv_all, img_sc, s_map_arg.s, rotate_num, model_denoiser)
        if (id_iter_sc + 1) % iter_arg.save_iter_step == 0:
            img_sc_iter[id_save, :] = torch.squeeze(img_sc).cpu()
            id_save += 1
            print("SC Iteration ", str(id_iter_sc + 1), " ends, time used:", time.time() - time_start, "s")

    print("Self-Collimation OSEM ends, time used:", time.time() - time_start)
    torch.cuda.empty_cache()

    # sc-d
    print("SC-D OSEM starts")
    id_save = 0
    for id_iter_scd in range(iter_arg.jsccd):
        img_scd = osem_bin_mode(sysmat_list_all, proj_d_list_all, rotmat_all, rotmat_inv_all, img_scd, s_map_arg.s, rotate_num, model_denoiser)
        if (id_iter_scd + 1) % iter_arg.save_iter_step == 0:
            img_scd_iter[id_save, :] = torch.squeeze(img_scd).cpu()
            id_save += 1
            print("SC-D Iteration ", str(id_iter_scd + 1), " ends, time used:", time.time() - time_start, "s")

    print("SC-D OSEM ends, time used:", time.time() - time_start)
    torch.cuda.empty_cache()

    if num_gpus == 1:
        # ========== jscc-d ==========
        print("JSCC-D OSEM starts")
        id_save = 0
        for id_iter_jsccd in range(iter_arg.jsccd):
            img_jsccd = osem_list_mode(t_list_all, rotmat_all, rotmat_inv_all, img_jsccd, s_map_arg.d, rotate_num, model_denoiser)
            if (id_iter_jsccd + 1) % iter_arg.save_iter_step == 0:
                img_jsccd_iter[id_save, :] = torch.squeeze(img_jsccd).cpu()
                id_save += 1
                print("JSCC-D Iteration ", str(id_iter_jsccd + 1), " ends, time used:", time.time() - time_start, "s")

        print("JSCC-D OSEM ends, time used:", time.time() - time_start)
        torch.cuda.empty_cache()

        # ========== jscc-sd ==========
        print("JSCC-SD OSEM starts")
        id_save = 0
        for id_iter_jsccsd in range(iter_arg.jsccsd):
            img_jsccsd = osem_joint_mode(sysmat_list_all, proj_list_all, t_list_all, rotmat_all, rotmat_inv_all, img_jsccsd, s_map_arg.j, alpha, rotate_num, model_denoiser)
            if (id_iter_jsccsd + 1) % iter_arg.save_iter_step == 0:
                img_jsccsd_iter[id_save, :] = torch.squeeze(img_jsccsd).cpu()
                id_save += 1
                print("JSCC-SD Iteration ", str(id_iter_jsccsd + 1), " ends, time used:", time.time() - time_start, "s")

        print("JSCC-SD OSEM ends, time used:", time.time() - time_start)
        torch.cuda.empty_cache()

    else:
        # prepare queue for mp
        img_queue = mp.Queue()
        weight_queue = mp.Queue()

        # ========== jscc-d ==========
        print("JSCC-D OSEM starts (With Multiprocessing)")
        id_save = 0
        processes = []

        for i in range(num_gpus):
            p = mp.Process(target=osem_list_mode_mp, args=(t_list_all[i], rotmat_all, rotmat_inv_all, iter_arg.jsccd, i, img_queue, weight_queue, rotate_num, model_denoiser))
            p.start()
            processes.append(p)

        for id_iter_jsccd in range(iter_arg.jsccd):
            for i in range(iter_arg.osem_subset_num):
                weight_compton = 0 * img_jsccd
                for j in (range(num_gpus)):
                    img_queue.put(img_jsccd)
                for j in range(num_gpus):
                    weight_compton_tmp = weight_queue.get()
                    weight_compton = weight_compton + weight_compton_tmp

                img_jsccd = img_jsccd * weight_compton / s_map_arg.d
                torch.cuda.empty_cache()

            if (id_iter_jsccd + 1) % iter_arg.save_iter_step == 0:
                img_jsccd_iter[id_save, :] = torch.squeeze(img_jsccd).cpu()
                id_save += 1
                print("JSCC-D Iteration ", str(id_iter_jsccd + 1), " ends, time used:", time.time() - time_start, "s")

        for p in processes:
            p.join()
        processes.clear()

        print("JSCC-D OSEM ends, time used:", time.time() - time_start)

        # ========== jscc-sd ==========
        if iter_arg.mode == 0:
            # osem
            print("JSCC-SD OSEM starts (With Multiprocessing)")
            id_save = 0
            processes = []

            for j in range(num_gpus):
                p = mp.Process(target=osem_list_mode_mp, args=(t_list_all[j], rotmat_all, rotmat_inv_all, iter_arg.jsccsd, j, img_queue, weight_queue, rotate_num, model_denoiser))
                p.start()
                processes.append(p)

            for id_iter_jsccsd in range(iter_arg.jsccsd):
                for sysmat_list, proj_list in zip(sysmat_list_all, proj_list_all):
                    weight_compton = 0 * img_jsccsd
                    weight_single = 0 * img_jsccsd
                    for j in range(num_gpus):
                        img_queue.put(img_jsccsd)
                    for j in range(num_gpus):
                        weight_compton_tmp = weight_queue.get()
                        weight_compton = weight_compton + weight_compton_tmp

                    for i in range(0, rotate_num):
                        for sysmat, proj, rotmat, rotmat_inv in zip(sysmat_list, proj_list, rotmat_all, rotmat_inv_all):
                            img_rotate = torch.index_select(img_jsccsd, 0, rotmat[:, i] - 1)
                            weight_single_tmp = get_weight_single(sysmat, proj[:, i].unsqueeze(1), img_rotate)
                            weight_single = weight_single + torch.index_select(weight_single_tmp, 0, rotmat_inv[:, i] - 1)

                    weight = alpha * weight_single + (2 - alpha) * weight_compton
                    img_jsccsd = img_jsccsd * weight / s_map_arg.j
                    torch.cuda.empty_cache()

                if (id_iter_jsccsd + 1) % iter_arg.save_iter_step == 0:
                    img_jsccsd_iter[id_save, :] = torch.squeeze(img_jsccsd).cpu()
                    id_save += 1
                    print("JSCC-SD Iteration ", str(id_iter_jsccsd + 1), " ends, time used:", time.time() - time_start, "s")

            for p in processes:
                p.join()
            processes.clear()

            print("JSCC-SD OSEM ends, time used:", time.time() - time_start)
            torch.cuda.empty_cache()

    # save images as binary file to 'Figure'
    save_img(img_sc, img_scd, img_jsccd, img_jsccsd, img_sc_iter, img_scd_iter, img_jsccd_iter, img_jsccsd_iter, iter_arg, save_path)