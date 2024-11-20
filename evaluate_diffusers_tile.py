import argparse
import math
import numpy as np
from tqdm import tqdm
import os
from skimage import io
import sys
import cv2
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.utils.data as data

from core.datasets_return_dict import KITTI, MpiSintel
from local_diffusers.pipelines.DDPM import DDPMPipeline
from core.utils import frame_utils
from core.utils import flow_viz


def compute_grid_indices(image_shape, patch_size, min_overlap_w =20, min_overlap_h=20):
    if min_overlap_h >= patch_size[0] or min_overlap_w >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap_h))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap_w))[:5]
    #print ("??? hs = ", hs)
    #print ("??? ws = ", ws)
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    # unique
    hs = np.unique(hs)
    # ws.append(32)
    return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, patch_size, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(
        [torch.arange(patch_size[0]), torch.arange(patch_size[1])],
        indexing = 'ij'
        )
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h + patch_size[0], w:w + patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx + 1, h:h + patch_size[0], w:w + patch_size[1]])

    return patch_weights


@torch.no_grad()
def validate_kitti(pipeline, args=None, sigma=0.05, start_t=8):
    IMAGE_SIZE = None
    TRAIN_SIZE = [320, 448]
    min_overlap = 250

    pipeline.unet = pipeline.unet.to(torch.bfloat16)
    val_dataset = KITTI(
            split='training',
            root= os.path.join(args.data_dir, "KITTI_2015")
            )
    val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=4)

    out_list, epe_list = [], []
    i = 0
    for batch in tqdm(val_loader):
        for k in batch:
            if type(batch[k]) == torch.Tensor:
                batch[k] = batch[k].cuda()

        B, _, H, W = batch["image0"].shape
        if IMAGE_SIZE is None or H != IMAGE_SIZE[0] or W != IMAGE_SIZE[1]:
            #print(f"replace {IMAGE_SIZE} with [{H}, {W}]")
            IMAGE_SIZE = [H, W]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE, min_overlap_w = min_overlap)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        batch["image0"] = 2 * (batch["image0"] / 255.0) - 1.0
        batch["image1"] = 2 * (batch["image1"] / 255.0) - 1.0
        #print ("??? batch input image size = ", batch["image0"].shape) #[1, 3, 375, 1242]

        resized_image1 = F.interpolate(batch["image0"], TRAIN_SIZE, mode='bicubic', align_corners=True)
        resized_image2 = F.interpolate(batch["image1"], TRAIN_SIZE, mode='bicubic', align_corners=True)
        inputs = torch.cat([resized_image1, resized_image2], dim=1)
        resized_flow = pipeline(
            inputs=inputs.to(torch.bfloat16),
            batch_size=inputs.shape[0],
            num_inference_steps=args.ddpm_num_steps,
            output_type="tensor",
            normalize=False  # false for the coarse estimation
        ).images.to(torch.float32)
        
        #import pdb; pdb.set_trace()
        resized_flow = F.interpolate(resized_flow, IMAGE_SIZE, mode='bicubic', align_corners=True) * \
               torch.tensor([W / TRAIN_SIZE[1], H / TRAIN_SIZE[0]]).view(1, 2, 1, 1).cuda()

        flows = 0
        flow_count = 0

        # Sample noise that we'll add to the images
        noise = torch.randn(resized_flow.shape).to(resized_flow.device)
        timesteps = torch.ones(B).to(resized_flow.device) * (start_t - 1)
        noised_flow = pipeline.scheduler.add_noise(
                                resized_flow, 
                                noise, 
                                timesteps.to(torch.int32)
                                )

        image1_tiles = []
        image2_tiles = []
        noised_flow_tiles = []
        for idx, (h, w) in enumerate(hws):
            image1_tiles.append(batch["image0"][:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]])
            image2_tiles.append(batch["image1"][:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]])
            noised_flow_tiles.append(noised_flow[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]])

        inputs = torch.cat([torch.cat(image1_tiles, dim=0), torch.cat(image2_tiles, dim=0), torch.cat(noised_flow_tiles, dim=0)], dim=1)
        # inputs shape : [N=10, C=8, H, W], 
        # 10=number of tiles; C=3(rgb_0)+3(rgb_1)+2(flow);
        #print ("??? inputs shape = ", inputs.shape)
        flow_pre_total = pipeline(
            inputs=inputs.to(torch.bfloat16),
            batch_size=inputs.shape[0],
            num_inference_steps=args.ddpm_num_steps,
            output_type="tensor",
            normalize=args.normalize_range,
            start_t=start_t
        ).images

        
        for idx, (h, w) in enumerate(hws):
            flow_pre = flow_pre_total[idx*B:(idx+1)*B]
            padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow = flows / flow_count

        epe = torch.sum((flow - batch['target']) ** 2, dim=1).sqrt()
        mag = torch.sum(batch['target'] ** 2, dim=1).sqrt()
        for index in range(B):
            epe_indexed = epe[index].view(-1)
            mag_indexed = mag[index].view(-1)
            val = batch['valid'][index].view(-1) >= 0.5
            out = ((epe_indexed > 3.0) & ((epe_indexed / mag_indexed) > 0.05)).float()
            epe_list.append(epe_indexed[val].mean().cpu().item())
            out_list.append(out[val].cpu().numpy())


            # save flow
            flow_est_b = flow[index].cpu().numpy().transpose((1, 2, 0))
            os.makedirs(args.result_dir, exist_ok=True)
            flo_path = os.path.join(args.result_dir, f"{(B*i+index):05d}")
            #frame_utils.writeFlow( flo_path + ".flo",  flow_est_b)
            flo_clr = flow_viz.flow_to_image(flow_est_b)
            flo_clr = frame_utils.add_text_to_image(
                text_str= f'epe: {epe_list[-1]:.2f}', 
                posi= (20, 30),#(x,y) 
                img = flo_clr,
                fontScale= 0.6,
                fontColor= (0,0,0)
                )
            io.imsave(flo_path + "-flow-clr.png", flo_clr)
        
        #if i > 2:
        #    break
        i += 1 

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    
    print("Validation KITTI: epe=%f, f1=%f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_sintel(pipeline, args=None, sigma=0.05, start_t=32):
    """ Peform validation using the Sintel (train) split """

    IMAGE_SIZE = None
    TRAIN_SIZE = [320, 448]
    min_overlap = 304

    pipeline.unet = pipeline.unet.to(torch.bfloat16)

    results = {}
    for dstype in ['final', "clean"]:
        result_dir = os.path.join(args.result_dir, dstype)
        val_dataset = MpiSintel(
            split='training', dstype=dstype,
            root= os.path.join(args.data_dir, "Sintel")
            )
        val_loader = data.DataLoader(val_dataset, 
                                     batch_size=args.train_batch_size, 
                                     pin_memory=True, 
                                     shuffle=False,
                                     num_workers=4)

        epe_list = []
        
        i = 0
        for batch in tqdm(val_loader):
            for k in batch:
                if type(batch[k]) == torch.Tensor:
                    batch[k] = batch[k].cuda()

            B, _, H, W = batch["image0"].shape
            if IMAGE_SIZE is None or H != IMAGE_SIZE[0] or W != IMAGE_SIZE[1]:
                #print(f"replace {IMAGE_SIZE} with [{H}, {W}]")
                IMAGE_SIZE = [H, W]
                hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE, min_overlap_w = min_overlap)
                weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

            batch["image0"] = 2 * (batch["image0"] / 255.0) - 1.0
            batch["image1"] = 2 * (batch["image1"] / 255.0) - 1.0

            resized_image1 = F.interpolate(batch["image0"], TRAIN_SIZE, mode='bicubic', align_corners=True)
            resized_image2 = F.interpolate(batch["image1"], TRAIN_SIZE, mode='bicubic', align_corners=True)
            inputs = torch.cat([resized_image1, resized_image2], dim=1)
            resized_flow = pipeline(
                inputs=inputs.to(torch.bfloat16),
                batch_size=inputs.shape[0],
                num_inference_steps=args.ddpm_num_steps,
                output_type="tensor",
                normalize=False  # false for the coarse estimation
            ).images.to(torch.float32)

            resized_flow = F.interpolate(resized_flow, IMAGE_SIZE, mode='bicubic', align_corners=True) * \
                           torch.tensor([W / TRAIN_SIZE[1], H / TRAIN_SIZE[0]]).view(1, 2, 1, 1).cuda()

            flows = 0
            flow_count = 0
            # Sample noise that we'll add to the images
            noise = torch.randn(resized_flow.shape).to(resized_flow.device)
            timesteps = torch.ones(B).to(resized_flow.device) * (start_t - 1)
            noised_flow = pipeline.scheduler.add_noise(resized_flow, noise, timesteps.to(torch.int32))

            image1_tiles = []
            image2_tiles = []
            noised_flow_tiles = []
            for idx, (h, w) in enumerate(hws):
                image1_tiles.append(batch["image0"][:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]])
                image2_tiles.append(batch["image1"][:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]])
                noised_flow_tiles.append(noised_flow[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]])

            inputs = torch.cat(
                [torch.cat(image1_tiles, dim=0), torch.cat(image2_tiles, dim=0), torch.cat(noised_flow_tiles, dim=0)],
                dim=1)
            flow_pre_total = pipeline(
                inputs=inputs.to(torch.bfloat16),
                batch_size=inputs.shape[0],
                num_inference_steps=args.ddpm_num_steps,
                output_type="tensor",
                normalize=args.normalize_range,
                start_t=start_t
            ).images

            for idx, (h, w) in enumerate(hws):
                flow_pre = flow_pre_total[idx * B:(idx + 1) * B]
                padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pre * weights[idx], padding)
                flow_count += F.pad(weights[idx], padding)

            flow = flows / flow_count
            

            epe = torch.sum((flow - batch['target']) ** 2, dim=1).sqrt()
            epe_list.append(epe.view(-1).cpu().numpy())
            
            # save flow
            if i % 100 == 0:
                index = 0
                flow_est_b = flow[index].cpu().numpy().transpose((1, 2, 0))
                
                os.makedirs(result_dir, exist_ok=True)
                flo_path = os.path.join(result_dir, f"{(B*i+index):05d}")
                #frame_utils.writeFlow( flo_path + ".flo",  flow_est_b)
                flo_clr = flow_viz.flow_to_image(flow_est_b)
                flo_clr = frame_utils.add_text_to_image(
                    text_str= f'epe: {epe_list[-1].mean():.2f}', 
                    posi= (20, 30),#(x,y) 
                    img = flo_clr,
                    fontScale= 0.6,
                    fontColor= (0,0,0)
                    )
                io.imsave(flo_path + "-flow-clr.png", flo_clr)
            
            #if i > 2:
            #    break
            
            i += 1

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[f"{dstype}_tile"] = epe
        results.update({
            f'{dstype}/epe': epe, 
            f'{dstype}/1px': px1,
            f'{dstype}/3px': px3,
            f'{dstype}/5px': px5,
            })

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_path', help="restore pipeline")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 448])
    parser.add_argument('--train_batch_size', type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument('--ddpm_num_steps', type=int, default=64)
    parser.add_argument("--normalize_range", action="store_true",
                        help="Whether to normalize the flow range into [-1,1].")
    #parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--validation_data', type = str, default='kitti', help="data set to test")
    parser.add_argument('--data_dir', type = str, default= "datasets/", help="input data dir")
    parser.add_argument('--result_dir', type = str, default= "./results", help="result dir")
    parser.add_argument('--machine_name', type = str, help="result dir")
    parser.add_argument('--eval_gpu_id', type = str, default='0', help="result dir")
    parser.add_argument('--model', type = str, default='ddvm', help="model name")
    
    args = parser.parse_args()
    pipeline = DDPMPipeline.from_pretrained(args.pipeline_path).to('cuda')
    val_dataset = args.validation_data
    results = {}

    if args.result_dir.find("results_nfs/") >= 0:
        pos = args.result_dir.find("results_nfs/")
        general_csv_root = args.result_dir[:pos+len("results_nfs/")]
    elif args.result_dir.find("results/") >= 0:
        pos = args.result_dir.find("results/")
        general_csv_root = args.result_dir[:pos+len("results/")]
    else:
        raise NotImplementedError

    if val_dataset == 'kitti':
        results.update(validate_kitti(pipeline, args=args))
        
        os.makedirs(args.result_dir, exist_ok=True)
        timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        csv_file = os.path.join(args.result_dir, f"{val_dataset}-err.csv")
        messg = timeStamp + f",model={args.model},resultDir,{args.result_dir}" + \
                f",dataset,{val_dataset},data_root,{args.data_dir}," + \
                f",epe,{results['kitti-epe']},F1,{results['kitti-f1']}\n"

        print (messg)
        with open( csv_file, 'w') as fwrite:
            fwrite.write(messg)
        dst_csv_file = os.path.join(general_csv_root, 'kt15_eval_err.csv') 
        print (dst_csv_file)
        os.system(f'cat {csv_file} >> {dst_csv_file}')  
    
    elif val_dataset == 'sintel':
        results.update(validate_sintel(pipeline, args=args))
        
        os.makedirs(args.result_dir, exist_ok=True)
        timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        csv_file = os.path.join(args.result_dir, f"{val_dataset}-err.csv")

        messg = timeStamp + ",model={},resultDir,{},dataset,{},data_root,{},".format(
                args.model, args.result_dir, val_dataset, args.data_dir) + \
                (",epe(clean),{},1px(clean),{},3px(clean),{},5px(clean),{}").format(
                    results['clean/epe'], 
                    results['clean/1px'], 
                    results['clean/3px'], 
                    results['clean/5px'], 
                ) + \
                (",epe(final),{},1px(final),{},3px(final),{},5px(final),{}").format(
                    results['final/epe'], 
                    results['final/1px'], 
                    results['final/3px'], 
                    results['final/5px'], 
                ) + "\n"
        
        print (messg)
        with open( csv_file, 'w') as fwrite:
            fwrite.write(messg + "\n")
        dst_csv_file = os.path.join(general_csv_root, 'sintel_eval_err.csv') 
        print (dst_csv_file)
        os.system(f'cat {csv_file} >> {dst_csv_file}')
