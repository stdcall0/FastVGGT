import argparse
import numpy as np
import torch
import time
import os

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from visual_utils import predictions_to_glb

from vggt.utils.eval_utils import (
    load_poses,
    get_vgg_input_imgs,
    get_sorted_image_paths,
    build_frame_selection,
    load_images_rgb,
    infer_vggt_and_reconstruct,
    evaluate_scene_and_save,
)

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the FastVGGT model.")
    
    parser.add_argument("--data_path", type=str, default='',
                        help="Path to the input image directory.")
    parser.add_argument("--save_path", type=str, default='out.glb',
                        help="Path to save the output .glb file.")
    parser.add_argument("--ckpt", type=str, default='./ckpt/model_tracker_fixed_e20.pt',
                        help="Path to the model checkpoint file.")
    parser.add_argument("--merging", type=int, default=0,
                        help="VGGT aggregator merging steps")
    parser.add_argument("--conf", type=float, default=50.0,
                        help="GLB confidence (1~100)")
                        
    args = parser.parse_args()

    # 1. Prepare model
    print("Loading model...")
    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    model = VGGT(merging=args.merging, enable_point=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    
    model = model.cuda().eval()
    model = model.to(torch.bfloat16)
    
    del ckpt

    # 2. Prepare input data
    print("Loading images...")
    
    # get all image paths under args.data_path
    image_names = [f for f in os.listdir(args.data_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = [os.path.join(args.data_path, name) for name in image_names]

    
    images = load_images_rgb(image_paths)
    images_array = np.stack(images)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
    # imgs = load_and_preprocess_images(image_paths).to(device)
    model.update_patch_dimensions(patch_width, patch_height)
    print(f"📐 Image patch dimensions: {patch_width}x{patch_height}")

    # 3. Inference
    print("Running model inference...")
    
    device = torch.device("cuda")
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.amp.autocast('cuda', dtype=dtype):
            vgg_input_cuda = vgg_input.cuda().to(torch.bfloat16)
            predictions = model(vgg_input_cuda, image_paths=image_paths)
            
        torch.cuda.synchronize()
        end_time = time.time()
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    print(f"Model inference finished.")
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    print(f"Peak GPU VRAM usage: {peak_vram_gb:.4f} GB")

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"],
                                                        (vgg_input.shape[2], vgg_input.shape[3]))
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].float().cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    
    torch.cuda.empty_cache()
    
    print("Building GLB scene...")
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=args.conf,
        show_cam=False
    )

    print(f"Saving GLB to: {args.save_path}")
    glbscene.export(file_obj=args.save_path)
    print("Done.")
