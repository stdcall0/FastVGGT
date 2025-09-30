import argparse
import torch
import time
import os

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from visual_utils import predictions_to_glb

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
    model = model.to(dtype)
    
    del ckpt

    # 2. Prepare input data
    print("Loading images...")
    
    # get all image paths under args.data_path
    image_names = [f for f in os.listdir(args.data_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    device = torch.device("cuda")
    imgs = load_and_preprocess_images(image_names).to(device)

    # 3. Inference
    print("Running model inference...")
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            start_time = time.time()

            predictions = model(imgs)

            torch.cuda.synchronize()
            end_time = time.time()
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    print(f"Model inference finished.")
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    print(f"Peak GPU VRAM usage: {peak_vram_gb:.4f} GB")

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], imgs.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
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
        conf_thres=args.conf
    )

    print(f"Saving GLB to: {args.save_path}")
    glbscene.export(file_obj=args.save_path)
    print("Done.")
