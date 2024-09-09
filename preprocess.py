import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import glob
import logging
import json
from typing import Any, Dict
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from collections import OrderedDict

from openpose.src import util
from openpose.src.body import Body
from openpose.src.hand import Hand
import SCHP.networks
from SCHP.utils.transforms import transform_logits
from SCHP.datasets.simple_extractor_dataset import SimpleFolderDataset
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger
from DensePose.densepose import add_densepose_config
from DensePose.densepose.vis.base import CompoundVisualizer
from DensePose.densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from DensePose.densepose.vis.densepose_results_textures import get_texture_atlas
from DensePose.densepose.vis.densepose_outputs_vertex import get_texture_atlases
from DensePose.densepose.vis.extractor import CompoundExtractor, create_extractor

# Set image folder
INPUT_DIR = './Input'

# Dataset label for human parsing
dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Tosor-skin', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },

}

# Ooenpose
class PoseEstimator:
    def __init__(self, body_model_path, hand_model_path, output_image_dir, output_json_dir):
        # Model pth files in ./checkpoints
        self.body_estimation = Body(body_model_path)
        self.hand_estimation = Hand(hand_model_path)
        
        # Define input and output directories for visualised image and json files
        self.input_dir = INPUT_DIR
        self.output_image_dir = output_image_dir
        self.output_json_dir = output_json_dir

    def process_image(self, filename):
        input_path = os.path.join(self.input_dir, filename)
        name, ext = os.path.splitext(filename)
        # Filenames for images file and body & hand points json file
        output_image_filename = f"{name}_pose{ext}"
        output_json_filename = f"{name}_keypoints.json"
        output_handjson_filename = f"{name}_handpoints.json"
        # Output paths
        output_image_path = os.path.join(self.output_image_dir, output_image_filename)
        output_json_path = os.path.join(self.output_json_dir, output_json_filename)
        output_handjson_path = os.path.join(self.output_json_dir, output_handjson_filename)

        # Read the image
        oriImg = cv2.imread(input_path)  # B,G,R order
        if oriImg is None:
            print(f"Error reading image {filename}. Skipping.")
            return

        # Body pose estimation
        candidate, subset = self.body_estimation(oriImg)
        canvas = np.zeros_like(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        # Hand pose estimation
        hands_list = util.handDetect(candidate, subset, oriImg)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
            all_hand_peaks.append(peaks)
        canvas = util.draw_handpose(canvas, all_hand_peaks)

        # Convert BGR to RGB for saving
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # Save the processed image to the output directory
        plt.imsave(output_image_path, canvas_rgb)


        # Prepare data for JSON output
        pose_data = {
            'candidate': candidate.tolist(),
            'subset': subset.tolist(),           
        }

        # Save the JSON data to the output JSON directory
        with open(output_json_path, 'w') as json_file:
            json.dump(pose_data, json_file)

        # Hand data for JSON output
        hand_data = { 'hand_peaks': [peaks.tolist() for peaks in all_hand_peaks] }

        # Check if all values in hand_peaks are 0
        if any(any(peak != 0 for peak in peaks) for peaks in hand_data['hand_peaks']):
            with open(output_handjson_path, 'w') as json_file:
                json.dump(hand_data, json_file)
        else:
            print("All values are zero. File not created.")

    def run(self):
        # Loop through all files in the input directory
        for filename in os.listdir(self.input_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')): 
                self.process_image(filename)


# Human Parsing
class HumanParsing:
    def __init__(self):
        self.dataset = 'lip'
        self.model_restore = 'checkpoints/final.pth' 
        self.gpu = '0'
        self.input_dir = INPUT_DIR
        self.output_dir = 'Output/parse'
        self.logits = False

        # Load dataset settings
        self.num_classes = dataset_settings[self.dataset]['num_classes']
        self.input_size = dataset_settings[self.dataset]['input_size']
        self.label = dataset_settings[self.dataset]['label']
        print(f"Evaluating total class number {self.num_classes} with {self.label}")

        # Initialize model with pre-trained weights
        self.model = self.init_model()

        # Set up transformations and dataloader
        self.transform = self.get_transform()
        self.dataset = SimpleFolderDataset(root=self.input_dir, input_size=self.input_size, transform=self.transform)
        self.dataloader = DataLoader(self.dataset)

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Get color palette
        self.palette = self.get_palette(self.num_classes)
    

    def get_palette(self, num_cls):
        # Returns the color map for visualizing the segmentation mask.
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def init_model(self):
        model = SCHP.networks.init_model('resnet101', num_classes=self.num_classes, pretrained=None)

        state_dict = torch.load(self.model_restore)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.cuda()
        model.eval()
        return model

    def get_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

    def run(self):
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.dataloader)):
                image, meta = batch
                img_name = meta['name'][0]
                c = meta['center'].numpy()[0]
                s = meta['scale'].numpy()[0]
                w = meta['width'].numpy()[0]
                h = meta['height'].numpy()[0]

                # Model inference
                output = self.model(image.cuda())
                upsample = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)
                upsample_output = upsample(output[0][-1][0].unsqueeze(0))
                upsample_output = upsample_output.squeeze()
                upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

                # Apply transformation and get logits
                logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=self.input_size)
                parsing_result = np.argmax(logits_result, axis=2)
                
                # Save segmentation mask / label results
                label_map_path = os.path.join(self.output_dir, img_name[:-4] + '_label.png')
                label_image = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
                label_image.save(label_map_path)
                
                # Save clour-coded parsing results
                parsing_result_path = os.path.join(self.output_dir, img_name[:-4] + '_parsed.png')
                output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
                output_img.putpalette(self.palette)
                output_img.save(parsing_result_path)
                
                if self.logits:
                    logits_result_path = os.path.join(self.output_dir, img_name[:-4] + '.npy')
                    np.save(logits_result_path, logits_result)



class DensePoseVisualizer:
    VISUALIZERS = {"dp_segm": DensePoseResultsFineSegmentationVisualizer,}

    def __init__(self, cfg_fpath, model_fpath, input_spec, visualizations, min_score=0.8, 
                 nms_thresh=None, texture_atlas=None, texture_atlases_map=None, output='Output/dense'):
        self.cfg_fpath = cfg_fpath
        self.model_fpath = model_fpath
        self.input_spec = input_spec
        self.visualizations = visualizations
        self.min_score = min_score
        self.nms_thresh = nms_thresh
        self.texture_atlas = texture_atlas
        self.texture_atlases_map = texture_atlases_map
        self.output_dir = output
        self.logger = logging.getLogger("DensePoseVisualizer")
        self.cfg = self._setup_config()

    def _setup_config(self):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(self.cfg_fpath)
        opts = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", str(self.min_score)]
        if self.nms_thresh:
            opts.extend(["MODEL.ROI_HEADS.NMS_THRESH_TEST", str(self.nms_thresh)])
        cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = self.model_fpath
        cfg.freeze()
        return cfg

    def _get_input_files(self):
        if os.path.isdir(self.input_spec):
            return [os.path.join(self.input_spec, f) for f in os.listdir(self.input_spec) if os.path.isfile(os.path.join(self.input_spec, f))]
        return [self.input_spec] if os.path.isfile(self.input_spec) else glob.glob(self.input_spec)

    def _create_context(self):
        vis_specs = self.visualizations.split(",")
        visualizers = [self.VISUALIZERS[spec](cfg=self.cfg, texture_atlas=get_texture_atlas(self.texture_atlas), texture_atlases_dict=get_texture_atlases(self.texture_atlases_map)) for spec in vis_specs]
        return {
            "extractor": CompoundExtractor([create_extractor(vis) for vis in visualizers]),
            "visualizer": CompoundVisualizer(visualizers),
            "out_fname": self.output_dir,
            "entry_idx": 0,
        }

    def _process_image(self, context, entry, outputs):
        import cv2, numpy as np
        image = np.tile(cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)[:, :, np.newaxis], [1, 1, 3])
        data = context["extractor"](outputs)
        image_vis = context["visualizer"].visualize(image, data)
        out_fname = f'{self.output_dir}/{os.path.basename(entry["file_name"]).split(".")[0]}_dense.jpg'
        os.makedirs(self.output_dir, exist_ok=True)
        cv2.imwrite(out_fname, image_vis)
        self.logger.info(f"Output saved to {out_fname}")
        context["entry_idx"] += 1

    def run(self):
        predictor = DefaultPredictor(self.cfg)
        file_list = self._get_input_files()
        if not file_list:
            self.logger.warning(f"No input images found in {self.input_spec}")
            return
        context = self._create_context()
        for file_name in file_list:
            img = read_image(file_name, format="BGR")
            with torch.no_grad():
                outputs = predictor(img)["instances"]
                self._process_image(context, {"file_name": file_name, "image": img}, outputs)


def main():
   #Openpose
    body_model_path = 'checkpoints/body_pose_model.pth'
    hand_model_path = 'checkpoints/hand_pose_model.pth'
    OUTPUT_IMAGE_DIR = 'Output/pose/img'  # Directory for output images
    OUTPUT_JSON_DIR = 'Output/pose/json'
    pose_estimator = PoseEstimator(body_model_path, hand_model_path, OUTPUT_IMAGE_DIR, OUTPUT_JSON_DIR)
    pose_estimator.run()



    #Human Parsing
    human_parser = HumanParsing()
    human_parser.run()


    #DensePose
    setup_logger(name="DensePoseVisualizer")
    visualizer = DensePoseVisualizer(
        cfg_fpath="DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml",
        model_fpath="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl",
        input_spec="Input",
        visualizations="dp_segm", 
        min_score=0.8,
        nms_thresh=0.5,
        texture_atlas=None,
        texture_atlases_map=None,
        output="Output/dense"
    )
    visualizer.run()


if __name__ == "__main__":
    main()
