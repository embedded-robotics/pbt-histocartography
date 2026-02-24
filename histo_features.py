# %%
import sys
from PIL import Image
import os
import numpy as np
import openslide
from tqdm import tqdm
import pickle

# %%
sys.path.append(os.path.join(os.getcwd(), 'histocartography'))
from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder
from histocartography.visualization import OverlayGraphVisualization, InstanceImageVisualization

# %% [markdown]
# Initiating all the required modules

# %%
nuclei_detector = NucleiExtractor()
feature_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=224, resize_size=224)
knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)

# %% [markdown]
# Processing files one by one

# %%
raw_wsi_dir = "/data/jy26539/data/wsi_raw"
wsi_graph_dir = "wsi_graphs"
os.makedirs(wsi_graph_dir, exist_ok=True)
wsi_names = os.listdir(raw_wsi_dir)
print(f"Found {len(wsi_names)} WSIs in {raw_wsi_dir}.")

# %%
wsi_resolution_level = 1
features_dict = {}
cell_graph_dict = {}

for i in tqdm(range(len(wsi_names)), desc="Processing"):
    wsi_name = wsi_names[i]
    print(f"Processing {wsi_name} ({i+1}/{len(wsi_names)})...")
    wsi_path = os.path.join(raw_wsi_dir, wsi_name)
    wsi_graph_path = os.path.join(wsi_graph_dir, f"{wsi_name.split('.')[0]}_Graph.png")

    slide = openslide.OpenSlide(wsi_path)
    svs_image = slide.read_region((0, 0), wsi_resolution_level, slide.level_dimensions[wsi_resolution_level]).convert('RGB')
    svs_image_np = np.array(svs_image)
    
    nuclei_map, nuclei_centers = nuclei_detector.process(svs_image_np)
    features = feature_extractor.process(svs_image_np, nuclei_map)
    features_dict[wsi_name.split('.')[0]] = features.detach().cpu().numpy()  # Store features for later use
    
    if len(nuclei_centers) > 5:
        cell_graph = knn_graph_builder.process(nuclei_map, features)
        cell_graph_dict[wsi_name.split('.')[0]] = cell_graph  # Store cell graph for later use
        
        visualizer = OverlayGraphVisualization(instance_visualizer=InstanceImageVisualization(instance_style="filled+outline"))
        viz_cg = visualizer.process(canvas=svs_image_np, graph=cell_graph, instance_map=nuclei_map)
        viz_cg.save(wsi_graph_path)
        print(f"Processed {wsi_name} and saved graph visualization to {wsi_graph_path}.")
    else:
        print(f"Less than 5 nuclei detected in {wsi_name}. Skipping graph construction and visualization.")

    slide.close()

# %% [markdown]
# Save the feature_dict and cell_graph_dict as pickle file

# %%
with open('wsi_raw_graph_features.pkl', 'wb') as f:
    pickle.dump(features_dict, f)

with open('wsi_raw_cell_graphs.pkl', 'wb') as f:
    pickle.dump(cell_graph_dict, f)

# %% [markdown]
# Read the feature_dict and cell_graph_dict

# %%
with open('wsi_raw_graph_features.pkl', 'rb') as f:
    features_dict = pickle.load(f)

with open('wsi_raw_cell_graphs.pkl', 'rb') as f:
    cell_graph_dict = pickle.load(f)
