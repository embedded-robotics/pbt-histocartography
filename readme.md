Histocartography setup for Pediatric Brain Tumor

1. Clone HistoCartography
- git clone git@github.com:BiomedSciAI/histocartography.git

2. conda create -n pbt-histo python=3.9

3. pip install pillow
4. pip install h5py
5. pip install pandas
6. pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 (installed like this because dgl was available for 2.4.x and Cuda12.4)
7. pip install requests
8. pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html (this needs to be compatible with the torch version installed)
    - https://www.dgl.ai/pages/start.html
9. pip install opencv-python
10. pip install scikit-image==0.18
11. histocartography/histocartography/preprocessing/feature_extraction.py:20
    - Change from skimage.feature import greycomatrix, greycoprops -> skimage.feature import graycomatrix, graycoprops
12. pip install scikit-learn
13. histocartography/histocartography/preprocessing/feature_extraction.py:15
    - Change from skimage.future import graph -> from skimage import graph
14. Take the checkpoints from the MIATA /data/mn27889/pbt-histocartography/histocartography/checkpoints [The old links provided in the code are broken]
15. pip install seaborn
16. histocartography/histocartography/utils/draw_utils.py:7
    - plt.style.use("seaborn-whitegrid") -> plt.style.use("seaborn-v0_8-whitegrid")
17. histocartography/histocartography/preprocessing/nuclei_extraction.py: [Change datatype from uint16 to uint32 to support the image labels for high resolution WSIs]
def process_instance(pred_map: np.ndarray, output_dtype: str = "uint16") -> def process_instance(pred_map: np.ndarray, output_dtype: str = "uint32"):

18. histocartography/histocartography/preprocessing/graph_builders.py [Don't Convert kneighbors_graph() to .toarray()]
kneighbors_graph(centroids,self.k,mode="distance",include_self=False, metric="euclidean").toarray() -> kneighbors_graph(centroids,self.k,mode="distance",include_self=False, metric="euclidean")