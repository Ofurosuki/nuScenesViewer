from viewer.nuscenesController.controller import NuScenesVisualizer 
def main():
    #print("Hello World!")
    dataset_root = "./data/"
    vis = NuScenesVisualizer(dataset_root)
    vis.visualize_n_frames(n_frames=3,scene_index=9)
