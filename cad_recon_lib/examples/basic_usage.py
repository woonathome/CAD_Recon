from cad_recon_lib_holefix import (
    ABCMultiModalDataset,
    ReconstructionOptions,
    visualize_brep_reconstruction_comparison,
    visualize_multimodal_sample,
)


def main():
    dataset = ABCMultiModalDataset(
        base_dir="./abc_dataset_filtered-1",
        pcd_num_points=2048,
        voxel_res=64,
    )
    sample = dataset[0]

    visualize_multimodal_sample(dataset, sample=sample)

    opts = ReconstructionOptions(
        fast_vis_mode=True,
        offset_x=2.5,
    )
    visualize_brep_reconstruction_comparison(
        sample,
        options=opts,
        enforce_closed_solid_prior=True,
    )


if __name__ == "__main__":
    main()
