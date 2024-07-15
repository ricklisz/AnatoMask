from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

trainer = 'STUNetTrainer_base_finetune_big_total__nnUNetPlans__3d_fullres'

if __name__ == '__main__':
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_STUNet(
        join(nnUNet_results, 'Dataset503_SegTHOR/'  + trainer),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )
    # variant 1: give input and output folders
    predictor.predict_from_files(join(nnUNet_raw, 'Dataset503_SegTHOR/imagesTr'),
                                 join(nnUNet_results, 'Dataset503_SegTHOR/' + trainer + '/validation'),
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=8, num_processes_segmentation_export=8,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)


