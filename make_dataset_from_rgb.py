
import argparse
from dataset.build_dataset import make_sup_dataset, make_unsup_dataset, make_master_df, make_relabeled_dataset, make_dataset_from_manos_tracks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Crea il dataset di video tiles per training su arimassRGB',
        add_help=False)
    parser.add_argument('--on',
        type=str,
        default='leonardo',
        #metavar='NAME',
        help='imposta i path sulla macchina')
    parser.add_argument('--master_df',
        action='store_true',
        help='se creare il master dataframe')
    parser.add_argument('--relabeled_df',
        action='store_true',
        help='se modificare le label')
    parser.add_argument('--cloudy',
        action='store_true',
        help='se escludere il cielo sereno')
    parser.add_argument('--manos_tracks',
        #action='store_true',
        type=str,
        default='medicanes_new_windows.csv',
        help='specificare il file csv di tracks derivato da Manos')

    args = parser.parse_args()

    if args.on == 'leonardo':
        input_dir = "$FAST/Medicanes_Data/from_gcloud"
        output_dir = "$FAST/airmass/" 
    elif args.on == 'ewc':
        input_dir = "../fromgcloud"
        output_dir = "../airmassRGB/supervised/"

    if args.master_df:
        make_master_df(input_dir, output_dir)
    elif args.relabeled_df:        
        make_relabeled_dataset(input_dir, output_dir)
    elif args.cloudy:
        make_relabeled_dataset(input_dir, output_dir, cloudy=True)
        #make_relabeled_dataset(input_dir, output_dir, cloudy=True, 
        #                       master_df_path="all_data_CL7_tracks_SHORT4TEST.csv")
    elif args.manos_tracks:
        make_dataset_from_manos_tracks(args.manos_tracks, input_dir, output_dir)

    else:
        make_sup_dataset(input_dir, output_dir)
