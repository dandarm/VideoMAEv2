
import argparse
from dataset.build_dataset import (
    make_sup_dataset,
    make_unsup_dataset,
    make_master_df,
    make_relabeled_dataset,
    make_dataset_from_manos_tracks,
    make_dataset_from_entire_year,
    make_tracking_dataset_from_manos_tracks,
    make_neighboring_multiclass_dataset,
)


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
    
    # TODO: verificare come mai abbiamo due opzioni simili: manos_tracks e tracking_manos, sembrano fare la stessa cosa andando contro la richiesta di togliere neighboring manos
    parser.add_argument('--manos_tracks',
        type=str,
        help='specificare il file csv di tracks derivato da Manos')
    parser.add_argument('--tracking_manos',
        type=str,
        #default='medicanes_new_windows.csv',
        help='file CSV Manos per creare dataset di tracking (split train/test/val)')
    parser.add_argument('--neighboring',
        action='store_true',
        help="crea il dataset di classificazione a 3 classi (0,1,2 con neighboring)")
    parser.add_argument(
        '--neighboring-prefix',
        type=str,
        default='neighboring',
        help="prefisso per i CSV generati dalla pipeline a 3 classi (default: 'neighboring')"
    )
    
    parser.add_argument('--all_year',
        #action='store_true',
        type=int,
        #default=2023,
        help="specificare l'anno ")
    
    parser.add_argument('--no-validation', action='store_true',
                        help='se non creare un set di validazione')

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
    elif args.neighboring:
        make_neighboring_multiclass_dataset(input_dir, output_dir, **vars(args))
    elif args.cloudy:
        make_relabeled_dataset(input_dir, output_dir, cloudy=True)
        #make_relabeled_dataset(input_dir, output_dir, cloudy=True, 
        #                       master_df_path="all_data_CL7_tracks_SHORT4TEST.csv")
    elif args.manos_tracks:
        make_dataset_from_manos_tracks(input_dir, output_dir, **vars(args))
    elif args.tracking_manos:
        make_tracking_dataset_from_manos_tracks(input_dir, output_dir, **vars(args))
    elif args.all_year:
        make_dataset_from_entire_year(args.all_year, input_dir, output_dir)
    else:
        make_sup_dataset(input_dir, output_dir)
