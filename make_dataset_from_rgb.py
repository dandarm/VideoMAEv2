import argparse

from dataset.build_dataset import make_sup_dataset, make_unsup_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Crea il dataset di video tiles per training su arimassRGB',
        add_help=False)
    parser.add_argument('--on',
        type=str,
        default='leonardo',
        #metavar='NAME',
        help='imposta i path sulla macchina'
    ),

    args =  parser.parse_args()

    if args.on == 'leonardo':
        input_dir = "$FAST/Medicanes_Data/fromgcloud"
        output_dir = "$FAST/airmass/" 
    elif args.on == 'ewc':
        input_dir = "../fromgcloud"
        output_dir = "../airmassRGB/supervised/"


    make_sup_dataset(input_dir, output_dir)
