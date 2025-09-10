# view_med_preds.py
# script per visualizzare le predizioni di tutte le tile del Mediterraneo


from pathlib import Path
import os
import sys
import pandas as pd
import torch

from arguments import prepare_finetuning_args
from dataset.data_manager import make_validation_data_builder_from_manos_tracks, make_validation_data_builder_from_entire_year
from dataset.build_dataset import calc_tile_offsets
from view_test_tiles import filling_missing_tile, render_and_save_frame
from view_test_tiles import make_animation_parallel_ffmpeg




def view_mediterranean_predictions(df_pred_path, input_dir, output_dir, entire_year=None):

    args = prepare_finetuning_args()

    # Flag di fallback: se non c'è GPU usa CPU
    ALLOW_CPU_FALLBACK = True
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        if not ALLOW_CPU_FALLBACK:
            raise RuntimeError('CUDA non disponibile; abilita ALLOW_CPU_FALLBACK per usare CPU')
        print('CUDA non disponibile: uso CPU per validazione/logits.')
        dev = torch.device('cpu')

    args.device=dev


    if entire_year is not None:
        print(f"data builder form year {entire_year}")
        val_builder = make_validation_data_builder_from_entire_year(entire_year, input_dir, output_dir)
    else:
        val_builder = make_validation_data_builder_from_manos_tracks("medicane_data_input/medicanes_new_windows.csv", input_dir, output_dir)
    
    df_predictions = pd.read_csv(df_pred_path)
    df_filtrato_on_video_path = val_builder.df_video.merge(df_predictions, on='path')

    # Espando dataframe con le associazioni path_offsets -> predictions
    records = []
    for _, row in df_filtrato_on_video_path.iterrows():
        for orig_path in row['orig_paths']:
            records.append({
                'path': orig_path,
                'predictions': row['predictions'],
                'tmp_label': row['labels'],
                'tile_offset_x': row['tile_offset_x'],
                'tile_offset_y': row['tile_offset_y'],
                'neighboring': row['neighboring'], 
            })

    # Li trasformiamo in un nuovo DataFrame
    df_mapping = pd.DataFrame(records)
    print(df_mapping[['path', 'tile_offset_x', 'tile_offset_y','tmp_label']].duplicated().sum())
    #Non ci sono path duplicati se in combinazione con gli offsets, 
    # a parte se i video sono generati con un certo overlap in tempo!
    # ottengo le predictions per ogni immagine
    df_mapping.drop_duplicates(inplace=True)

    # ottengo il dataframe di immagini che serve per creare il video del Mediterraneo, mi servono anche le 'x_pix', 'y_pix', 'source'
    df_data_merg = df_mapping.merge(val_builder.master_df, on=['path', 'tile_offset_x', 'tile_offset_y'], how='left').drop(columns='label').rename(columns={'tmp_label':'label'})

    ### devo risolvere con le tile mancanti
    offsets = calc_tile_offsets(stride_x=213, stride_y=196)
    df_offsets = pd.DataFrame(offsets, columns=['tile_offset_x', 'tile_offset_y'])
    def expand_group(group):
        merged = df_offsets.merge(group, on=['tile_offset_x', 'tile_offset_y'], how='left', indicator=True)
        path_value = group['path'].iloc[0]
        merged['path'] = path_value

        extra_cols = [col for col in group.columns if col not in ['path', 'tile_offset_x', 'tile_offset_y']]
        # Ricopia i valori costanti del gruppo originale
        for col in ['datetime']:
            val = group[col].iloc[0]   #.mode()[0] if not group[col].isnull().all() else None
            merged[col] = merged[col].fillna(val)

        merged[filling_missing_tile] = merged['_merge'] == 'left_only'  # True se mancava
        return merged[['path', 'tile_offset_x', 'tile_offset_y'] + extra_cols + [filling_missing_tile]]

    # Applichiamo la funzione a ogni path
    expanded_df = df_data_merg.groupby('path', group_keys=False).apply(expand_group).reset_index(drop=True)
    expanded_df.predictions = expanded_df.predictions.astype('Int8')
    expanded_df.label = expanded_df.label.astype('Int8')


    # # prova con una immagine
    # grouped = expanded_df.groupby("path", dropna=False)
    # print(f" abbiamo {len(list(grouped))} gruppi", flush=True)
    # list_grouped_df = list(grouped)

    # render_and_save_frame((90, list_grouped_df, "./"), overwrite=True)

    
    make_animation_parallel_ffmpeg(expanded_df, nomefile='validation_entire_2023')



if __name__ == "__main__":
    input_dir = "../fromgcloud"
    output_dir = "../airmassRGB/supervised/"
    df_pred_path = 'output/inference_predictions_entire_year.csv'
    entire_year = 2023
    view_mediterranean_predictions(df_pred_path, input_dir, output_dir, entire_year)