import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
#import torchvision.transforms.functional as F
import torch.nn.functional as F



from utils import NativeScalerWithGradNormCount as NativeScaler

from dataset import build_pretraining_dataset
from torch.utils.data import DataLoader
from run_mae_pretraining import get_model
from run_mae_pretraining import main as main_pretraining
from arguments import prepare_args, Args  # NON TOGLIERE: serve a torch.load per caricare il mio modello addestrato



# Funzione per caricare le immagini
def load_images(image_folder, transform, device):
    images = []
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        images.append(image)
    return torch.cat(images)


def load_frame_sequence(frame_dir, transform, num_frames=16, device="cuda"):
    """
    Carica una sequenza di frame da una directory e li preprocessa.
    Args:
        frame_dir (str): Percorso della directory contenente i frame.
        transform (callable): Trasformazione da applicare ai frame.
        num_frames (int): Numero di frame da includere nella sequenza.
        device (str): Dispositivo su cui caricare i tensori.
    Returns:
        Tensor: Tensore della sequenza con dimensioni (1, num_frames, C, H, W).
    """
    frame_paths = sorted([os.path.join(frame_dir, fname) for fname in os.listdir(frame_dir)])

    # Assicurati di avere abbastanza frame
    if len(frame_paths) < num_frames:
        raise ValueError(f"Non ci sono abbastanza frame in {frame_dir}. Trovati: {len(frame_paths)}")

    # Seleziona i primi `num_frames`
    selected_frames = frame_paths[:num_frames]

    # Carica e trasforma i frame
    frames = []
    for frame_path in selected_frames:
        img = Image.open(frame_path).convert("RGB")
        frames.append(transform(img).unsqueeze(0))  # Aggiungi dimensione batch
    frames = torch.cat(frames, dim=0)  # Combina i frame lungo il batch

    # Aggiungi dimensione per batch e sposta sul dispositivo
    return frames.unsqueeze(0).to(device)  # (1, num_frames, C, H, W)



# Funzione per calcolare l'errore di ricostruzione (MSE e PSNR)
def reconstruction_metrics(original, reconstructed):
    mse = F.mse_loss(reconstructed.logit(), original).item()
    psnr = 20 * np.log10(1.0) - 10 * np.log10(mse)
    return mse, psnr

# Configurazione dei parametri
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#pretrained_model_path = './pytorch_model.bin'  # Modello preaddestrato
#specialized_model_path = './output_old2/checkpoint-149.pth'  # Modello addestrato
#image_folder = './sequenced_imgs/freq-1.6_part3'  # Cartella con le immagini di test




#region Funzioni per eseguire l'inferenza

def predict_label(model, videos):    
    #with torch.no_grad():
    with torch.cuda.amp.autocast():
        logits = model(videos)  # (B, nb_classes)
        predicted_classes = torch.argmax(logits, dim=1)  # intero con l'indice di classe
    
    return predicted_classes

def get_path_pred_label(model, data_loader):
    all_paths = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for videos, labels, folder_path in data_loader:
            videos = videos.to(device, non_blocking=True)
            predicted_classes = predict_label(model, videos) # shape (batch, num_class)
            labels = labels.detach().cpu().numpy()
            pred_classes = predicted_classes.detach().cpu().numpy()
            
            all_labels.extend(labels)
            all_preds.extend(pred_classes)
            all_paths.extend(folder_path)

    return all_paths, all_preds, all_labels

def create_df_predictions(all_paths, all_preds, all_labels):
    video_folder_name = pd.Series(all_paths).str.split('/').str.get(-1)
    predictions_series = pd.Series(all_preds)
    labels_series = pd.Series(all_labels)
    res_df = pd.concat([video_folder_name, predictions_series, labels_series], axis=1)
    res_df.columns = ['path', 'predictions', 'labels']

    return res_df


# se non mi servono le predizioni uso questo:
def get_only_labels(data_loader):    
    all_labels = []
    all_paths = []
    for videos, labels, folder_path in data_loader:
        labels = labels.detach().cpu().numpy()    
        all_labels.extend(labels)
        all_paths.extend(folder_path)
    
    all_preds = [0] * len(all_labels)
    return all_paths, all_preds, all_labels



def video_pred_2_img_pred(df_video_w_predictions):
    # Espando dataframe con le associazioni (path X offsets) -> predictions
    records = []
    for _, row in df_video_w_predictions.iterrows():
        for orig_path in row['orig_paths']:
            records.append({
                'path': orig_path,
                'predictions': row['predictions'],
                'tmp_label': row['labels'],
                'tile_offset_x': row['tile_offset_x'],
                'tile_offset_y': row['tile_offset_y']
            })

    # Li trasformiamo in un nuovo DataFrame
    df_mapping = pd.DataFrame(records)

    #df_mapping[['path', 'tile_offset_x', 'tile_offset_y']].duplicated().sum()
    # se ci sono path duplicati in combinazione con gli offsets
    return df_mapping

#endregion




def calc_metrics():
    args = prepare_args()
    device = torch.device(args.device)

    # Carica i modelli
    pretrained_model = get_model(args)
    pretrained_model.to(device)
    # cambio il path dentro args
    #args.init_ckpt = specialized_model_path
    #specialized_model = get_model(args)

    patch_size = pretrained_model.encoder.patch_embed.patch_size
    data_loader_test = get_dataloader(args, patch_size)

    # Ricostruzione e calcolo delle metriche
    with torch.no_grad():
        for batch in data_loader_test:
            images, bool_masked_pos, decode_masked_pos = batch

            # Sposta i dati sul dispositivo
            images = images.to(args.device)
            bool_masked_pos = bool_masked_pos.to(args.device, non_blocking=True).flatten(1).to(torch.bool)
            decode_masked_pos = decode_masked_pos.to(args.device, non_blocking=True).flatten(1).to(torch.bool)

            # Passa i dati al modello
            # Ricostruzioni dei modelli
            pretrained_reconstructed = pretrained_model(images, bool_masked_pos, decode_masked_pos)
            #specialized_reconstructed = specialized_model(images, bool_masked_pos, decode_masked_pos)

            #print("Shape dell'output:", outputs.shape)
            # Calcola le metriche
            #pretrained_mse, pretrained_psnr = reconstruction_metrics(images, pretrained_reconstructed)
            #specialized_mse, specialized_psnr = reconstruction_metrics(images, specialized_reconstructed)

    # Stampa i risultati
    print('=== Risultati delle Metriche di Ricostruzione ===')
    #print(f'Modello Preaddestrato - MSE: {pretrained_mse:.4f}, PSNR: {pretrained_psnr:.2f} dB')
    #print(f'Modello Specializzato - MSE: {specialized_mse:.4f}, PSNR: {specialized_psnr:.2f} dB')

    # Conclusione
    #if specialized_mse < pretrained_mse:
    #    print('Il modello specializzato ha una migliore ricostruzione!')
    #else:
    #    print('Il modello preaddestrato ha una migliore ricostruzione.')

if __name__ == "__main__":
    calc_metrics(image_folder)