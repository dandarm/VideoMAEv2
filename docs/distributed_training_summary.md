# Distribuzione multi-GPU e multi-nodo in VideoMAEv2

Questo promemoria elenca dove viene configurato l'uso di più GPU/nodi con PyTorch e Slurm e fornisce uno schema riutilizzabile per replicare lo stesso setup in altri repository.

## Inizializzazione distribuita lato PyTorch
L'unica funzione di utilità centrale è `utils.get_resources()`, che autodetecta il tipo di lancio (torchrun/mpirun/srun) e restituisce rank globale, rank locale, world size, task per nodo e numero di worker CPU da usare. Per comodità, la funzione è riportata qui integralmente:

```python
def get_resources():

    rank = 0
    local_rank = 0
    world_size = 1

    if os.environ.get("RANK"):
        # launched with torchrun (python -m torch.distributed.run)
        rank = int(os.getenv("RANK"))
        local_rank = int(os.getenv("LOCAL_RANK"))
        world_size = int(os.getenv("WORLD_SIZE"))
        local_size = int(os.getenv("LOCAL_WORLD_SIZE"))
        if rank == 0:
            print("launch with torchrun")

    elif os.environ.get("OMPI_COMMAND"):
        # launched with mpirun
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
        if rank == 0:
            print("launch with mpirun")

    elif os.environ.get("SLURM_PROCID"):
        # launched with srun (SLURM)
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NPROCS"])
        local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
        if rank == 0:
           print("launch with srun")

    else:
        rank = 0
        local_rank = rank
        world_size = 1
        local_size=1

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 10))

    return rank, local_rank, world_size, local_size, num_workers
```

Con queste informazioni, gli script impostano il device corretto, inizializzano `torch.distributed.init_process_group` con backend `nccl` e scalano batch size/learning rate per `world_size`.【F:utils.py†L427-L468】

## Script di training/inference che attivano il parallelismo
- **Pretraining (specialization)**: `launch_specialization_training` richiama `get_resources`, inizializza il process group `nccl`, assegna la GPU locale e avvolge il modello con `DistributedDataParallel` quando `world_size>1`. Batch size e LR vengono scalati con `world_size`.【F:specialization.py†L40-L103】
- **Fine-tuning di classificazione**: `classification.launch_classification` usa `get_resources` per settare rank/world size, chiama `init_process_group` se necessario, imposta la GPU con `torch.cuda.set_device` e incapsula il modello con `DistributedDataParallel` per sincronizzare i gradienti sui processi. Anche qui LR e batch size sono scalati per il numero totale di GPU.【F:classification.py†L82-L110】【F:classification.py†L155-L179】【F:classification.py†L188-L204】
- **Inference distribuita (classificazione)**: stessa logica di discovery delle risorse e inizializzazione `nccl`, utile per batch paralleli durante la generazione di embedding o predizioni.【F:inference_classification.py†L98-L118】
- Altri script (tracking, inference_tracking) seguono lo stesso schema basato su `utils.get_resources()` e `torch.distributed.init_process_group`.

## Script Slurm per multi-nodo/multi-GPU
I job `sbatch_*.sh` mostrano come preparare l'ambiente Slurm, settare `MASTER_ADDR`/`MASTER_PORT` e lanciare con `mpirun` (o, in alternativa commentata, `torch.distributed.run`). Esempi chiave:
- `sbatch_job.sh`/`sbatch_neighboring.sh`: training di classificazione su 4 nodi × 4 GPU, con caricamento dei moduli HPC e lancio `mpirun python classification.py ...`.【F:sbatch_job.sh†L2-L25】【F:sbatch_neighboring.sh†L2-L18】
- `sbatch_tracking.sh`: tracking distribuito con la stessa configurazione e variabili master esplicite.【F:sbatch_tracking.sh†L2-L25】
- `sbatch_pred.sh`, `sbatch_pred_val2.sh`, `sbatch_pred_tracking.sh`: job di inference distribuita (classificazione o tracking) con le stesse direttive `#SBATCH` e lancio `mpirun` su tutte le GPU/nodi.【F:sbatch_pred.sh†L2-L29】【F:sbatch_pred_val2.sh†L2-L30】【F:sbatch_pred_tracking.sh†L2-L30】

### Script `sh` di lancio multi-GPU/multi-nodo (esempio riutilizzabile)
Per allinearsi agli script del repository, sotto è riportato **il contenuto integrale di `sbatch_job.sh`**, che lancia `classification.py` con `mpirun` su 4 nodi × 4 GPU su Slurm. Puoi riusarlo come modello per altri job modificando direttive `#SBATCH`, moduli, entry point Python e argomenti.

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=17:58:00
#SBATCH --error=myJob.err
#SBATCH --output=myJob_medicanes.out

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/videomae/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12340


mpirun --map-by socket:PE=4 --report-bindings python classification.py --on leonardo

#srun --ntasks-per-node=4 \
#python -m torch.distributed.run \
#    --nproc_per_node 4 \
#    --nnodes 4 \
#    specialization.py
# module load openmpi
```

Note operative:
- `mpirun` popola le variabili `OMPI_COMM_WORLD_*` lette da `utils.get_resources()` per determinare `rank`, `local_rank`, `world_size` e per inizializzare `torch.distributed` con backend `nccl`.
- `MASTER_ADDR` e `MASTER_PORT` sono impostati nel contesto Slurm per permettere l'inizializzazione del process group tra nodi diversi; assicurati che la porta sia libera.
- Le direttive `#SBATCH` controllano il numero di nodi, task/GPU per nodo e risorse CPU. Se cambi il numero di GPU per nodo, aggiorna `--ntasks-per-node` e l'opzione `--map-by` coerentemente.
- Per adattare lo script a un altro entry point (`specialization.py`, `tracking.py`, ecc.) basta sostituire il comando `python classification.py ...` preservando il prefisso `mpirun` (oppure decommentare il blocco `torch.distributed.run` se preferisci `torchrun`).

## Esempio di pretraining multi-nodo già documentato
`docs/PRETRAIN.md` include uno script di esempio che usa `torch.distributed.launch` impostando `MASTER_PORT`, `N_NODES`, `GPUS_PER_NODE`, `node_rank` e `master_addr`, utile come modello per altri job `srun`/Slurm. Il comando passa `--nnodes`, `--node_rank` e `--master_addr` a `torch.distributed.launch` prima di chiamare `run_mae_pretraining.py`.【F:docs/PRETRAIN.md†L82-L130】

## Ricetta riutilizzabile per altri repository
1. **Rileva il contesto di lancio**: leggi `RANK`, `LOCAL_RANK`, `WORLD_SIZE` e le variabili SLURM (`SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS`, `MASTER_ADDR`, `MASTER_PORT`) per determinare rank locale e globale, e il numero di worker CPU disponibili.
2. **Imposta il device**: chiama `torch.cuda.set_device(local_rank)` prima di creare il modello o caricare dati sul device.
3. **Inizializza il process group**: usa `torch.distributed.init_process_group(backend="nccl", init_method=<tcp://MASTER_ADDR:PORT>, world_size=..., rank=...)` e sincronizza con `torch.distributed.barrier()` per stabilire la comunicazione tra processi.
4. **Avvolgi il modello**: racchiudi il modello in `torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)` per sincronizzare i gradienti e supportare multi-GPU/multi-nodo.
5. **Scala gli iperparametri globali**: moltiplica batch size e learning rate per `world_size` per mantenere costante la scala effettiva del batch, come avviene negli script di classificazione/pretraining.
6. **Lancio Slurm**: crea uno script `sbatch` che definisce `--nodes`, `--ntasks-per-node` e `--gres=gpu:<per-node>`, imposta `MASTER_ADDR`/`MASTER_PORT`, carica i moduli necessari e lancia con `mpirun` o `torchrun` (`python -m torch.distributed.run --nproc_per_node <gpus> --nnodes <nodes> ...`).

Seguendo questi punti si può replicare rapidamente il setup di addestramento e inference distribuiti di VideoMAEv2 in altri progetti PyTorch.
