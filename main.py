import os
import random
from run_class_finetuning import main as main_finetuning
from run_mae_pretraining import main as main_pretraining

from arguments import prepare_args




def prepare_and_run():
    # Generate dynamic environment variables
    master_port = 12000 + random.randint(0, 20000)
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['OMP_NUM_THREADS'] = '1'

    args = prepare_args()

    # Call the imported main function with arguments
    main_pretraining(args)





if __name__ == "__main__":
    prepare_and_run()
