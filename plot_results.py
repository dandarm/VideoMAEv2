import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
plt.rcParams.update({'font.size': 15})



def collect_data(log_file_path):
    train_epochs = []
    train_losses = []

    test_epochs = []
    test_losses = []

    val_epochs = []
    val_losses = []
    val_accs, val_bal_acc, val_pod, val_far  = [], [], [], []
    #val_fprs = []
    #val_fnrs = []

    val2_losses = []
    val2_accs, val2_bal_acc, val2_pod, val2_far = [], [], [], []
    #val2_fprs = []
    #val2_fnrs = []


    lr_epochs = []

    with open(log_file_path, 'r', encoding="utf-8") as file:
        val_epoch = None
        
        for line in file:
            data = json.loads(line)
            epoch = data["epoch"] + 1
            
            if "train_loss" in data:
                train_loss = data["train_loss"]        
                if epoch not in train_epochs:
                    train_epochs.append(epoch)
                    train_losses.append(train_loss)
                    if 'train_lr' in data:
                        lr_epochs.append(data['train_lr'])

            if "test_loss" in data:
                test_loss = data["test_loss"]
                if epoch not in test_epochs:
                    test_epochs.append(epoch)
                    test_losses.append(test_loss)        
                
            
            # Controlliamo se c'è una validation loss e accuracy
            if "val_loss" in data:
                #print(line)
                val_epoch = epoch
                val_losses.append(data["val_loss"])
                val_epochs.append(val_epoch)

                if "val_acc" in data:
                    val_accs.append(data["val_acc"])
                if "val_bal_acc" in data:
                    val_bal_acc.append(data["val_bal_acc"])
                if "val_pod" in data:
                    val_pod.append(data["val_pod"])
                if "val_far" in data:
                    val_far.append(data["val_far"])

                #if 'val_fpr' in data:
                #    val_fprs.append(data['val_fpr'])
                #if 'val_fnr' in data:
                #    val_fnrs.append(data['val_fnr'])

            if "val2_loss" in data:
                #print(line)
                val2_losses.append(data["val2_loss"])

                if "val2_acc" in data:
                    val2_accs.append(data["val2_acc"])

                if 'val2_bal_acc' in data:
                    val2_bal_acc.append(data['val2_bal_acc'])
                if 'val2_pod' in data:
                    val2_pod.append(data['val2_pod'])
                if 'val2_far' in data:
                    val2_far.append(data['val2_far'])


            
    return train_epochs, train_losses, test_epochs, test_losses, val_epochs, val_losses, val_accs, lr_epochs, val_bal_acc, val_pod, val_far, val2_losses, val2_accs, val2_bal_acc, val2_pod, val2_far
                
            
# Ogni 10 epoche, salviamo la media della validation loss e accuracy
#if val_epoch is not None and val_epoch % args.VAL_FREQ == 0:
    #avg_val_loss = sum(val_loss_accumulator) / len(val_loss_accumulator)
    #avg_val_acc = sum(val_acc_accumulator) / len(val_acc_accumulator)
    #val_epochs.append(val_epoch)
    #val_losses.append(avg_val_loss)
    #val_accs.append(avg_val_acc)
    #val_loss_accumulator = []  # Reset dell'accumulatore
    #val_acc_accumulator = []
    #val_epoch = None


def axis_color(ax, colore_asse):
    ax.yaxis.label.set_color(colore_asse)
    ax.axis["right"].label.set_color(colore_asse)
    ax.axis["right"].major_ticks.set_color(colore_asse)
    ax.axis["right"].major_ticklabels.set_color(colore_asse)
    ax.axis["right"].line.set_color(colore_asse)


def plot_training_curves(tuple_vars, plot_file_name=None, log=True):

    #(train_epochs, train_losses, test_epochs, test_losses, val_epochs, val_losses, val_accs, lr_epochs) = tuple_vars
    #(train_epochs, train_losses, test_epochs, test_losses, val_epochs, val_losses, val_accs, lr_epochs, val_fprs, val_fnrs, val2_losses, val2_accs, val2_fprs, val2_fnrs) = tuple_vars
    (train_epochs, train_losses, test_epochs, 
        test_losses, val_epochs, val_losses, val_accs, 
        lr_epochs, val_bal_acc, val_pod, val_far, val2_losses, 
        val2_accs, val2_bal_acc, val2_pod, val2_far) = tuple_vars
    

    # Plot del grafico
    fig = plt.figure(figsize=(20, 10))

    ax1 = host_subplot(111, axes_class=AA.Axes, figure=fig)
    #plt.subplots_adjust(right=0.75)
    ax2 = ax1.twinx()

    # Asse sinistro per la loss
    ax1.plot(train_epochs, train_losses, marker='.', label='Training Loss') #, marker='o', linestyle='')
    if test_losses is not None and len(test_losses) > 0:
        ax1.plot(test_epochs, test_losses, label='Test Loss') #color='r', marker='s', 
    ax1.plot(val_epochs, val_losses, marker='.', label='Validation Loss') #color='r', marker='s', 
    if val2_losses is not None and len(val2_losses) > 0:
        ax1.plot(val_epochs, val2_losses, marker='.', label='Validation2 Loss', color='peachpuff') # marker='s',
    

    tick_length = 20
    tick_width  = 180
    set_ticklines(ax1, tick_length, tick_width)


    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    if log:
        ax1.set_yscale('log')
        ax1.set_xscale('log')

    if len(val_accs)>0:
        # Asse destro per l'accuracy
        
        ax2.axis["right"].toggle(all=True)
        p2 = ax2.plot(val_epochs, val_accs, color='g', marker='.', label='Validation Accuracy')        
        #p2 = ax2.plot(val_epochs, val_bal_acc, color='turquoise', marker='.', label='Validation Balanced Accuracy')
        colore_asse = p2[0].get_color()
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, color=colore_asse, linestyle='--', linewidth=1.5, axis='y',  )
        #ax2.yaxis.set_major_locator(MultipleLocator(5))
        #ax2.legend(loc='center right')
        ax2.legend(loc='lower left')
        ax2.set_ylim(0,1)

        axis_color(ax2, colore_asse)


    if len(lr_epochs) > 0:
        ax3 = ax1.twinx()
        new_fixed_axis = ax3.get_grid_helper().new_fixed_axis
        ax3.axis["right"] = new_fixed_axis(loc="right", axes=ax3, offset=(100, 0))
        ax3.axis["right"].toggle(all=True)

        p3 = ax3.plot(train_epochs, lr_epochs, label='Learning rate', color='black')
        colore_asse = p3[0].get_color()
        ax3.set_ylabel('Learning rate')
        #ax3.spines['right'].set_position(('outward', 60))
        set_ticklines(ax3, 190, tick_width)

        if log:
            ax3.set_yscale('log')
            ax3.set_xscale('log')

        axis_color(ax3, colore_asse)

    plt.title('Training Loss, Validation Loss, and Accuracy per Epoch')
    fig.canvas.draw()

    if plot_file_name is not None:
        plt.savefig(plot_file_name)
     
        ### plot POD FAR
        plot_file_name = plot_file_name.replace('.png', '_podfar.png')

    if  len(val_pod) > 0 and len(val_far) > 0:
        if len(val2_pod) > 0 and len(val2_far) > 0:
            plot_podfar(val_bal_acc, val_pod, val_far, val_epochs, val2_bal_acc, val2_pod, val2_far, plot_file_name=plot_file_name, log=log)
        else:
            plot_podfar(val_accs, val_pod, val_far, val_epochs, plot_file_name=plot_file_name, log=log)
        


def plot_podfar(val_accs, val_pods, val_fars, val_epochs, val2_accs=None, val2_pods=None, val2_fars=None, plot_file_name=None, log=True):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax2 = ax.twinx()    

    ax.plot(val_epochs, val_pods, marker='.', label='Validation_1 POD', color='b')
    ax.plot(val_epochs, val_fars, marker='.', label='Validation_1 FAR', color='r')
    

    if val2_accs is not None and val2_pods is not None and val2_fars is not None:
        ax.plot(val_epochs, val2_pods, marker='.', label='Validation_2 POD', color='lightblue')
        ax.plot(val_epochs, val2_fars, marker='.', label='Validation_2 FAR', color='lightcoral')

    ax.set_ylim(0,1)
    ax.set_yticks(np.arange(0, 0.6, 0.1))
        

    
    p2 = ax2.plot(val_epochs, val_accs, marker='.', label='Validation_1 Balanced Accuracy', color='g')
    ax2.plot(val_epochs, val2_accs, marker='.', label='Validation_2 Balanced Accuracy', color='lightgreen')
    #ax2.axis["right"].toggle(all=True)
    colore_asse = p2[0].get_color()
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, color=colore_asse, linestyle='--', linewidth=1.5, axis='y',  )
    #ax2.yaxis.set_major_locator(MultipleLocator(5))

    # Ticks per accuracy tra 0 e 1
    ticks = np.arange(0.65, 1.01, 0.1)
    ax2.set_yticks(ticks)


    #ax2.legend(loc='center right')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0,1)

    #axis_color(ax2, colore_asse)


    ax.set_xlabel('Epochs')
    ax.set_ylabel('POD/FAR')
    ax.grid(True)
    ax.legend(loc='lower left')
    if log:
        #ax.set_yscale('log')
        ax.set_xscale('log')

    plt.title('Validation POD and FAR per Epoch')

    if plot_file_name is not None:
        plt.savefig(plot_file_name)

    plt.show()

def set_ticklines(ax1, tick_length, tick_width):
    ax1.tick_params(
        axis='both',       # 'x', 'y' o 'both'
        which='both',      # 'major', 'minor' o 'both'
        width=3,           # spessore
        length=50, 
        direction='in', # tick dentro e fuori
    )

    for side in ["bottom", "left"]:
        ax = ax1.axis[side]
        ticks = ax.major_ticks        # l'oggetto Ticks delle major

        # imposta lunghezza e spessore
        ticks.set_ticksize(tick_length)
        ticks.set_linewidth(tick_width)
        #for line in ticks.tick1line + ticks.tick2line:
        #    line.set_linewidth(tick_width)

        minors = ax.minor_ticks
        minors.set_ticksize(tick_length / 2)
        minors.set_linewidth(tick_width / 2)
