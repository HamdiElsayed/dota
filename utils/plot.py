
import matplotlib.pyplot as plt
import numpy as np



def plot_slice(inputs, ground_truth, outputs,slice_number,input_ray, figsize=(10,12), fontsize=10):
    """
    plot_slice: Function to plot the input geometry, ground truth dose distribution, model prediction, 
    absolute dose difference, relative error, and input ray data for a given slice number.

    Args:
        inputs (np.ndarray): The input geometry used in the prediction.
        ground_truth (np.ndarray): The ground truth dose distribution for comparison.
        outputs (np.ndarray): The model's predicted dose distribution.
        slice_number (int): The slice number to plot. This is either the x or y dimension.
        input_ray (np.ndarray): The input ray data used for the prediction.
        figsize (tuple, optional): The size of the figure. Defaults to (10,12).
        fontsize (int, optional): The fontsize of the text. Defaults to 10.
    """

    
    # Initialize figure and axes.
    fig, axs = plt.subplots(6, 1, figsize=figsize)
    axs[0].set_title("CT scan", fontsize=fontsize, fontweight='bold')
    axs[1].set_title("Target (MC)", fontsize=fontsize, fontweight='bold')
    axs[2].set_title("Predicted (model)", fontsize=fontsize, fontweight='bold')
    
    plt.subplots_adjust(hspace=0.675, wspace=0.0675)


    
    # Scale the dose values to Gy/10^9 particles
    ground_truth=ground_truth* (1e9 / 1e6)
    outputs=outputs* (1e9 / 1e6)
    
    
    min_input, max_input = np.min(inputs), np.max(inputs)
    min_output, max_output = np.min(outputs), np.max(outputs)
    min_ground_truth, max_ground_truth = np.min(ground_truth), np.max(ground_truth)


    vmin_val,vmax_val=min(min_output, min_ground_truth),max(max_output, max_ground_truth)
    cb_ticks=np.linspace(0, max_output, num=4)
    cb_ticks_gt=np.linspace(0, max_ground_truth, num=4)
    cb_ticks_input=np.linspace(min_input, max_input, num=4)
    
    
    
    # 1st row: input values
    cbh0 = axs[0].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
        cmap='gray', vmin=min_input, vmax=max_input)
    plt.sca(axs[0])
    plt.yticks([64,32,1],['2','64','128'], fontsize=fontsize)
    plt.xticks([25, 50, 75, 100, 125, 150], ['50', '100', '150', '200', '250', '300'], fontsize=fontsize)
    axs[0].set_ylabel("mm", loc='top', fontsize=fontsize)
    axs[0].set_xlabel("mm", loc='right', fontsize=fontsize)
    cb0 = fig.colorbar(cbh0, ax=axs[0], aspect=fontsize, ticks=cb_ticks_input)
    cb0.ax.set_ylabel("HU", size=fontsize)
    cb0.ax.tick_params(labelsize=fontsize)

    # 2nd row: ground truth dose distribution
    axs[1].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
        cmap='gray', alpha=0.4, vmin=min_input, vmax=max_input)
    cbh1 = axs[1].imshow(np.transpose(ground_truth[:,:,slice_number]), aspect='auto',
        cmap='turbo', alpha=0.6, vmin=min_ground_truth, vmax=max_ground_truth)
    plt.sca(axs[1])
    plt.yticks([64,32,1],['2','64','128'], fontsize=fontsize)
    plt.xticks([25, 50, 75, 100, 125, 150], ['50', '100', '150', '200', '250', '300'], fontsize=fontsize)
    axs[1].set_ylabel("mm", loc='top', fontsize=fontsize)
    axs[1].set_xlabel("mm", loc='right', fontsize=fontsize)
    cb1 = fig.colorbar(cbh1, ax=axs[1], aspect=fontsize,ticks=cb_ticks_gt)
    cb1.ax.set_ylabel(r"Gy/$10^9$ particles", size=fontsize)
    cb1.ax.tick_params(labelsize=fontsize)

    # 3rd row: model prediction 
    axs[2].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
        cmap='gray', alpha=0.4, vmin=min_input, vmax=max_input)
    cbh2 = axs[2].imshow(np.transpose(outputs[:,:,slice_number]), aspect='auto', 
        cmap='turbo', alpha=0.6, vmin=min_output, vmax=max_output)
    plt.sca(axs[2])
    plt.yticks([64,32,1],['2','64','128'], fontsize=fontsize)
    plt.xticks([25, 50, 75, 100, 125, 150], ['50', '100', '150', '200', '250', '300'], fontsize=fontsize)
    axs[2].set_ylabel("mm", loc='top', fontsize=fontsize)
    axs[2].set_xlabel("mm", loc='right', fontsize=fontsize)
    cb2 = fig.colorbar(cbh2, ax=axs[2], aspect=fontsize, ticks=cb_ticks)
    cb2.ax.set_ylabel(r"Gy/$10^9$ particles", size=fontsize)
    cb2.ax.tick_params(labelsize=fontsize)

    
    # 4th row: absolute dose difference
    axs[3].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
        cmap='gray', alpha=0.4, vmin=min_input, vmax=max_input)
    difference=np.transpose(np.absolute(ground_truth[:,:,slice_number]-outputs[:,:,slice_number]))
    cbh3 = axs[3].imshow(difference,
        aspect='auto', cmap='turbo', alpha=0.6, vmin=min_ground_truth, vmax=max_ground_truth)
        
    plt.sca(axs[3])
    plt.yticks([68,34,1], ['2','34','68'], fontsize=fontsize)
    plt.xticks([25, 50, 75, 100, 125, 150], ['50', '100', '150', '200', '250', '300'], fontsize=fontsize)
    axs[3].set_ylabel("mm", loc='top', fontsize=fontsize)
    axs[3].set_xlabel("mm", loc='right', fontsize=fontsize)
    cb3 = fig.colorbar(cbh3, ax=axs[3], aspect=fontsize,ticks=cb_ticks_gt)
    axs[3].set_title("Dose difference max={}".format(np.max(np.round(difference,2))), fontsize=fontsize, fontweight='bold')
    cb3.ax.set_ylabel(r"Gy/$10^9$ particles", size=fontsize)
    cb3.ax.tick_params(labelsize=fontsize)
    

    # 5th row: relative error
    axs[4].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
            cmap='gray', alpha=0.4)
    relative_error = difference*100/max_ground_truth
    ticks_relative_error=np.linspace(0, np.max(relative_error), num=4)
   
    cbh4 = axs[4].imshow(relative_error, aspect='auto', cmap='turbo', alpha=0.6, vmin=0, vmax=np.max(relative_error))
    axs[4].set_title(f"Relative Error: {np.round(np.median(relative_error[relative_error>0]),2)} %", fontsize=fontsize, fontweight='bold')
    cb4 = fig.colorbar(cbh4, ax=axs[4], aspect=fontsize,ticks=ticks_relative_error)
    cb4.ax.set_ylabel(r"%", size=fontsize)
    cb4.ax.tick_params(labelsize=fontsize)
    plt.sca(axs[4])
    plt.yticks([68,34,1], ['2','34','68'], fontsize=fontsize)
    plt.xticks([25, 50, 75, 100, 125, 150], ['50', '100', '150', '200', '250', '300'], fontsize=fontsize)

    ticks_ray=np.linspace(0, np.max(input_ray), num=4)

    # 6th row: input ray
    axs[5].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
            cmap='gray', alpha=0.4)
    ray_input=np.transpose(np.swapaxes(input_ray,0,2)[:,:,slice_number])
    cbh4 = axs[5].imshow(ray_input, aspect='auto', cmap='turbo', alpha=0.6)
    axs[5].set_title(f"Input Ray", fontsize=fontsize, fontweight='bold')
    cb4 = fig.colorbar(cbh4, ax=axs[5], aspect=fontsize,ticks=ticks_ray)
    cb4.ax.set_ylabel(r"[-]", size=fontsize)
    cb4.ax.tick_params(labelsize=fontsize)
    plt.sca(axs[5])
    plt.yticks([64,32,1],['2','64','128'], fontsize=fontsize)
    plt.xticks([25, 50, 75, 100, 125, 150], ['50', '100', '150', '200', '250', '300'], fontsize=fontsize)


    plt.show()



