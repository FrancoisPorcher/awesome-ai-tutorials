# utils.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Improved visualization

def plot_row_pe(token_positions=[10, 40, 80], pos_enc=None):
    """
    Plot positional encoding values for given token positions.

    :param token_positions: List of token positions.
    :param pos_enc: Positional encoding values.
    """
    # Using Seaborn's style
    sns.set_style("whitegrid")

    # Define color palette
    colors = sns.color_palette("husl", 2)  # "husl" palette is colorful yet distinct

    # Create a subplot of 3 rows and 1 column
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharey=True)

    for idx, token_position in enumerate(token_positions):
        ax = axes[idx]

        row_with_pair_indices = pos_enc[token_position, 0::2]
        row_with_impair_indices = pos_enc[token_position, 1::2]
        
        x_pair = np.arange(0, 512, 2)
        x_impair = np.arange(1, 512, 2)
        
        ax.plot(x_pair, row_with_pair_indices.numpy(), color=colors[0], linewidth=2, label='Even Indices')
        ax.plot(x_impair, row_with_impair_indices.numpy(), color=colors[1], linewidth=2, label='Odd Indices')
        
        ax.set_xlabel('Dimension Coordinate', fontsize=12)
        ax.set_title(f'Token Position {token_position}', fontsize=14)
        ax.legend(fontsize=10)

    # Common Y Label
    axes[0].set_ylabel('Positional Encoding Value', fontsize=12)

    plt.tight_layout()  # Adjust spacing between subplots
    plt.suptitle('Positional Encoding across Different Token Positions', fontsize=16, y=1.1)
    plt.show()


# Define the positional encoding function from the previous response here...

def visualize_positional_encoding(pos_enc):
    """
    Visualize the positional encodings.

    :param position: Maximum length of the sequence.
    :param d_model: Dimension of the model.
    """
    # Generate positional encodings
    pe = pos_enc.numpy()
    position = pe.shape[0]
    d_model = pe.shape[1]

    # Plot
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(pe, cmap='viridis')
    plt.xlabel('Embedding Dimensions')
    plt.xlim((0, d_model))
    plt.ylim((position, 0))
    plt.ylabel('Token Position')
    plt.colorbar()
    plt.title('Positional Encoding')
    plt.show()
    

def visualize_positional_encoding(pos_enc):
    """
    Visualize the positional encodings using the 'magma' colormap.

    :param pos_enc: Positional encoding matrix.
    """
    position, d_model = pos_enc.shape

    # Styling with Seaborn
    sns.set_style("whitegrid")
    cmap = "magma"

    # Plot
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(
        pos_enc, 
        cmap=cmap, 
        cbar_kws={'label': 'Encoding Value Magnitude'},
        xticklabels=50,
        yticklabels=10
    )
    
    # Titles and labels
    ax.set_title('Positional Encoding Visualization', fontsize=16)
    ax.set_xlabel('Embedding Dimensions', fontsize=14)
    ax.set_ylabel('Token Position', fontsize=14)

    plt.show()
    
    
def visualize_column_pe(dimensions=[10, 100, 300], pos_enc=None):
    """
    Plot positional encoding values for given embedding dimensions.

    :param dimensions: List of embedding dimensions to inspect.
    :param pos_enc: Positional encoding values.
    """
    # Using Seaborn's style
    sns.set_style("whitegrid")

    # Define color palette
    colors = sns.color_palette("husl", 2)  # "husl" palette is colorful yet distinct

    # Create a subplot of 3 rows and 1 column
    fig, axes = plt.subplots(len(dimensions), 1, figsize=(18, 14), sharey=True)

    for idx, dimension in enumerate(dimensions):
        ax = axes[idx]

        # Plotting the provided dimension
        ax.plot(pos_enc[:, dimension], color=colors[0], linewidth=2, label=f'Dimension {dimension}')

        # Plotting the dimension + 1
        ax.plot(pos_enc[:, dimension + 1], color=colors[1], linewidth=2, label=f'Dimension {dimension + 1}')
        
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_title(f'Embedding Dimensions {dimension} & {dimension + 1}', fontsize=14)
        ax.legend(fontsize=10)

    # Common Y Label
    axes[0].set_ylabel('Positional Encoding Value', fontsize=12)

    plt.tight_layout()  # Adjust spacing between subplots
    plt.suptitle('Positional Encoding for Specific Embedding Dimensions', fontsize=16, y=1.1)
    plt.show()






