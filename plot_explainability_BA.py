import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle as pkl
import torch

def load_metadata():
    metadata = pd.read_csv("data/PEMS-BAY-META.csv")
    return metadata

def load_nodes_coordinates():
    metadata = load_metadata()
    coordinates = metadata[["Latitude","Longitude"]].values
    nodes = metadata["sensor_id"].values
    return coordinates, nodes

def load_adj_mx():
    adj_matrix = np.load("data/pems_adj_mat.npy")
    return adj_matrix

def node_directions():
    metadata = load_metadata()
    directions = metadata["Dir"].values
    # N is [1, 0], S is [-1, 0], E is [0, 1], W is [0, -1]
    directions = np.array([[1, 0] if direction == "N" else [-1, 0] if direction == "S" else [0, 1] if direction == "E" else [0, -1] for direction in directions])
    return directions

def node_directions_labels():
    metadata = load_metadata()
    directions = metadata["Dir"].values
    return directions

def main():
    coordinates, coord_node_order = load_nodes_coordinates()
    # sort the coord_node_order and coordinates by the order of the nodes
    coordinates = coordinates[np.argsort(coord_node_order)]
    directions = node_directions_labels()
    point_shapes = np.array(["o" if direction == "N" else "v" if direction == "S" else ">" if direction == "E" else "<" for direction in directions])
    print("point_shapes:", point_shapes)

    # scatter plot of the nodes and color the node 0 differently
    node = 110
    colors = ["red" if i == node else "blue" for i in range(coordinates.shape[0])]
    sizes = np.array([200 if i == node else 30 for i in range(coordinates.shape[0])])

    # load adj_mx_bay.pkl
    with open("data/adj_mx_bay.pkl", "rb") as f:
        adj_mx_all_info = pkl.load(f, encoding="latin1")

    # get the adjacency matrix
    adj_mx = adj_mx_all_info[2]

    # take off self-loops
    idx = np.arange(adj_mx.shape[0])
    adj_mx[idx, idx] = 0

    # create graph G from adj_mx
    G = nx.from_numpy_matrix(adj_mx)

    # load the node_masks.pt file to get the continous node masks that we will plot as colors
    full_masks = torch.load("node_mask.pt", map_location="cpu")
    full_masks = full_masks.detach().numpy()
    full_masks = full_masks[:,:]

    if len(full_masks.shape) > 1:
        node_masks = np.mean(full_masks, axis=1)
    else:
        node_masks = full_masks

    # plot the nodes in the coordinates of the nodes and the node_masks as colors, without using nx
    # the sizes of the nodes are defined by the sizes array
    # the shapes of the nodes are defined by the directions
    plt.figure(figsize=(8, 8))
    pos = dict(zip(range(coordinates.shape[0]), coordinates))
    shape_types = ['o', 'v', '>', '<']
    # share the same colorbar for all the shapes, such that the minimum and maximum values are the same
    minimum = np.min(node_masks)
    maximum = np.max(node_masks)
    for shape in shape_types:
        mask = np.where(point_shapes == shape)[0]
        if len(mask) > 0:
            plt.scatter(coordinates[mask, 1], coordinates[mask, 0], s=sizes[mask], c=node_masks[mask], cmap="OrRd", marker=shape, vmin=minimum, vmax=maximum)

    plt.colorbar()
    plt.title("PEMS-BAY Network")
    plt.show()

    # heatmap of the full_masks
    to_plot = full_masks
    # sort the nodes by the sum of the node_masks descending
    to_plot_order = np.argsort(np.sum(to_plot, axis=1))[::-1]
    to_plot = to_plot[to_plot_order]
    to_plot = to_plot.T

    plt.figure(figsize=(8, 8))
    plt.imshow(to_plot, cmap="OrRd", aspect="auto")
    # x axis is the nodes, y axis are the features (2 features with 12 time-steps each)
    plt.xlabel("Nodes")
    plt.ylabel("Features")
    plt.colorbar()
    plt.title("Node Masks")
    plt.show()

    from matplotlib.colors import Normalize, to_rgba_array

    # Normalize and get colormap
    norm = Normalize(vmin=full_masks.min(), vmax=full_masks.max())
    cmap = plt.cm.OrRd

    to_plot = full_masks
    # sort the nodes by the sum of the node_masks descending
    to_plot_order = np.argsort(np.sum(to_plot, axis=1))[::-1]
    to_plot_order_inv = np.argsort(to_plot_order)
    to_plot = to_plot[to_plot_order]
    to_plot = to_plot.T

    # Compute the colors
    colors = cmap(norm(to_plot))

    # Create scatter plot and heatmap
    fig, (ax1, ax2) = plt.subplots(2,1)

    # Scatter plot
    sc = ax1.scatter(coordinates[:, 1], coordinates[:, 0], c=node_masks, s=sizes, cmap="OrRd")
    ax1.set_title('Nodes Scatter Plot')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')

    # Initial heatmap with colors
    hm = ax2.imshow(colors, aspect='auto')
    ax2.set_title('Feature Importance Heatmap')
    ax2.set_xlabel('Feature')
    ax2.set_ylabel('Node')

    # Function to update heatmap colors
    def update_heatmap(selected_index=None):
        new_colors = colors.copy()
        if selected_index is not None:
            # Set all alpha values to 0.1
            new_colors[..., -1] = 0.1  # Set alpha of all to 0.1
            # Set alpha of selected row to 1
            new_colors[:, selected_index, -1] = 1
        else:
            # Reset all alpha values to 1
            new_colors[..., -1] = 1
        hm.set_data(new_colors)
        fig.canvas.draw()

    # Callback function for click event
    def onpick(event):
        if event.artist != sc:
            # Clicked outside the scatter plot points, reset heatmap
            update_heatmap(None)
        else:
            # Clicked on scatter plot point
            ind = event.ind[0]
            x, y = coordinates[ind]
            print(f'Clicked on node {ind} at ({x}, {y})')
            # Update heatmap to highlight selected node
            update_heatmap(to_plot_order_inv[ind])

    # Connect pick event for scatter plot
    fig.canvas.mpl_connect('pick_event', onpick)

    # Connect button_press_event for general clicks
    def on_click(event):
        if event.inaxes != ax1:
            # Clicked outside the scatter plot axis, reset heatmap
            update_heatmap(None)

    fig.canvas.mpl_connect('button_press_event', on_click)

    # Enable picking on scatter plot points
    sc.set_picker(True)

    plt.show()



if __name__ == "__main__":
    main()