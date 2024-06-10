import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle as pkl

def load_metadata(data_type="bay"):
    if data_type == "bay":
        metadata = pd.read_csv("data/PEMS-BAY-META.csv")
    elif data_type == "la":
        metadata = pd.read_csv("data/METR-LA-META.csv")
    return metadata

def load_nodes_coordinates(data_type):
    metadata = load_metadata(data_type)
    coordinates = metadata[["Latitude","Longitude"]].values
    nodes = metadata["sensor_id"].values
    if data_type == "bay":
        # sort the nodes by the order of the coordinates
        coord_node_order = np.argsort(nodes)
        nodes = nodes[coord_node_order]
        coordinates = coordinates[coord_node_order]
    return coordinates, nodes

def load_adj_mx():
    adj_matrix = np.load("data/pems_adj_mat.npy")
    return adj_matrix

def load_adj_pkl(data_type):
    suffix = "METR-LA" if data_type == "la" else "BAY"
    with open(f"data/adj_mx_{suffix}.pkl", "rb") as f:
        adj_mx_all_info = pkl.load(f, encoding="latin1")
    return adj_mx_all_info

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

def plot_scatter_graph(coordinates, node_colors):
    plt.scatter(coordinates[:, 1], coordinates[:, 0], c=node_colors, cmap="viridis", s=10)
    plt.colorbar()
    plt.show()

def load_data(data_type):
    if data_type == "bay":
        data = np.load("data/pems_node_values.npy").transpose((1, 2, 0))[:,0:1,:]
    elif data_type == "la":
        data = np.load("data/node_values.npy").transpose((1, 2, 0))[:,0:1,:]
    return data

def main(data_type="bay", plot_graph=False, plot_nodes=True, plot_distributions=True, plot_graph_distributions=False):
    coordinates, coord_node_order = load_nodes_coordinates(data_type)
    # sort the coord_node_order and coordinates by the order of the nodes

    adj_mx = load_adj_pkl(data_type)[2]

    if plot_graph:
        # if it is directed
        G = nx.from_numpy_matrix(adj_mx)
        directed = nx.is_directed(G)
        print("Is the graph directed?", directed)
        # if it is weighted
        weighted = nx.is_weighted(G)
        print("Is the graph weighted?", weighted)
        # if it has self-loops
        self_loops = np.sum(np.diag(adj_mx))
        print("Does the graph have self-loops?", self_loops > 0)
        print("Number of self-loops/N:", self_loops/adj_mx.shape[0])
        
        # take off self-loops
        idx = np.arange(adj_mx.shape[0])
        adj_mx[idx, idx] = 0
        suffix = "METR-LA" if data_type == "la" else "PEMS-BAY"
        # describing the graph: average node degree, distribution of node degrees,
        # if it has self-loops, if it is fully-connected, if it is directed, 
        # if it is weighted, and more
        # average node degree
        avg_node_degree_in = np.mean(np.sum(adj_mx, axis=0))
        avg_node_degree_out = np.mean(np.sum(adj_mx, axis=1))
        print("Average node degree:", (avg_node_degree_in + avg_node_degree_out)/2)
        # distribution of node degrees
        node_degrees = (np.sum(adj_mx, axis=0) + np.sum(adj_mx, axis=1))/2
        plt.hist(node_degrees, bins=20, range=(0, 20))
        plt.title("Distribution of Node Degrees")
        plt.show()
        # distribution of node degrees with binary adjacency matrix
        binary_adj_mx = np.where(adj_mx > 0, 1, 0)
        node_degrees = (np.sum(binary_adj_mx, axis=1))# + np.sum(binary_adj_mx, axis=1))/2
        print("average binary node degree:", np.mean(node_degrees))
        plt.hist(node_degrees, bins=20, range=(0, 20))
        plt.title("Distribution of Node Degrees with Binary Adjacency Matrix")
        plt.show()
        node_hubs = np.where(node_degrees > 12)[0]
        print("Node hubs:", node_hubs)
        plot_scatter_graph(coordinates, [1 if i in node_hubs else 0 for i in range(coordinates.shape[0])])

        # distribution of edge weights
        edge_weights = adj_mx[adj_mx > 0]
        print('Average edge weight:', np.mean(edge_weights))
        plt.hist(edge_weights, bins=20)
        plt.title("Distribution of Edge Weights")
        plt.show()
        # number of edges
        num_edges = np.sum(adj_mx > 0)
        edge_density = num_edges/(adj_mx.shape[0]**2 - adj_mx.shape[0])
        print("Number of edges:", num_edges)
        print("Edge density:", edge_density)
        # if it is fully-connected
        fully_connected = np.all(adj_mx)
        print("Is the graph fully-connected?", fully_connected)
        # if it is connected
        G = nx.from_numpy_matrix(adj_mx)
        connected = nx.is_connected(G)
        print("Is the graph connected?", connected)
        # if not connected, which nodes are not connected
        if not connected:
            # get largest connnected component
            largest_cc = max(nx.connected_components(G), key=len)
            print("number disconnected nodes is:", adj_mx.shape[0] - len(largest_cc))

            # get diameter of largest cc
            largest_cc = G.subgraph(largest_cc)
            diameter = nx.diameter(largest_cc)
            print("Diameter of largest connected component:", diameter)
            # get average shortest path length
            avg_shortest_path = nx.average_shortest_path_length(largest_cc)
            print("Average shortest path length of largest connected component:", avg_shortest_path)
                


        # plot the different centralities of the nodes
        # degree centrality
        degree_centrality = nx.degree_centrality(G)
        # closeness centrality
        closeness_centrality = nx.closeness_centrality(G)
        # betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(G)
        # eigenvector centrality
        eigenvector_centrality = nx.eigenvector_centrality(G)
        # plot the centralities
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        pos = dict(zip(range(coordinates.shape[0]), coordinates))
        nx.draw(G, pos, node_size=10, node_color=list(degree_centrality.values()), cmap="viridis", font_size=8, font_color="black", font_weight="bold", arrows=True, arrowsize=20, ax=ax)
        ax.set_title(f"{suffix} Network, degree centrality")
        plt.show()

        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        pos = dict(zip(range(coordinates.shape[0]), coordinates))
        nx.draw(G, pos, node_size=10, node_color=list(closeness_centrality.values()), cmap="viridis", font_size=8, font_color="black", font_weight="bold", arrows=True, arrowsize=20, ax=ax)
        ax.set_title(f"{suffix} Network, closeness centrality")
        plt.show()

        # node clustering coefficient
        clustering_coefficient = nx.clustering(G)
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        pos = dict(zip(range(coordinates.shape[0]), coordinates))
        nx.draw(G, pos, node_size=10, node_color=list(clustering_coefficient.values()), cmap="viridis", font_size=8, font_color="black", font_weight="bold", arrows=True, arrowsize=20, ax=ax)
        ax.set_title(f"{suffix} Network, clustering coefficient")
        plt.show()

        # average clustering coefficient
        avg_clustering_coefficient = nx.average_clustering(G)
        print("Average clustering coefficient:", avg_clustering_coefficient)


        # 

    if plot_nodes:
        # load data
        data = load_data(data_type)

        # get statistics of data
        print("Data shape:", data.shape)
        print("Data mean:", np.mean(data))
        print("Data std:", np.std(data))

        if data_type=='la':
            # take out the values that are 0
            data_non_zero = data[data != 0]
            # get statistics of data
            print("Data mean:", np.mean(data_non_zero))
            print("Data std:", np.std(data_non_zero))

        # plot distribution of node values
        plt.hist(data.flatten(), bins=20)
        plt.title("Distribution of Node Values")
        plt.show()

        # plot the mean of the values in each node
        node_means = [np.mean(data[i][data[i]>0]<60) for i in range(data.shape[0])]
        node_stds = [np.std(data[i][data[i]>0]) for i in range(data.shape[0])]
        fig, axis = plt.subplots(1, 2, figsize=(12, 6))
        
        axis[0].scatter(coordinates[:, 1], coordinates[:, 0], c=node_means, cmap="viridis", s=10)
        axis[0].set_title("Mean of Node Values")
        # plot the std of the values in each node
        axis[1].scatter(coordinates[:, 1], coordinates[:, 0], c=node_stds, cmap="viridis", s=10)
        axis[1].set_title("Std of Node Values")
        plt.show()

        if data_type=='la':
            # plot the amount of zero values in each node
            node_zeros = [np.sum(data[i] == 0) for i in range(data.shape[0])]
            plt.scatter(coordinates[:, 1], coordinates[:, 0], c=node_zeros, cmap="viridis", s=10)
            plt.colorbar()
            plt.title("Amount of Zero Values in Each Node")
            plt.show()


    if plot_graph_distributions:
        adj_mx_la = load_adj_pkl("la")[2]
        adj_mx_bay = load_adj_pkl("bay")[2]

        # take off self-loops
        idx = np.arange(adj_mx_bay.shape[0])
        adj_mx_bay[idx, idx] = 0
        idx = np.arange(adj_mx_la.shape[0])
        adj_mx_la[idx, idx] = 0


        fig, axis = plt.subplots(1, 2, figsize=(12, 6))

        #plot both distributions of the degrees in a same plot, in a frequency plot such that the frequencies add up to 1
        axis[0].hist(np.sum(adj_mx_la, axis=0), bins=20, alpha=0.5, density=True, label="METR-LA")
        axis[0].hist(np.sum(adj_mx_bay, axis=0), bins=20, alpha=0.5, density=True, label="PEMS-BAY")
        axis[0].set_title("Distribution of Node Degrees")
        axis[0].legend()
        # delte top and right axis
        axis[0].spines['top'].set_visible(False)
        axis[0].spines['right'].set_visible(False)
        axis[0].set_xlabel('Node Degree')
        axis[0].set_ylabel('Density')

        #plot both distributions of the edge weights in a same plot, in a frequency plot such that the frequencies add up to 1
        edge_weights_la = adj_mx_la[adj_mx_la > 0]
        edge_weights_bay = adj_mx_bay[adj_mx_bay > 0]
        axis[1].hist(edge_weights_la, bins=20, alpha=0.5, density=True, label="METR-LA")
        axis[1].hist(edge_weights_bay, bins=20, alpha=0.5, density=True, label="PEMS-BAY")
        axis[1].set_title("Distribution of Edge Weights")
        axis[1].legend()
        # delte top and right axis
        axis[1].spines['top'].set_visible(False)
        axis[1].spines['right'].set_visible(False)
        axis[1].set_xlabel('Edge Weight')
        axis[1].set_ylabel('Density')
        plt.show()

        # plot the two different graphs
        bay_coordinates, bay_nodes = load_nodes_coordinates("bay")
        la_coordinates, la_nodes = load_nodes_coordinates("la")

        # no self-loops
        # idx = np.arange(adj_mx_bay.shape[0])
        # adj_mx_bay[idx, idx] = 0
        # idx = np.arange(adj_mx_la.shape[0])
        # adj_mx_la[idx, idx] = 0

        G_bay = nx.from_numpy_matrix(adj_mx_bay)
        G_la = nx.from_numpy_matrix(adj_mx_la)

        fig, axis = plt.subplots(1, 2, figsize=(12, 6))
        pos_bay = dict(zip(range(bay_coordinates.shape[0]), bay_coordinates[:,::-1]))
        pos_la = dict(zip(range(la_coordinates.shape[0]), la_coordinates[:,::-1]))
        nx.draw(G_bay, pos_bay, node_size=10, node_color="skyblue", font_size=8, font_color="black", font_weight="bold", arrows=True, arrowsize=20, ax=axis[1])
        nx.draw(G_la, pos_la, node_size=10, node_color="skyblue", font_size=8, font_color="black", font_weight="bold", arrows=True, arrowsize=20, ax=axis[0])
        axis[1].set_title("PEMS-BAY Network")
        axis[0].set_title("METR-LA Network")
        plt.show()




    if plot_distributions:
        data_bay = load_data("bay")
        data_la = load_data("la")
        data_la_non_zero = data_la[data_la != 0]

        # print number of zeros of la data
        print("Number of zeros in METR-LA data:", np.sum(data_la == 0))
        # and the frequency of zeros
        print("Frequency of zeros in METR-LA data:", np.sum(data_la == 0)/data_la.size)

        print(f'METR-LA mean: {data_la_non_zero.mean()}')
        print(f'PEMS-BAY mean: {data_bay.mean()}')

        # plot both distributions of node values in a same plot, in a frequency plot such that the frequencies add up to 1, and put the maximum x range to 80
        plt.hist(data_la_non_zero.flatten(), bins=80, alpha=0.5, density=True, label="METR-LA", range=(0, 80))
        plt.hist(data_bay.flatten(), bins=80, alpha=0.5, density=True, label="PEMS-BAY", range=(0, 80))
        plt.title("Distribution of Sensor Readings")
        plt.legend()
        # delte top and right axis
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # speed is in mile per hour
        plt.xlabel('Speed Reading (mph)')
        plt.ylabel('Density')
        plt.show()

    



if __name__ == "__main__":
    # parser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="bay") # bay or la
    parser.add_argument("--plot_graph", action="store_true")
    parser.add_argument("--plot_nodes", action="store_true")
    parser.add_argument("--plot_distributions", action="store_true")
    parser.add_argument("--plot_graph_distributions", action="store_true")
    args = parser.parse_args()


    main(args.data, plot_graph=args.plot_graph, plot_nodes=args.plot_nodes, 
         plot_distributions=args.plot_distributions, plot_graph_distributions=args.plot_graph_distributions)