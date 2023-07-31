import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator

add_legend = False

# Set the path to the GraphML file you want to plot
graphml_file = "../../data/network_simulations/graphs/blue/network_size=256-alpha=5.0-resampling=False-simulation_num=99.graphml"

# alphas = ['0.5','1.0','5.0']
# network_sizes = [64,128,256]
alphas = [1.5,2.0,3.0]
network_sizes = [64,128,256]
simulation_num = [6]

# Set custom minimum and maximum values for the color scale
custom_vmin = 0
custom_vmax = 25

# Get max number of edges
# for alpha in alphas:
#     for network in network_sizes:
#         for sim in simulation_num:
#             graph = nx.read_graphml(f"../../data/network_simulations/graphs/blue/network_size={network}-alpha={alpha}-resampling=False-simulation_num={sim}.graphml")
#             # Get max number of edges
#             max_edges = max([len(graph.edges(node)) for node in graph.nodes()])
#             if max_edges > custom_vmax:
#                 custom_vmax = max_edges

# Plot and save the graph
for alpha in alphas:
    for network in network_sizes:
        for sim in simulation_num:
            graph = nx.read_graphml(f"../../data/network_simulations/graphs/blue/network_size={network}-alpha={alpha}-resampling=False-simulation_num={sim}.graphml")

            # Create a matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 10))

            # Color the nodes according to their outgoing edges
            node_degrees = [len(graph.edges(node)) for node in graph.nodes()]

            # Normalize the node degrees with custom vmin and vmax values, and create the color map
            normalize = mcolors.Normalize(vmin=custom_vmin, vmax=custom_vmax)
            cmap = cm.viridis

            # Set the layout of the nodes
            pos = nx.kamada_kawai_layout(graph)

            # Normalize node_degrees for node colors
            node_colors = [cmap(normalize(degree)) for degree in node_degrees]

            # Plot the graph without node labels
            # Set the node and edge alpha values
            edge_alpha = 0.15

            # Create an array of edge colors with the desired alpha value
            edge_colors = [(0, 0, 0, edge_alpha) for _ in graph.edges()]

            # Draw the nodes and edges with the color map
            nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors)
            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors)

            # Remove the axis and its ticks
            ax.set_axis_off()

            if add_legend:
                # Add a color bar as a legend for the node degrees
                sm = cm.ScalarMappable(cmap=cmap, norm=normalize)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
                cbar.set_label('Node degree')

                # Set the tick formatter to display only integers
                cbar.locator = MaxNLocator(integer=True)
                cbar.update_ticks()

            # Save the figure as an image (you can change the format to PDF, PNG, etc.)
            output_folder = "network_plots"
            # Create the output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            # Save the figure
            
            filename = f"size={network}-alpha={alpha}-sim={sim}.pdf"
            filepath = os.path.join(output_folder, filename)
            plt.savefig(filepath, bbox_inches='tight')

            # Close the figure
            plt.close(fig)