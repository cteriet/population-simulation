# hierarchy.py

import networkx as nx

# Import pyplot here for the plot method, but handle potential ImportError
import matplotlib.pyplot as plt
import warnings  # Import warnings module


class Hierarchy:
    """Base class for Product and Action Hierarchies."""

    def __init__(self, edges):
        self.graph = nx.DiGraph()
        # Handle empty edges list
        if edges:
            self.graph.add_edges_from(edges)

        # Ensure all nodes are added even if they are only roots/parents
        all_nodes = set()
        if edges:
            for parent, child in edges:
                all_nodes.add(parent)
                all_nodes.add(child)
        # If edges is empty, there might still be nodes intended, but we can't know.
        # Assume nodes must be present in edges for now.
        self.graph.add_nodes_from(all_nodes)

        # Find roots (nodes with no parents)
        self.roots = sorted(
            [node for node, degree in self.graph.in_degree() if degree == 0]
        )
        # Find leaves (nodes with no children)
        self.leaves = sorted(
            [node for node, degree in self.graph.out_degree() if degree == 0]
        )
        # Non-leaf nodes are all nodes minus the leaves
        self.non_leaves = sorted(
            [node for node in self.graph.nodes() if node not in self.leaves]
        )
        # All nodes in a defined order
        self.all_nodes = sorted(
            list(self.graph.nodes())
        )  # Sort for consistent column ordering later

        if not self.roots and self.graph.number_of_nodes() > 0:
            warnings.warn(
                "No root nodes found in hierarchy. This might indicate a cycle or a disconnected graph."
            )
        if not self.leaves and self.graph.number_of_nodes() > 0:
            warnings.warn(
                "No leaf nodes found in hierarchy. This might indicate a cycle or nodes with only outgoing edges."
            )
        if len(self.roots) > 1:
            warnings.warn(
                f"Multiple root nodes found: {self.roots}. Hierarchies are typically single-rooted."
            )

    def get_all_non_root_nodes(self):
        """
        Returns a list of all node names in the hierarchy as strings,
        excluding the root node(s).
        """
        # Get all nodes and convert to a set for efficient difference calculation
        all_nodes_set = set(self.get_all_nodes())
        # Get root nodes and convert to a set
        root_nodes_set = set(self.get_roots())

        # Calculate the difference to get non-root nodes
        non_root_nodes = all_nodes_set - root_nodes_set

        # Convert to a list and sort for consistent order
        return sorted(list(non_root_nodes))

    def get_all_nodes(self):
        """Returns a list of all nodes in the hierarchy."""
        return self.all_nodes

    def get_leaves(self):
        """Returns a list of leaf nodes in the hierarchy."""
        return self.leaves

    def get_non_leaves(self):
        """Returns a list of non-leaf nodes in the hierarchy."""
        return self.non_leaves

    def get_roots(self):
        """Returns a list of root nodes in the hierarchy."""
        return self.roots

    def get_ancestors(self, node):
        """Returns a set of ancestors for a given node (including the node itself)."""
        if node not in self.graph:
            return set()
        # nx.ancestors includes all nodes reachable by traversing edges backwards
        ancestors = nx.ancestors(self.graph, node)
        ancestors.add(node)  # Include the node itself
        return ancestors

    def get_descendants(self, node):
        """Returns a set of descendants for a given node (including the node itself)."""
        if node not in self.graph:
            return set()
        # nx.descendants includes all nodes reachable by traversing edges forwards
        descendants = nx.descendants(self.graph, node)
        descendants.add(node)  # Include the node itself
        return descendants

    def __repr__(self):
        """Provides a concise string representation of the hierarchy."""
        num_nodes = len(self.graph.nodes())
        num_edges = len(self.graph.edges())
        roots_str = ", ".join(self.roots) if self.roots else "None"
        leaves_str = ", ".join(self.leaves) if self.leaves else "None"
        return (
            f"{self.__class__.__name__}(Nodes: {num_nodes}, Edges: {num_edges}, "
            f"Roots: [{roots_str}], Leaves: [{leaves_str}])"
        )

    def plot(self, layout=None, ax=None, **kwargs):
        """
        Plots the hierarchy graph using networkx and matplotlib.

        Args:
            layout: The networkx layout function to use (e.g., nx.spring_layout, nx.planar_layout).
                    Defaults to nx.planar_layout if possible, otherwise nx.spring_layout.
            ax: The matplotlib axes to draw on. If None, a new figure and axes are created.
            **kwargs: Additional keyword arguments passed to nx.draw.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "Matplotlib is required for plotting. Please install it (`pip install matplotlib`)."
            )
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            show_plot = True  # Indicate we created the figure and should show it
        else:
            show_plot = False  # User provided axes, assume they will show the plot

        if not self.graph or self.graph.number_of_nodes() == 0:
            print("Hierarchy is empty or has no nodes to plot.")
            if show_plot:
                plt.show()
            return

        # Choose layout
        if layout is None:
            try:
                # Try hierarchical layout if possible (requires graphviz/pygraphviz)
                # Using planar or spring as default fallback
                layout = nx.planar_layout(self.graph)
            except nx.NetworkXException:
                layout = nx.spring_layout(self.graph)  # Fallback layout

        # Default drawing options
        default_kwargs = {
            "with_labels": True,
            "node_color": "skyblue",
            "node_size": 2000,
            "edge_color": "gray",
            "arrows": True,
            "ax": ax,
            "font_size": 10,
            "node_shape": "o",
            "arrowstyle": "-|>",  # Nicer arrows
            "arrowsize": 20,
        }
        # Update with user-provided kwargs
        draw_kwargs = {**default_kwargs, **kwargs}

        nx.draw(self.graph, pos=layout, **draw_kwargs)
        ax.set_title(f"{self.__class__.__name__} Visualization")
        plt.tight_layout()  # Adjust layout to prevent labels overlapping

        if show_plot:
            plt.show()


class ProductHierarchy(Hierarchy):
    """Represents the product hierarchy with associated rewards."""

    def __init__(self, edges, rewards):
        super().__init__(edges)
        self.rewards = rewards
        # Validate rewards against nodes that are intended to be purchasable (those with rewards)
        defined_rewards = set(rewards.keys())
        all_graph_nodes = set(self.graph.nodes())

        # Check if all defined rewards correspond to nodes in the graph
        if not defined_rewards.issubset(all_graph_nodes):
            unknown_reward_nodes = defined_rewards - all_graph_nodes
            warnings.warn(
                f"Rewards defined for nodes not in hierarchy graph: {unknown_reward_nodes}"
            )

        # Check if all graph leaves have a defined reward
        # This warning might be noisy if some leaves are just structural but not purchasable
        # Let's keep it for now but note its limitation.
        hierarchy_leaves = set(self.get_leaves())
        leaves_without_rewards = hierarchy_leaves - defined_rewards
        if leaves_without_rewards:
            warnings.warn(
                f"Hierarchy leaves without defined rewards: {leaves_without_rewards}. Defaulting reward to 0.0 for these."
            )
        # We don't strictly require defined_rewards == hierarchy_leaves.
        # Some non-leaf nodes *could* potentially have rewards in a more complex model.

    def get_reward(self, product_node):
        """Returns the reward for a specific product node.
        Typically used for leaf nodes with defined rewards."""
        # Use .get() with a default of 0.0 if the node doesn't have a defined reward
        # Or if the node doesn't exist in the hierarchy.
        if product_node not in self.graph:
            warnings.warn(
                f"Reward requested for node '{product_node}' not in hierarchy graph. Returning 0.0."
            )
            return 0.0
        return self.rewards.get(product_node, 0.0)


class ActionHierarchy(Hierarchy):
    """Represents the action hierarchy with associated costs."""

    def __init__(self, edges, costs):
        super().__init__(edges)
        self.costs = costs
        # Validate costs against nodes that are intended to be actionable (those with costs)
        defined_costs = set(costs.keys())
        all_graph_nodes = set(self.graph.nodes())

        # Check if all defined costs correspond to nodes in the graph
        if not defined_costs.issubset(all_graph_nodes):
            unknown_cost_nodes = defined_costs - all_graph_nodes
            warnings.warn(
                f"Costs defined for nodes not in hierarchy graph: {unknown_cost_nodes}"
            )

        # Check if all graph leaves have a defined cost
        # Similar to ProductHierarchy, this might be noisy.
        hierarchy_leaves = set(self.get_leaves())
        leaves_without_costs = hierarchy_leaves - defined_costs
        if leaves_without_costs:
            warnings.warn(
                f"Hierarchy leaves without defined costs: {leaves_without_costs}. Defaulting cost to 0.0 for these."
            )
        # We don't strictly require defined_costs == hierarchy_leaves.
        # 'NoAction' with a self-loop is a prime example of a node with a cost that isn't a leaf.

    def get_cost(self, action_node):
        """Returns the cost for a specific action node.
        Typically used for leaf nodes with defined costs, but can include others."""
        # Use .get() with a default of 0.0 if the node doesn't have a defined cost
        # Or if the node doesn't exist in the hierarchy.
        if action_node not in self.graph:
            warnings.warn(
                f"Cost requested for node '{action_node}' not in hierarchy graph. Returning 0.0."
            )
            return 0.0
        return self.costs.get(action_node, 0.0)
