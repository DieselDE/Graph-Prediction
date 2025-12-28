import numpy as np
import pickle
import prediction as pr

def save_graph(graph: dict, filename: str):
    """Save graph to disk."""
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Graph saved to {filename}")


def load_graph(filename: str) -> dict:
    """Load graph from disk."""
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
    print(f"Graph loaded from {filename}")
    return graph

def create_windows(data: list[float], window_size: int = 20, 
                   use_pct_change: bool = True) -> np.ndarray:
    """
    Create sliding windows from price data.
    
    Args:
        data: List of prices [100, 102, 101, ...]
        window_size: How many time points per window
        use_pct_change: If True, convert to percentage changes
    
    Returns:
        Array of shape (num_windows, window_size) or (num_windows, window_size-1)
        Each row is one window
    """
    if len(data) < window_size:
        return np.array([])
    
    windows = []
    
    for i in range(len(data) - window_size + 1):
        window = data[i : i + window_size]
        
        if use_pct_change:
            # Convert to percentage changes: (price[t+1] - price[t]) / price[t] * 100
            window = np.array(window)
            pct_changes = (window[1:] - window[:-1]) / window[:-1] * 100
            windows.append(pct_changes)
        else:
            windows.append(window)
    
    return np.array(windows)

def calculate_distances(windows: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise distances between all windows.
    
    Args:
        windows: Array of shape (num_windows, window_size)
    
    Returns:
        Distance matrix of shape (num_windows, num_windows)
        distance[i, j] = how different window i and window j are
    """
    num_windows = len(windows)
    distances = np.zeros((num_windows, num_windows))
    
    # Calculate distance between every pair of windows
    for i in range(num_windows):
        for j in range(num_windows):
            # Euclidean distance: sqrt(sum of squared differences)
            diff = windows[i] - windows[j]
            distances[i, j] = np.sqrt(np.sum(diff ** 2))
    
    return distances

def create_edges(distances: np.ndarray, k: int = 5, 
                 max_distance: float = None) -> np.ndarray:
    """
    Create graph edges by connecting each node to its k nearest neighbors.
    
    Args:
        distances: Distance matrix of shape (num_windows, num_windows)
        k: Number of nearest neighbors to connect to
        max_distance: Only connect if distance <= this threshold (optional)
    
    Returns:
        Edge list of shape (2, num_edges)
    """
    num_nodes = len(distances)
    edges_list = []
    
    for i in range(num_nodes):
        node_distances = distances[i]
        nearest_indices = np.argsort(node_distances)
        
        # Skip first one (itself) and take next k
        nearest_neighbors = nearest_indices[1 : k + 1]
        
        for neighbor in nearest_neighbors:
            # Only add edge if within distance threshold
            if max_distance is None or distances[i, neighbor] <= max_distance:
                edges_list.append([i, neighbor])
    
    if len(edges_list) == 0:
        # No edges found! Return empty array with correct shape
        return np.array([[], []], dtype=int)
    
    edges = np.array(edges_list).T
    return edges

def create_graph_from_prices(data: list[float], 
                             window_size: int = 20, 
                             k: int = 5) -> dict:
    """
    Convert time series price data into a graph structure.
    
    Args:
        data: List of prices
        window_size: Size of each window (node)
        k: Number of nearest neighbors to connect
    
    Returns:
        Dictionary containing:
            - 'windows': Node features (percentage changes)
            - 'edges': Edge connections
            - 'num_nodes': Number of nodes
            - 'num_edges': Number of edges
    """
    # Step 1: Create windows (nodes)
    windows = create_windows(data, window_size=window_size, use_pct_change=True)
    
    if len(windows) == 0:
        print("Not enough data to create windows!")
        return None
    
    # Step 2: Calculate distances between all windows
    distances = calculate_distances(windows)
    
    # Step 3: Create edges based on similarity
    edges = create_edges(distances, k=k)
    
    # Package everything together
    graph = {
        'windows': windows,
        'edges': edges,
        'num_nodes': len(windows),
        'num_edges': edges.shape[1] if edges.shape[1] > 0 else 0,
        'distances': distances  # Keep this for debugging
    }
    
    return graph

def create_labels(data: list[float], window_size: int = 20, 
                  prediction_steps: int = 5) -> np.ndarray:
    """
    Create prediction labels: what happens after each window.
    
    Args:
        data: List of prices
        window_size: Size of each window
        prediction_steps: How many steps ahead to predict
    
    Returns:
        Array of shape (num_windows, prediction_steps)
        Each row contains the percentage changes after that window
    """
    # We need enough data for: window + prediction_steps
    max_start = len(data) - window_size - prediction_steps
    
    if max_start < 1:
        print("Not enough data for labels!")
        return np.array([])
    
    labels = []
    
    for i in range(max_start + 1):
        # Get the prices after this window
        future_prices = data[i + window_size : i + window_size + prediction_steps + 1]
        
        # Safety check: make sure we have exactly prediction_steps + 1 prices
        if len(future_prices) != prediction_steps + 1:
            print(f"Warning: Skipping window {i}, not enough future data")
            continue
        
        # Convert to percentage changes
        future_prices = np.array(future_prices)
        pct_changes = (future_prices[1:] - future_prices[:-1]) / future_prices[:-1] * 100
        
        # Double check the length
        if len(pct_changes) != prediction_steps:
            print(f"Warning: Window {i} has wrong prediction length")
            continue
            
        labels.append(pct_changes)
    
    # Convert to numpy array
    labels_array = np.array(labels)
    
    print(f"Created {len(labels_array)} labels from {len(data)} data points")
    
    return labels_array

def create_graph_with_labels(data: list[float], 
                              window_size: int = 20,
                              k: int = 5,
                              prediction_steps: int = 5) -> dict:
    """
    Create graph AND labels together.
    """
    # Create labels first (this determines how many windows we can use)
    labels = create_labels(data, window_size=window_size, 
                          prediction_steps=prediction_steps)
    
    if len(labels) == 0:
        return None
    
    num_valid_windows = len(labels)
    
    # Create windows
    windows = create_windows(data, window_size=window_size, use_pct_change=True)
    
    # Trim windows to match labels
    windows = windows[:num_valid_windows]
    
    # Calculate distances and edges
    distances = calculate_distances(windows)
    edges = create_edges(distances, k=k)
    
    graph = {
        'windows': windows,
        'labels': labels,
        'edges': edges,
        'num_nodes': len(windows),
        'num_edges': edges.shape[1] if edges.shape[1] > 0 else 0,
    }
    
    return graph

graph = load_graph("crypto_graph.pkl")