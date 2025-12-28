import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import graph_creation as gc
import prediction as pr

class SimpleGNNLayer(nn.Module):
    """
    A simple Graph Neural Network layer.
    Does message passing: each node collects info from its neighbors.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: Size of input features (19 in our case - the percentage changes)
            output_dim: Size of output features (we can choose this)
        """
        super(SimpleGNNLayer, self).__init__()
        
        # Neural network to transform features
        # This learns how to process the information
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: do message passing.
        
        Args:
            x: Node features, shape (num_nodes, input_dim)
            edges: Edge list, shape (2, num_edges)
        
        Returns:
            Updated node features, shape (num_nodes, output_dim)
        """
        num_nodes = x.shape[0]
        
        # Step 1: Aggregate - collect messages from neighbors
        # Create a tensor to store aggregated messages
        aggregated = torch.zeros(num_nodes, x.shape[1])
        
        # For each edge, add the neighbor's features
        for i in range(edges.shape[1]):
            source = edges[0, i]  # Where the edge comes from
            target = edges[1, i]  # Where the edge goes to
            
            # Target node receives message from source node
            aggregated[target] += x[source]
        
        # Step 2: Average the aggregated messages
        # Count how many neighbors each node has
        neighbor_count = torch.zeros(num_nodes)
        for i in range(edges.shape[1]):
            target = edges[1, i]
            neighbor_count[target] += 1
        
        # Avoid division by zero (nodes with no neighbors)
        neighbor_count = torch.clamp(neighbor_count, min=1)
        
        # Average: divide by number of neighbors
        aggregated = aggregated / neighbor_count.unsqueeze(1)
        
        # Step 3: Combine with original features and transform
        # Add original features (skip connection)
        combined = x + aggregated
        
        # Transform through neural network
        output = self.linear(combined)
        
        # Apply activation function (ReLU = set negatives to 0)
        output = torch.relu(output)
        
        return output

class GNNModel(nn.Module):
    """
    Complete Graph Neural Network for stock prediction.
    
    Architecture:
    - Multiple GNN layers (message passing)
    - Final prediction layer (outputs future price changes)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        """
        Args:
            input_dim: Size of input features (19 - our percentage changes)
            hidden_dim: Size of hidden layers (we choose this, e.g., 64)
            output_dim: Size of output (5 - predicting 5 steps ahead)
            num_layers: How many GNN layers to stack
        """
        super(GNNModel, self).__init__()
        
        self.num_layers = num_layers
        
        # Create list of GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.gnn_layers.append(SimpleGNNLayer(input_dim, hidden_dim))
        
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.gnn_layers.append(SimpleGNNLayer(hidden_dim, hidden_dim))
        
        # Final prediction layer: hidden_dim -> output_dim
        self.predictor = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire network.
        
        Args:
            x: Node features, shape (num_nodes, input_dim)
            edges: Edge list, shape (2, num_edges)
        
        Returns:
            Predictions, shape (num_nodes, output_dim)
        """
        # Pass through all GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edges)
            # After each layer, nodes have updated features based on neighbors
        
        # Final prediction layer (no more message passing)
        predictions = self.predictor(x)
        
        return predictions

def train_gnn_improved(graph: dict, epochs: int = 200, learning_rate: float = 0.001, 
                       hidden_dim: int = 128):
    """
    Improved training with:
    1. More epochs (200 instead of 100)
    2. Larger hidden_dim (128 instead of 64) - more capacity to learn
    3. Train/validation split - see how well it generalizes
    """
    
    # Convert to tensors
    x = torch.FloatTensor(graph['windows'])
    y = torch.FloatTensor(graph['labels'])
    edges = torch.LongTensor(graph['edges'])
    
    # Split data: 80% train, 20% validation
    num_nodes = len(x)
    num_train = int(0.8 * num_nodes)
    
    # Use first 80% for training, last 20% for validation
    train_idx = list(range(num_train))
    val_idx = list(range(num_train, num_nodes))
    
    print(f"Training nodes: {len(train_idx)}, Validation nodes: {len(val_idx)}")
    
    # Create model with larger capacity
    input_dim = x.shape[1]
    output_dim = y.shape[1]
    
    model = GNNModel(input_dim=input_dim, 
                     hidden_dim=hidden_dim, 
                     output_dim=output_dim, 
                     num_layers=3)  # Added one more layer!
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        predictions = model(x, edges)
        
        # Calculate loss only on training nodes
        train_loss = criterion(predictions[train_idx], y[train_idx])
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Validation (every 10 epochs)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_predictions = model(x, edges)
                val_loss = criterion(val_predictions[val_idx], y[val_idx])
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss.item():.6f} | Val Loss: {val_loss.item():.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
    
    print("-" * 60)
    print(f"Training complete! Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    return model

test_data = pr.read_data("test_data.tsv")
graph = gc.create_graph_with_labels(test_data)

# Train improved model
print("=== Training Improved GNN ===\n")
improved_model = train_gnn_improved(graph, epochs=200, learning_rate=0.001, hidden_dim=128)

# Test it
print("\n=== Testing Improved Predictions ===")
with torch.no_grad():
    x = torch.FloatTensor(graph['windows'])
    edges = torch.LongTensor(graph['edges'])
    predictions = improved_model(x, edges)

# Show multiple samples
for i in range(5):
    print(f"\nNode {i}:")
    print(f"  Predicted: {predictions[i].numpy()}")
    print(f"  Actual:    {graph['labels'][i]}")
    
    # Calculate mean absolute error
    error = np.abs(predictions[i].numpy() - graph['labels'][i]).mean()
    print(f"  Avg Error: {error:.4f}")