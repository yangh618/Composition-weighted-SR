import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PeriodicTableVisualizer:
    def __init__(self, figsize=(20, 10)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(-0.5, 17.5)
        self.ax.set_ylim(1.5, 10.5)
        self.ax.axis('off')
        
        self.sym2num = {  # Symbol to atomic number
            'H': 1, 'He': 2,
            'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
            'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
            'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44,
            'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52,
            'I': 53, 'Xe': 54,
            'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62,
            'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
            'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
            'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
            'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94,
            'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102,
            'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
            'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118,
        }

        self.num2sym = {v: k for k, v in self.sym2num.items()}

        # Complete periodic table positions (period, group)
        self.elements = {
            # Period 1
            'H': (1, 1), 'He': (1, 18),
            # Period 2
            'Li': (2, 1), 'Be': (2, 2), 'B': (2, 13), 'C': (2, 14), 'N': (2, 15), 'O': (2, 16), 'F': (2, 17), 'Ne': (2, 18),
            # Period 3
            'Na': (3, 1), 'Mg': (3, 2), 'Al': (3, 13), 'Si': (3, 14), 'P': (3, 15), 'S': (3, 16), 'Cl': (3, 17), 'Ar': (3, 18),
            # Period 4
            'K': (4, 1), 'Ca': (4, 2), 'Sc': (4, 3), 'Ti': (4, 4), 'V': (4, 5), 'Cr': (4, 6), 'Mn': (4, 7), 'Fe': (4, 8),
            'Co': (4, 9), 'Ni': (4, 10), 'Cu': (4, 11), 'Zn': (4, 12), 'Ga': (4, 13), 'Ge': (4, 14), 'As': (4, 15), 'Se': (4, 16),
            'Br': (4, 17), 'Kr': (4, 18),
            # Period 5
            'Rb': (5, 1), 'Sr': (5, 2), 'Y': (5, 3), 'Zr': (5, 4), 'Nb': (5, 5), 'Mo': (5, 6), 'Tc': (5, 7), 'Ru': (5, 8),
            'Rh': (5, 9), 'Pd': (5, 10), 'Ag': (5, 11), 'Cd': (5, 12), 'In': (5, 13), 'Sn': (5, 14), 'Sb': (5, 15), 'Te': (5, 16),
            'I': (5, 17), 'Xe': (5, 18),
            # Period 6
            'Cs': (6, 1), 'Ba': (6, 2), 'Hf': (6, 4), 'Ta': (6, 5), 'W': (6, 6), 'Re': (6, 7), 'Os': (6, 8), 'Ir': (6, 9), 'Pt': (6, 10),
            'Au': (6, 11), 'Hg': (6, 12), 'Tl': (6, 13), 'Pb': (6, 14), 'Bi': (6, 15), 'Po': (6, 16), 'At': (6, 17), 'Rn': (6, 18),
            # Period 7
            'Fr': (7, 1), 'Ra': (7, 2), 'Rf': (7, 4), 'Db': (7, 5), 'Sg': (7, 6), 'Bh': (7, 7), 'Hs': (7, 8), 'Mt': (7, 9), 'Ds': (7, 10),
            'Rg': (7, 11), 'Cn': (7, 12), 'Nh': (7, 13), 'Fl': (7, 14), 'Mc': (7, 15), 'Lv': (7, 16), 'Ts': (7, 17), 'Og': (7, 18),
            # Lanthanides (below period 6)
            'La': (8, 3), 'Ce': (8, 4), 'Pr': (8, 5), 'Nd': (8, 6), 'Pm': (8, 7), 'Sm': (8, 8),
            'Eu': (8, 9), 'Gd': (8, 10), 'Tb': (8, 11), 'Dy': (8, 12), 'Ho': (8, 13), 'Er': (8, 14), 'Tm': (8, 15), 'Yb': (8, 16), 'Lu': (8, 17),
            # Actinides (below period 7)
            'Ac': (9, 3), 'Th': (9, 4), 'Pa': (9, 5), 'U': (9, 6), 'Np': (9, 7), 'Pu': (9, 8),
            'Am': (9, 9), 'Cm': (9, 10), 'Bk': (9, 11), 'Cf': (9, 12), 'Es': (9, 13), 'Fm': (9, 14), 'Md': (9, 15), 'No': (9, 16), 'Lr': (9, 17),
        }
    
    def add_formula_box(self, formula, x, y):
        """Add a formula at position (x, y) in LaTeX-like style"""
        # Convert formula to LaTeX format for better rendering
        latex_formula = formula.replace('**', '^').replace('*', '\\cdot ')
        
        self.ax.text(x, y, f"${latex_formula}$", ha='center', va='center', 
                    fontsize=16, fontweight='bold')

    def add_element_box(self, is_present, symbol, x, y, weights=None, normalized_weights=None, cmap='viridis', vmin=0, vmax=1):
        """Add element box with separate colored pieces for each weight"""

        if weights is None:
            # Default single box if no weights provided
            color = plt.colormaps.get_cmap(cmap)(0.3)
            rect = patches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, 
                                     linewidth=1.5, edgecolor='black', 
                                     facecolor=color, alpha=0.8)
            self.ax.add_patch(rect)
            self.ax.text(x, y, symbol, ha='center', va='center', 
                        fontsize=12, fontweight='bold')
        else:
            if not is_present:
                # Create separate colored pieces for each weight (horizontal pieces)
                num_weights = len(weights)
                # Make the symbol piece slightly taller than the weight pieces
                symbol_height = 0.3
                remaining_height = 0.8 - symbol_height
                piece_height = remaining_height / num_weights
                
                # Add atomic symbol as the first piece (top of the box)
                symbol_y = y - 0.4 + remaining_height
                symbol_rect = patches.Rectangle((x - 0.4, symbol_y), 0.8, symbol_height, 
                                            linewidth=1, edgecolor='black', 
                                            facecolor='lightgrey', alpha=0.5)
                self.ax.add_patch(symbol_rect)
                self.ax.text(x, symbol_y + symbol_height/2, symbol, ha='center', va='center', 
                            fontsize=12, fontweight='bold')
                
                for i, weight_val in enumerate(weights):
                    # Use normalized weights if provided, otherwise normalize using fixed scale
                    if normalized_weights is not None and i < len(normalized_weights):
                        norm_val = normalized_weights[i]
                    else:
                        # Fallback to original normalization
                        norm_val = min(max(abs(weight_val) / 10.0, 0), 1)
                    
                    # Use different colormaps for positive and negative weights
                    weight_cmap = 'Reds' if weight_val >= 0 else 'Blues'
                    color = plt.colormaps.get_cmap(weight_cmap)(norm_val)
                    
                    # Position each piece (horizontal layout, below the symbol)
                    piece_y = y - 0.4 + (i * piece_height)
                    
                    # Create rectangle for this weight piece
                    rect = patches.Rectangle((x - 0.4, piece_y), 0.8, piece_height, 
                                            linewidth=1, edgecolor='black', 
                                            facecolor='lightgrey', alpha=0.5)
                    self.ax.add_patch(rect)


            else:
                # Create separate colored pieces for each weight (horizontal pieces)
                num_weights = len(weights)
                # Make the symbol piece slightly taller than the weight pieces
                symbol_height = 0.3
                remaining_height = 0.8 - symbol_height
                piece_height = remaining_height / num_weights
                
                # Add atomic symbol as the first piece (top of the box)
                symbol_y = y - 0.4 + remaining_height
                symbol_rect = patches.Rectangle((x - 0.4, symbol_y), 0.8, symbol_height, 
                                            linewidth=1, edgecolor='black', 
                                            facecolor='white', alpha=0.9)
                self.ax.add_patch(symbol_rect)
                self.ax.text(x, symbol_y + symbol_height/2, symbol, ha='center', va='center', 
                            fontsize=12, fontweight='bold')
                
                for i, weight_val in enumerate(weights):
                    # Use normalized weights if provided, otherwise normalize using fixed scale
                    if normalized_weights is not None and i < len(normalized_weights):
                        norm_val = normalized_weights[i]
                    else:
                        # Fallback to original normalization
                        norm_val = min(max(abs(weight_val) / 10.0, 0), 1)
                    
                    # Use different colormaps for positive and negative weights
                    weight_cmap = 'Reds' if weight_val >= 0 else 'Blues'
                    color = plt.colormaps.get_cmap(weight_cmap)(norm_val)
                    
                    # Position each piece (horizontal layout, below the symbol)
                    piece_y = y - 0.4 + (i * piece_height)
                    
                    # Create rectangle for this weight piece
                    rect = patches.Rectangle((x - 0.4, piece_y), 0.8, piece_height, 
                                            linewidth=1, edgecolor='black', 
                                            facecolor=color, alpha=0.9)
                    self.ax.add_patch(rect)
                    
                    # Add weight value text in the piece
                    self.ax.text(x, piece_y + piece_height/2, f"{weight_val:.8f}", 
                            ha='center', va='center', fontsize=8, fontweight='bold')
    

    def plot_table(self, expression, weights, is_presents=None):
        """
        expression: string representation of the mathematical expression
        weights: 2D array of tabulated weights for each element (elements x variables)
        """
        self.add_formula_box(expression, 6.5, 9.5)

        # Calculate max absolute weight for each feature across all elements
        if weights is not None and weights.shape[0] > 0:
            # Get max absolute value for each feature (column)
            feature_max = np.max(np.abs(weights[is_presents]), axis=0)
            # Ensure minimum value of 1 to avoid division by zero
            feature_max = np.maximum(feature_max, 1.0)
        else:
            feature_max = np.ones(1)

        for symbol, (period, group) in self.elements.items():
            x = group - 1
            y = 11 - period
            
            # Get atomic number for this element
            atomic_num = self.sym2num.get(symbol, 0)
            is_present = is_presents[atomic_num - 1] if is_presents is not None and atomic_num > 0 else True

            # Get weights for this element (if available)
            if weights is not None and atomic_num > 0 and atomic_num <= weights.shape[0]:
                element_weights = weights[atomic_num - 1]  # Convert to 0-based indexing
                
                # Normalize weights using feature-specific max values
                normalized_weights = []
                for i, weight_val in enumerate(element_weights):
                    # Use the max value for this specific feature
                    max_val = feature_max[i] if i < len(feature_max) else feature_max[-1]
                    norm_val = abs(weight_val) / max_val
                    normalized_weights.append(norm_val)
                
                self.add_element_box(is_present, symbol, x, y, weights=element_weights, 
                                   normalized_weights=normalized_weights, cmap='YlOrRd')
            else:
                self.add_element_box(is_present, symbol, x, y, 0.3, cmap='Greys')
        
        plt.tight_layout()
    
    def show(self):
        plt.show()
    
    def save(self, filename):
        plt.savefig(filename, dpi=300, bbox_inches='tight')

# Example usage
if __name__ == '__main__':
    viz = PeriodicTableVisualizer()
    
    # Create sample weights array (118 elements for all elements)
    weights = np.random.uniform(-5, 5, (118, 2))
    is_presents = np.random.choice([True, False], size=(118,))

    expression = "2**x0+2*x1"
    viz.plot_table(expression, weights, is_presents)
    viz.save('./periodic_table_rgb.eps')
    viz.show()
