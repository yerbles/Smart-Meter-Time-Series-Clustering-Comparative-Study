from matplotlib.markers import MarkerStyle
import matplotlib.colors as mcolors

def get_custom_palette():
    # https://sashamaps.net/docs/resources/20-colors/
    hex_colors = [
        "#e6194B",  # Red
        "#3cb44b",  # Green
        "#ffe119",  # Yellow
        "#4363d8",  # Blue
        "#f58231",  # Orange
        "#911eb4",  # Purple
        "#42d4f4",  # Cyan
        "#9A6324",   # Brown
        "#f032e6",  # Magenta
        "#808000",  # Olive
        "#469990",  # Teal
        "#800000",  # Maroon
        "#000075",  # Navy

    ]
    
    # Convert hex to RGB (0-1 scale)
    return [mcolors.to_rgb(color) for color in hex_colors]

def generate_color_marker_palette(n=13):
    # Get the custom color palette
    colors = get_custom_palette()
    
    palette_colors = []
    markers = []
    
    for i in range(n):
        color_index = i % len(colors)
        palette_colors.append(colors[color_index])
        
        if i >= n - 5:  # Last 5 markers are unfilled
            markers.append(MarkerStyle('o', fillstyle='none'))
        else:
            markers.append('o')  # The rest are filled circles
    
    return palette_colors, markers