_palette = None

def set_palette(palette):
    global _palette
    _palette = palette

def _standard_palette(n_colors):
    return [f"hsl({((((i)*209)%360) + 25) % 360}, 67%, 61%)" for i in range(n_colors)]

def _color_palette(n_colors):
    # Standard palette given by the library, 360 different hues but gets crammed / hard to distinguish after 10
    if _palette is None:
        return _standard_palette(n_colors)
    
    # Use user-defined palettes
    if isinstance(_palette, list):
        return (_palette*(round(n_colors/len(_palette)+1)))[:n_colors]
    
    # Use seaborn-palette for string
    if isinstance(_palette, str):
        try:
            import seaborn as sns
        except ImportError:
            raise Exception("Named palettes can only be used if seaborn is installed in your python environment. Run `pip install seaborn` to install it.")
        # Converting to SVG rgb format
        return [f'rgb({color[0]*255},{color[1]*255},{color[2]*255})' for color in sns.color_palette(palette = _palette, n_colors = n_colors)]
    import warnings
    warnings.warn("The user-defined palette is not valid. Resorting to default.")
    return _standard_palette(n_colors)
    

def _shorten_number(num):
        if num < 1000:
            return num
        if num < 1_000_000:
            return f"{round(num/1_000)}K"
        if num < 1_000_000_000:
            return f"{round(num/1_000_000)}M"
        if num < 1_000_000_000_000:
            return f"{round(num/1_000_000_000)}B"
        if num < 1_000_000_000_000_000:
            return f"{round(num/1_000_000_000_000)}T"
        return f"{round(num/1_000_000_000_000_000)}Q"