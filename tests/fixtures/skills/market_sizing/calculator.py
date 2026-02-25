"""Market sizing calculator template."""


def calculate_tam(market_size: float, growth_rate: float, years: int) -> float:
    """Calculate Total Addressable Market projection."""
    return market_size * ((1 + growth_rate) ** years)


def calculate_sam(tam: float, serviceable_pct: float) -> float:
    """Calculate Serviceable Addressable Market."""
    return tam * serviceable_pct


def calculate_som(sam: float, capture_pct: float) -> float:
    """Calculate Serviceable Obtainable Market."""
    return sam * capture_pct
