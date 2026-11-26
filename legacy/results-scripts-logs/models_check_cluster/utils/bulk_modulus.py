#models_check/utils/bulk_modulus.py
"""Bulk modulus reference data for pure elements (GPa)."""

BULK_MODULUS = {
    "Ti": 110.0,
    "Al": 76.0,
    "Nb": 170.0,
    "Ni": 180.0,
    "Zr": 83.0,
    "Cu": 140.0,
    "Fe": 170.0,
    "Cr": 160.0,
    "Mo": 230.0,
    "W": 310.0,
    "V": 160.0,
    "Ta": 200.0,
    "Co": 180.0,
    "Mn": 120.0,
    "Si": 100.0,
    "Sn": 58.0,
    "In": 36.0,
    "Ag": 100.0,
    "Au": 220.0,
    "Pt": 230.0,
    "Pd": 180.0,
    "Li": 11.0,
    "Na": 6.3,
    "K": 3.1,
    "Mg": 45.0,
    "Ca": 17.0,
    "Zn": 70.0,
    "Ga": 56.0,
    "Ge": 75.0,
    "Sb": 42.0,
    "Y": 41.0,
}


def get_bulk_modulus(composition: str, counts: dict) -> float:
    """
    Get bulk modulus for pure element or alloy (rule of mixtures).
    
    Args:
        composition: Chemical formula or element symbol
        counts: Dict of {element: count} for alloys
    
    Returns:
        Bulk modulus in GPa
    """
    # Pure element
    if composition in BULK_MODULUS:
        return BULK_MODULUS[composition]
    
    # Alloy — rule of mixtures
    if not counts:
        return 100.0  # fallback
    
    total = sum(counts.values())
    if total == 0:
        return 100.0
    
    B_mix = 0.0
    for elem, count in counts.items():
        if elem in BULK_MODULUS:
            B_mix += (count / total) * BULK_MODULUS[elem]
        else:
            logger.warning(f"No bulk modulus for {elem}, using 100 GPa")
            B_mix += (count / total) * 100.0
    
    return B_mix