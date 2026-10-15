"""CPU-hum plugin stub.

Reuses the engine AudioSource and Preprocessor. Swaps the segmenter for fixed
windows (hum is continuous), the features for band energy, and the decoder
for an instruction-frequency prior.
"""

from plugins.cpuhum.decoder import ISAPriorDecoder, INSTRUCTIONS
from plugins.cpuhum.features import BandEnergyExtractor

__all__ = ["ISAPriorDecoder", "BandEnergyExtractor", "INSTRUCTIONS"]
