import math
from ncm_mac._vendor.utils.misc import round_levels

tokenizers = {
    "bpe": "ncm_mac._vendor.tokenization.bpe_tokenizer.BPETokenizer",
    "hf": "ncm_mac._vendor.tokenization.hf_tokenizer.HFTokenizer",
    "sudoku": "ncm_mac._vendor.tokenization.sudoku_tokenizer.SudokuTokenizer",
    "vqvae": "ncm_mac._vendor.tokenization.vqvae_tokenizer.VQVAETokenizer",
}

vector_quantizers = {
    "multiscale_residual_vector_quantizer": "ncm_mac._vendor.models.quant.MultiscaleResidualVectorQuantizer",
}

level_schedulers = {
    "linear": lambda start, end, n: round_levels([start + (end - start) * (i-1)/(n-1) for i in range(1, n+1)], start, end),
    "cubic": lambda start, end, n: round_levels([start + (end - start) * (i-1)**3/(n-1)**3 for i in range(1, n+1)], start, end),
    "exp": lambda start, end, n: round_levels([start + (end - start) * (math.exp(i-1))/(math.exp(n-1)) for i in range(1, n+1)], start, end),
    "sqrt": lambda start, end, n: round_levels([start + (end - start) * math.sqrt(i-1)/math.sqrt(n-1) for i in range(1, n+1)], start, end),
    "sine": lambda start, end, n: round_levels([start + (end - start) * math.sin((math.pi/2) * ((i-1)/(n-1))) for i in range(1, n+1)], start, end),
    "log": lambda start, end, n: round_levels([start + (end - start) * math.log(i)/math.log(n) for i in range(1, n+1)], start, end),
    "quadratic": lambda start, end, n: round_levels([start + (end - start) * (i-1)**2/(n-2)**2 for i in range(1, n+1)], start, end),
}
