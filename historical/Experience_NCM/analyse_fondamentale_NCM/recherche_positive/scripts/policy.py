"""Certified singleton readout. References and uniqueness solvers are never used."""
from base import *
def decode_native(model,z):
    assert len(z)==1
    return model.vqvae.decoder(z).argmax(-1).numpy()
def decode_certified(model,z,codes,prefix,mapping):
    assert len(z)==len(codes)==len(prefix)==1
    candidate=lookup_board(codes.numpy(),mapping)
    if certificate(candidate,prefix)[0]:return candidate
    return decode_native(model,z)
