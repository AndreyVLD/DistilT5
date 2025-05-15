import torch
import numpy as np
from typing import Any, Optional


def entropy_decode(encoded_data: dict[str, Any]) -> bytes:
    """
    Decompress data that was compressed with LZ4.

    Args:
        encoded_data: Dictionary with compressed data and metadata

    Returns:
        Original binary data
    """
    encoding = encoded_data.get('encoding', 'none')

    try:
        if encoding == 'lz4':
            import lz4.frame

            # Get compressed data
            compressed = bytes.fromhex(encoded_data.get('data'))

            # Decompress with LZ4
            decompressed = lz4.frame.decompress(compressed)
            return decompressed
        elif encoding == 'none':
            # No compression was applied
            return bytes.fromhex(encoded_data.get('data'))
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
    except ImportError:
        print("Error: LZ4 not installed. Please install with 'pip install lz4'.")
        raise
    except Exception as e:
        print(f"Error during LZ4 decompression: {e}")
        import traceback
        traceback.print_exc()
        raise


def decompress_logits(compressed_logits: Optional[dict[str, Any]]) -> Optional[torch.Tensor]:
    """
    Decompress logits that were compressed with bit-depth reduction and LZ4.

    Args:
        compressed_logits: The compressed representation

    Returns:
        PyTorch tensor with decompressed logits
    """
    if compressed_logits is None:
        return None

    try:
        # Extract basic information
        bits = compressed_logits.get('bits')
        shape = compressed_logits.get('shape')

        # Convert shape from list to tuple if necessary
        if isinstance(shape, list):
            shape = tuple(shape)

        # Step 1: Decompress LZ4 data
        encoded_data = compressed_logits.get('data_encoded')
        data_bytes = entropy_decode(encoded_data)

        # Step 2: Process according to bit depth
        if bits == 16:
            # 16-bit decompression (FP16)
            logits_np = np.frombuffer(data_bytes, dtype=np.float16).reshape(shape)
            # Convert to float32 for compatibility with PyTorch operations
            logits_np = logits_np.astype(np.float32)
            return torch.tensor(logits_np)

        elif bits == 8:
            # 8-bit dequantization
            logits_int8 = np.frombuffer(data_bytes, dtype=np.uint8).reshape(shape)

            # Dequantize
            scale = compressed_logits.get('scale', 1.0)
            zero_point = compressed_logits.get('zero_point', 0.0)
            logits_np = (logits_int8.astype(np.float32) - zero_point) * scale
            return torch.tensor(logits_np)

        elif bits == 4:
            # Check if values were packed
            is_packed = compressed_logits.get('packed', False)

            if is_packed:
                # Unpack 4-bit values (2 values per byte)
                packed = np.frombuffer(data_bytes, dtype=np.uint8)

                # Calculate total values in original tensor
                total_values = np.prod(shape)
                unpacked = np.zeros(total_values, dtype=np.uint8)

                # Handle even indices (lower 4 bits of each byte)
                even_indices = np.arange(0, total_values, 2)
                even_indices = even_indices[even_indices < total_values]
                unpacked[even_indices] = packed[:len(even_indices)] & 0x0F

                # Handle odd indices (upper 4 bits of each byte)
                odd_indices = np.arange(1, total_values, 2)
                odd_indices = odd_indices[odd_indices < total_values]
                unpacked[odd_indices] = (packed[:len(odd_indices)] >> 4) & 0x0F

                # Reshape to original shape
                logits_int4 = unpacked.reshape(shape)
            else:
                # Direct interpretation (rarely used for 4-bit)
                logits_int4 = np.frombuffer(data_bytes, dtype=np.uint8).reshape(shape)

            # Dequantize
            scale = compressed_logits.get('scale', 1.0)
            zero_point = compressed_logits.get('zero_point', 0.0)
            logits_np = (logits_int4.astype(np.float32) - zero_point) * scale
            return torch.tensor(logits_np)

        elif bits == 32:
            # 32-bit (float32) decompression
            logits_np = np.frombuffer(data_bytes, dtype=np.float32).reshape(shape)
            return torch.tensor(logits_np)

        else:
            raise ValueError(f"Unsupported bit depth: {bits}")

    except Exception as e:
        print(f"Error during decompression: {e}")
        import traceback
        traceback.print_exc()
        return None
