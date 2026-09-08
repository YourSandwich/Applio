import os
import torch

if torch.cuda.is_available() and torch.cuda.get_device_name().endswith("[ZLUDA]"):

    class STFT:
        def __init__(self):
            self.device = "cuda"
            self.fourier_bases = {}  # Cache for Fourier bases

        def _get_fourier_basis(self, n_fft):
            # Check if the basis for this n_fft is already cached
            if n_fft in self.fourier_bases:
                return self.fourier_bases[n_fft]
            fourier_basis = torch.fft.fft(torch.eye(n_fft, device="cpu")).to(
                self.device
            )
            # stack separated real and imaginary components and convert to torch tensor
            cutoff = n_fft // 2 + 1
            fourier_basis = torch.cat(
                [fourier_basis.real[:cutoff], fourier_basis.imag[:cutoff]], dim=0
            )
            # cache the tensor and return
            self.fourier_bases[n_fft] = fourier_basis
            return fourier_basis

        def transform(self, input, n_fft, hop_length, window):
            # fetch cached Fourier basis
            fourier_basis = self._get_fourier_basis(n_fft)
            # apply hann window to Fourier basis
            fourier_basis = fourier_basis * window
            # pad input to center with reflect
            pad_amount = n_fft // 2
            input = torch.nn.functional.pad(
                input, (pad_amount, pad_amount), mode="reflect"
            )
            # separate input into n_fft-sized frames
            input_frames = input.unfold(1, n_fft, hop_length).permute(0, 2, 1)
            # apply fft to each frame
            fourier_transform = torch.matmul(fourier_basis, input_frames)
            cutoff = n_fft // 2 + 1
            return torch.complex(
                fourier_transform[:, :cutoff, :], fourier_transform[:, cutoff:, :]
            )

    stft = STFT()
    _torch_stft = torch.stft

    def z_stft(input: torch.Tensor, window: torch.Tensor, *args, **kwargs):
        # only optimizing a specific call from rvc.train.mel_processing.MultiScaleMelSpectrogramLoss
        if (
            kwargs.get("win_length") == None
            and kwargs.get("center") == None
            and kwargs.get("return_complex") == True
        ):
            # use GPU accelerated calculation
            return stft.transform(
                input, kwargs.get("n_fft"), kwargs.get("hop_length"), window
            )
        else:
            # simply do the operation on CPU
            return _torch_stft(
                input=input.cpu(), window=window.cpu(), *args, **kwargs
            ).to(input.device)

    def z_jit(f, *_, **__):
        f.graph = torch._C.Graph()
        return f

    # hijacks
    torch.stft = z_stft
    torch.jit.script = z_jit
    # disabling unsupported cudnn
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)


# MIOpen's default find path has no dilated conv1d solver and falls back to GemmFwdRest. On
# gfx1100 that runs dilation 3 at 0.30 TFLOP/s against 18.7 undilated - the same arithmetic, 63x
# slower - and the HiFi-GAN / NSF ResBlocks are built on dilations (1, 3, 5), so it dominates the
# training step.
#
# Find mode 2 (Fast) misses the find-db and takes the Immediate-mode fallback, which does pick a
# working kernel: 85.4 ms -> 1.35 ms for N8 C512 L2048 k3 d3 bf16, measured from a cold cache.
# Undilated and pointwise convolutions are unaffected (1.29-1.37 ms either way).
#
# Do NOT also set MIOPEN_FIND_ENFORCE=2. It forces a real find, which bypasses the Immediate
# fallback and lands back on GemmFwdRest at 85.4 ms - measured, cold and warm.
#
# MIOpen reads this at the first convolution rather than at import, so setting it after `import
# torch` is early enough. setdefault so an explicit value from the environment still wins.
if torch.cuda.is_available() and "AMD" in torch.cuda.get_device_name():
    os.environ.setdefault("MIOPEN_FIND_MODE", "2")
