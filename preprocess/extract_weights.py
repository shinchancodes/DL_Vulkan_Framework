import argparse, json, struct, os
import torch
import torch.nn as nn

import yaml
from types import SimpleNamespace

class SimpleCNN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        return self.conv1(x)


def make_sobel_kernel(in_ch, out_ch):
    """
    Builds a [outC, inC, 3, 3] Sobel-Y kernel tensor.
    Each output channel filters its corresponding input channel independently.
    """
    sobel_y = torch.tensor([
        [-1., 0., +1.],
        [-2., 0., +2.],
        [-1., 0., +1.]
    ])  # shape [3, 3]

    # [outC, inC, kH, kW]
    w = torch.zeros(out_ch, in_ch, 3, 3)
    for oc in range(out_ch):
        ic = oc % in_ch          # map each output channel to its input channel
        w[oc, ic] = sobel_y
    return w

def main():
    # Load the file
    with open("config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    # Convert to a namespace so you don't have to change your existing code
    args = SimpleNamespace(**config_dict)
    print(args)

    # ---- Create output directories ----
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.pth) or ".", exist_ok=True)

    # ---- Build model and inject Sobel weights ----
    model = SimpleCNN(args.in_ch, args.out_ch, args.kernel_size)

    with torch.no_grad():
        model.conv1.weight.copy_(make_sobel_kernel(args.in_ch, args.out_ch))

    model.eval()
    print("Sobel weights assigned.")
    print("conv1.weight:\n", model.conv1.weight)

    # ---- Save .pth ----
    torch.save(model.state_dict(), args.pth)
    print(f"Saved model → {args.pth}")

    # ---- Extract for C++ (same as before) ----
    w = model.conv1.weight.detach().float().cpu()
    outC, inC, kH, kW = w.shape

    # Reorder [outC, inC, kH, kW] → [kH, kW, inC, outC]
    w = w.permute(2, 3, 1, 0).contiguous()

    bin_path = os.path.join(args.out, "kernel.bin")
    with open(bin_path, "wb") as f:
        f.write(struct.pack(f"{w.numel()}f", *w.flatten().tolist()))
    print(f"Saved kernel → {bin_path}")

    padding = model.conv1.padding
    stride  = model.conv1.stride
    meta = {
        "kH":      kH,   "kW":      kW,
        "inC":     inC,  "outC":    outC,
        "padH":    padding[0], "padW": padding[1],
        "strideH": stride[0],  "strideW": stride[1]
    }
    meta_path = os.path.join(args.out, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta   → {meta_path}")

if __name__ == "__main__":
    main()