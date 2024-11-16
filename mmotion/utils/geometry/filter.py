import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numbers


def gaussian_filter1d(input_tensor:torch.Tensor, sigma:float, axis=-1, order=0, output=None,
                              mode="reflect", cval=0.0, truncate=4.0, *, radius=None):
    """
    1-D Gaussian filter implementation in PyTorch.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor.
    sigma : scalar
        Standard deviation for Gaussian kernel.
    axis : int, optional
        The axis of the input tensor along which to apply the filter. Default is -1.
    order : int, optional
        The derivative order of the Gaussian filter. Default is 0 (Gaussian kernel).
    output : None or torch.Tensor, optional
        If provided, the output is written into this tensor.
    mode : str, optional
        Padding mode: 'reflect', 'constant', etc. Default is 'reflect'.
    cval : scalar, optional
        Value to fill pad if mode is 'constant'. Default is 0.0.
    truncate : float, optional
        Truncate the filter at this many standard deviations. Default is 4.0.
    radius : None or int, optional
        Radius of the Gaussian kernel. If specified, the size of the kernel will be
        ``2*radius + 1``, and `truncate` is ignored. Default is None.

    Returns
    -------
    output_tensor : torch.Tensor
        The filtered tensor.
    """

    # Ensure sigma is a float
    sigma = float(sigma)

    # Calculate the kernel radius
    lw = int(truncate * sigma + 0.5)
    if radius is not None:
        lw = radius

    # Check for invalid radius
    if not isinstance(lw, numbers.Integral) or lw < 0:
        raise ValueError('Radius must be a nonnegative integer.')

    # Generate 1D Gaussian kernel
    x = torch.arange(-lw, lw + 1, dtype=torch.float32)
    gauss_kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_kernel = gauss_kernel / gauss_kernel.sum()

    # Adjust kernel for derivative order if necessary
    if order == 1:
        gauss_kernel = -x / (sigma ** 2) * gauss_kernel
    elif order == 2:
        gauss_kernel = (x ** 2 - sigma ** 2) / (sigma ** 4) * gauss_kernel

    # Prepare for convolution
    gauss_kernel = gauss_kernel.view(1, 1, -1)

    # Move the axis to apply the filter along
    input_tensor = input_tensor.unsqueeze(0)
    if axis != -1:
        input_tensor = input_tensor.transpose(axis, -1)

    # Apply padding depending on the mode
    padding = lw
    input_tensor = F.pad(input_tensor, (padding, padding), mode=mode, value=cval)

    # Apply the convolution
    output_tensor = F.conv1d(input_tensor, gauss_kernel, padding=0)

    # If axis was moved, revert back
    if axis != -1:
        output_tensor = output_tensor.transpose(axis, -1)

    # Remove the extra dimensions added earlier
    output_tensor = output_tensor.squeeze(0)

    if output is not None:
        output.copy_(output_tensor)
        return output
    else:
        return output_tensor
