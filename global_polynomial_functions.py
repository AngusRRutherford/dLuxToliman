import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, jit, vmap
from jax.scipy.ndimage import map_coordinates
import jax.lax as lax
import optax
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
import dLux as dl
import dLux.utils as dlu
import dLuxToliman as dlT
import math
import numpy as np
import equinox as eqx
import zodiax as zdx
import optax

###############################################################################
# 1) Create normalized coordinate grid [-1, +1] & build polynomial powers
###############################################################################
def create_image_coords(npix, pixel_scale):
    """
    Returns two arrays (grid_x, grid_y) each of shape (npix, npix),
    spanning [-1,1]. (pixel_scale is ignored to keep the coordinate domain small.)
    """
    coords_1d = jnp.linspace(-1.0, 1.0, npix)
    grid_x, grid_y = jnp.meshgrid(coords_1d, coords_1d, indexing="xy")
    return grid_x, grid_y

def get_polynomial_powers(order):
    """
    Returns a list of (p, q) pairs for all terms x^p * y^q with p+q <= order.
    For order=7, that yields 36 terms.
    """
    powers = []
    for n in range(order + 1):
        for p in range(n + 1):
            q = n - p
            powers.append((p, q))
    return powers

###############################################################################
# 2) Distortion function: warp coordinates via polynomial expansions
###############################################################################
def distort_coords(params, coords, powers):
    """
    Applies a polynomial distortion to the given coordinates.
    
    params: array of shape (2*N,), where the first N elements are the x-distortion
            coefficients and the last N are the y-distortion coefficients.
    coords: tuple (x_grid, y_grid), each of shape (npix, npix) in [-1, 1].
    powers: list of (p, q) exponent pairs.
    
    Returns:
      (x_dist, y_dist) = distorted coordinates (each shape (npix, npix)).
    """
    x_grid, y_grid = coords
    n_terms = len(powers)
    px = params[:n_terms]
    py = params[n_terms:]
    
    shift_x = jnp.zeros_like(x_grid)
    shift_y = jnp.zeros_like(y_grid)
    
    for i, (p, q) in enumerate(powers):
        shift_x += px[i] * (x_grid ** p) * (y_grid ** q)
        shift_y += py[i] * (x_grid ** p) * (y_grid ** q)
    
    return x_grid + shift_x, y_grid + shift_y

###############################################################################
# 3) Sample an image at the distorted coordinates -> final "warped" image
###############################################################################
def sample_image_at_distorted_coords(image, params, coords, powers):
    """
    Given an image, applies the polynomial distortion to its coordinate grid
    and samples a warped version via map_coordinates.
    
    The input grid (in [-1, 1]) is mapped into pixel indices [0, npix-1].
    """
    x_dist, y_dist = distort_coords(params, coords, powers)
    npix = image.shape[0]
    half = (npix - 1) / 2.0
    row_indices = half * (y_dist + 1.0)
    col_indices = half * (x_dist + 1.0)
    sample_coords = jnp.stack([row_indices, col_indices], axis=0)
    return map_coordinates(image, sample_coords, order=1, mode="nearest")

###############################################################################
# 4) Build chi-squared loss function over multiple PSFs (vectorized)
# ##############################################################################
# def build_loss_fn(ideal_images, distorted_images, coords, powers, read_noise_var=0.0, epsilon=1e-14):
#     """
#     Returns a chi-squared loss function over a batch of ideal and distorted images.
    
#     The loss is computed as:
#        loss = sum((residual^2) / (predicted + read_noise_var + epsilon))
       
#     ideal_images, distorted_images: arrays of shape (B, npix, npix)
#     coords: (x_grid, y_grid) in [-1, 1]
#     powers: polynomial exponent pairs
#     read_noise_var: additional read noise variance (set to 0 if not used)
#     epsilon: a small constant to avoid division by zero.
#     """
#     def loss_fn(params):
#         predicted = vmap(lambda img: sample_image_at_distorted_coords(img, params, coords, powers))(ideal_images)
#         residuals = predicted - distorted_images
#         # Estimate variance per pixel (using model prediction as proxy for Poisson variance)
#         error = np.sqrt(distorted_images)
#         variance = predicted
#         chi2_loss = jnp.sum((residuals ** 2) / (variance + epsilon))
#         return chi2_loss
#     return loss_fn

# def build_loss_fn(ideal, data, coords, powers, read_noise=20.0, eps=1e-8):
#     def loss_fn(params):
#         pred = vmap(lambda im: sample_image_at_distorted_coords(im, params, coords, powers))(ideal)
#         resid = pred - data
#         var   = data + read_noise + eps
#         return jnp.sum((resid**2) / var)
#     return loss_fn

# def build_loss_fn(ideal_images, data_images, coords, powers,
#                   n_images: int = 1,
#                   read_noise_var: float = 20.0,
#                   epsilon: float = 1e-8):
#     """
#     Returns a chi-squared loss over a batch of ideal and observed PSFs.
#     ideal_images: (B, H, W) noise-free models
#     data_images : (B, H, W) observed averages of n_images exposures
#     variance = model/n_images + read_noise_var/n_images + epsilon
#     """
#     def loss_fn(params):
#         pred = vmap(lambda img: sample_image_at_distorted_coords(
#                      img, params, coords, powers))(ideal_images)
#         resid = pred - data_images
#         var   = pred / n_images + read_noise_var / n_images + epsilon

#         return jnp.sum((resid ** 2) / var)
#     return loss_fn

def build_loss_fn(
    ideal_images: jnp.ndarray,
    data_images: jnp.ndarray,
    coords: tuple[jnp.ndarray, jnp.ndarray],
    powers: list[tuple[int,int]],
    n_images: int = 1,
    read_noise_var: float = 0.0,
    epsilon: float = 1e-8,
    var_floor: float = 1e-6,
):
    """
    Returns a chi-squared loss function over a batch of PSFs.
    
    ideal_images:   (B, H, W) noiseless models
    data_images:    (B, H, W) observed stacks (averaged over n_images exposures)
    coords, powers: as before for the global polynomial warp
    n_images:       number of exposures averaged into data_images
    read_noise_var: σ_read² (per exposure)
    epsilon:        tiny floor to avoid div/0
    var_floor:      hard floor on variance to prevent collapse
    """
    def loss_fn(params):
        # 1) predict warped stack
        pred = vmap(lambda im: sample_image_at_distorted_coords(
            im, params, coords, powers
        ))(ideal_images)                            # → (B, H, W)
        var = pred / n_images + read_noise_var  / n_images + epsilon

        #var = pred / n_images + read_noise_var + epsilon
        var = jnp.maximum(var, var_floor)           # hard floor
        
        # 3) chi²
        resid = pred - data_images
        return jnp.sum((resid**2) / var)
    
    return loss_fn
###############################################################################
# 5) Fit function using ADAM with a warmup-cosine decay schedule and parameter averaging
###############################################################################
# def fit_polynomial_adam(ideal_images, distorted_images, coords, powers,
#                         learning_rate=1e-5, num_steps=5000, avg_window=100, true_params=None):
#     """
#     Recovers the polynomial distortion parameters by minimizing the chi-squared loss over all
#     image pairs using ADAM with a warmup-cosine decay learning rate schedule, and averages 
#     the last avg_window parameter vectors.
    
#     If true_params is provided, computes the mismatch (sum of absolute differences)
#     for each final parameter vector, then returns the mean and std of that mismatch.
    
#     Returns:
#       (params_avg, final_loss_avg, loss_history, mismatch_mean, mismatch_std, final_params_arr)
#     """
#     loss_fn = build_loss_fn(ideal_images, distorted_images, coords, powers)
#     grad_fn = jit(grad(loss_fn))

#     n_terms = len(powers)
#     param_size = 2 * n_terms
#     params = jnp.zeros(param_size)
#     #params = true_params

#     # Learning rate schedule: warmup to learning_rate over 500 steps, then cosine decay to 10% of learning_rate.
#     scheduler = optax.warmup_cosine_decay_schedule(
#         init_value=0.0,
#         peak_value=learning_rate,
#         warmup_steps=500,
#         decay_steps=num_steps - 500,
#         end_value=learning_rate * 0.1
#     )
#     optimizer = optax.adam(scheduler)
#     opt_state = optimizer.init(params)

#     loss_history = []
#     final_params_history = []
#     final_loss_history = []

#     @jit
#     def update(params, opt_state):
#         grads = grad_fn(params)
#         updates, opt_state = optimizer.update(grads, opt_state, params)
#         new_params = optax.apply_updates(params, updates)
#         return new_params, opt_state

#     params_list = []
#     for step in tqdm(range(num_steps), desc="Fitting", position=1, leave=True):
#         params, opt_state = update(params, opt_state)
#         params_list.append(params)
#         current_loss = loss_fn(params)
#         loss_history.append(current_loss)
#         if step >= num_steps - avg_window:
#             final_params_history.append(params)
#             final_loss_history.append(current_loss)
#         if step % 500 == 0:
#             print(f"Step {step}, Loss={float(current_loss):.6e}")
        


#     final_params_arr = jnp.stack(final_params_history, axis=0)
#     final_loss_arr = jnp.array(final_loss_history)
#     params_avg = jnp.mean(final_params_arr, axis=0)
#     final_loss_avg = float(jnp.mean(final_loss_arr))

#     mismatch_mean, mismatch_std = None, None
#     if true_params is not None:
#         mismatch_list = []
#         for pvec in final_params_arr:
#             diff = jnp.abs(pvec - true_params)
#             mismatch = jnp.sum(diff)
#             mismatch_list.append(mismatch)
#         mismatch_arr = jnp.array(mismatch_list)
#         mismatch_mean = float(jnp.mean(mismatch_arr))
#         mismatch_std = float(jnp.std(mismatch_arr))
    


#     return params_avg, final_loss_avg, loss_history, mismatch_mean, mismatch_std, final_params_arr, params_list

def fit_polynomial_adam(
    ideal_images: jnp.ndarray,
    data_images:  jnp.ndarray,
    coords:       tuple[jnp.ndarray, jnp.ndarray],
    powers:       list[tuple[int,int]],
    learning_rate: float     = 1e-5,
    num_steps:     int       = 5000,
    avg_window:    int       = 100,
    true_params:   jnp.ndarray|None = None,
    n_images:      int       = 1,
    read_noise_var: float    = 0.0,
    epsilon:       float     = 1e-8,
    var_floor:     float     = 1e-6,
):
    """
    Minimizes the χ² loss to recover the global‐polynomial params.
    
    Returns:
      params_avg           – average of the last avg_window iterates  
      final_loss_avg       – mean loss over the last avg_window iterates  
      loss_history         – list(length=num_steps) of losses per step  
      mismatch_mean, std   – if true_params given, mean+std of |p-p_true| sums  
      final_params_arr     – array(avg_window, 2*N_terms) of last params  
      params_list          – full list(length=num_steps) of all iterates
      mse_history          – list(length=num_steps) of MSE of params vs true_params
    """
    loss_fn = build_loss_fn(
        ideal_images, data_images,
        coords, powers,
        n_images=n_images,
        read_noise_var=read_noise_var,
        epsilon=epsilon,
        var_floor=var_floor,
    )
    grad_fn = jit(grad(loss_fn))

    # init parameters
    n_terms = len(powers)
    params  = jnp.zeros(2 * n_terms)

    #schedule + optim chain with gradient clipping
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value    = 0.0,
        peak_value    = learning_rate,
        warmup_steps  = 500,
        decay_steps   = num_steps - 500,
        end_value     = learning_rate * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(scheduler)
    )
    opt_state = optimizer.init(params)
    

    # storage
    loss_history     = []
    params_list      = []
    final_params     = []
    final_loss_vals  = []
    mse_history      = []  # NEW: track MSE each step

    @jit
    def step(p, st):
        g = grad_fn(p)
        updates, st = optimizer.update(g, st, p)
        return optax.apply_updates(p, updates), st

    # main loop
    for i in tqdm(range(num_steps), desc = "Fitting"):
        params, opt_state = step(params, opt_state)
        params_list.append(params)
        
        L = loss_fn(params)
        loss_history.append(float(L))

        # compute and store MSE if true_params provided
        if true_params is not None:
            diff = params - true_params
            mse = float(jnp.mean(diff**2))
        else:
            mse = float('nan')
        mse_history.append(mse)  # NEW

        if i >= num_steps - avg_window:
            final_params.append(params)
            final_loss_vals.append(L)

        if i % 500 == 0:
            print(f"Step {i:4d} | Loss = {float(L):.6e} | MSE = {mse:.3e}")

    # post‐process
    final_arr    = jnp.stack(final_params)
    params_avg   = jnp.mean(final_arr, axis=0)
    final_L_avg  = float(jnp.mean(jnp.array(final_loss_vals)))

    mismatch_mean = mismatch_std = None
    if true_params is not None:
        diffs = jnp.abs(final_arr - true_params)
        sums  = jnp.sum(diffs, axis=1)
        mismatch_mean = float(jnp.mean(sums))
        mismatch_std  = float(jnp.std(sums))

    return (
        params_avg,
        final_L_avg,
        loss_history,
        mismatch_mean,
        mismatch_std,
        final_arr,
        params_list,
        mse_history  # NEW: return MSE history
    )

# def fit_polynomial_adam_fast(
#     ideal_images, data_images, coords, powers,
#     key,                        # for initialisation
#     peak_lr        = 3e-4,
#     num_steps      = 600,
#     avg_window     = 200,
#     true_params    = None,
#     n_images       = 1,
#     read_noise_var = 0.,
#     epsilon        = 1e-8,
#     var_floor      = None      # ← if None we set it to read_noise_var
# ):
#     # 0) --- normalise coords to [-1,1] ---
#     x, y = coords
#     x_n  = (x - x.mean()) / jnp.abs(x).max()
#     y_n  = (y - y.mean()) / jnp.abs(y).max()
#     coords_n = (x_n, y_n)

#     # 1) loss -------------------------------------------------------------
#     if var_floor is None:
#         var_floor = read_noise_var

#     loss_fn = build_loss_fn(
#         ideal_images, data_images,
#         coords_n, powers,
#         n_images        = n_images,
#         read_noise_var  = read_noise_var,
#         epsilon         = epsilon,
#         var_floor       = var_floor,
#     )
#     grad_fn = jit(grad(loss_fn))

#     # 2) initialise -------------------------------------------------------
#     n_terms  = len(powers)
#     params   = 1e-5 * jr.normal(key, (2*n_terms,))   # ~ true scale

#     sched = optax.linear_onecycle_schedule(
#         transition_steps = num_steps,
#         peak_value       = peak_lr,
#         pct_start        = 0.15,
#         div_factor       = 20.0,
#         final_div_factor = 1e3
#     )
#     opt = optax.chain(
#         optax.clip_by_global_norm(1.0),
#         optax.adamw(sched, b1=0.9, b2=0.99, weight_decay=0.0)
#     )
#     opt_state = opt.init(params)

#     # 3) loop -------------------------------------------------------------
#     loss_hist, mse_hist = [], []
#     final_params, final_losses = [], []

#     @jit
#     def step(p, s):
#         g = grad_fn(p)
#         upd, s = opt.update(g, s, p)
#         return optax.apply_updates(p, upd), s

#     for i in range(num_steps):
#         params, opt_state = step(params, opt_state)
#         L = float(loss_fn(params))
#         loss_hist.append(L)

#         if true_params is not None:
#             mse_hist.append(float(jnp.mean((params-true_params)**2)))

#         if i >= num_steps - avg_window:
#             final_params.append(params)
#             final_losses.append(L)

#         if i % 200 == 0:
#             print(f"[{i:4d}/{num_steps}]  χ² = {L:.3e}")

#     final_arr    = jnp.stack(final_params)
#     params_avg   = jnp.mean(final_arr, 0)
#     final_L_avg  = float(jnp.mean(jnp.array(final_losses)))

#     return (params_avg, final_L_avg, loss_hist,
#             None, None, final_arr, None, mse_hist)

###############################################################################
# 6) Helper: Get dither offsets with a specified count (random, with center included)
###############################################################################
def get_dither_offsets_random(k, scale, r_factor=1.0, key=None):
    """
    Returns exactly k dither offsets (as a jnp.array of shape (k, 2)), where k is an odd integer.
    One offset is [0, 0], and the remaining (k-1) offsets are randomly placed around the origin,
    uniformly in both x and y, scaled by (scale * r_factor).
    
    Parameters:
      k (int): Total number of offsets (must be odd, e.g., 5, 7, 9, ..., 25).
      scale (float): Basic scale factor (e.g., detector pixel size in radians).
      r_factor (float): Multiplier for the spread.
      key (PRNGKey): A JAX random key. If None, defaults to jax.random.PRNGKey(0).
    
    Returns:
      jnp.array: Offsets of shape (k, 2).
    """
    if k % 2 == 0:
        raise ValueError("k must be an odd number.")
    if key is None:
        key = jax.random.PRNGKey(0)
    if k == 1:
        return scale * jnp.array([[0, 0]])
    
    # Generate k-1 random offsets uniformly within [-scale*r_factor, scale*r_factor] in both dimensions.
    random_offsets = jax.random.uniform(key, shape=(k - 1, 2), minval=-scale * r_factor, maxval=scale * r_factor)
    # Add the center offset [0, 0].
    offsets = jnp.concatenate([jnp.zeros((1, 2)), random_offsets], axis=0)
    return offsets

def get_dither_offsets_random_only_x(k, scale, r_factor=1.0, key=None):
    """
    Returns exactly k dither offsets (jnp.array of shape (k, 2)), where k is an odd integer.
    One offset is [0, 0], and the remaining (k-1) offsets are randomly placed along the x-axis,
    uniformly in the range [-scale * r_factor, scale * r_factor], with y fixed at 0.
    
    Parameters:
      k (int): Total number of offsets (must be odd, e.g., 5, 7, 9, ..., 25).
      scale (float): Basic scale factor (e.g., detector pixel size in radians).
      r_factor (float): Multiplier that sets the spread.
      key (PRNGKey): JAX random key. If None, a default key (PRNGKey(0)) is used.
    
    Returns:
      jnp.array: Offsets of shape (k, 2), with y-component always zero.
    """
    if k % 2 == 0:
        raise ValueError("k must be an odd number.")
    if key is None:
        key = jax.random.PRNGKey(0)
    if k == 1:
        return scale * jnp.array([[0, 0]])
    
    # Only dither in x-direction; y is always 0
    x_offsets = jax.random.uniform(key, shape=(k - 1, 1), minval=-scale * r_factor, maxval=scale * r_factor)
    y_offsets = jnp.zeros_like(x_offsets)
    random_offsets = jnp.concatenate([x_offsets, y_offsets], axis=1)

    # Include the central offset [0, 0]
    offsets = jnp.concatenate([jnp.zeros((1, 2)), random_offsets], axis=0)
    return offsets

def get_dither_offsets_random_only_y(k, scale, r_factor=1.0, key=None):
    """
    Returns exactly k dither offsets (jnp.array of shape (k, 2)), where k is an odd integer.
    One offset is [0, 0], and the remaining (k-1) offsets are randomly placed along the y-axis,
    uniformly in the range [-scale * r_factor, scale * r_factor], with the x-component fixed at 0.
    
    Parameters:
      k (int): Total number of offsets (must be odd, e.g., 5, 7, 9, ..., 25).
      scale (float): Basic scale factor (e.g., detector pixel size in radians).
      r_factor (float): Multiplier that sets the spread.
      key (PRNGKey): JAX random key. If None, defaults to jax.random.PRNGKey(0).
    
    Returns:
      jnp.array: Offsets of shape (k, 2), with x-component always zero.
    """
    if k % 2 == 0:
        raise ValueError("k must be an odd number.")
    if key is None:
        key = jax.random.PRNGKey(0)
    if k == 1:
        return scale * jnp.array([[0, 0]])
    
    # Only dither in the y direction: generate random y offsets and fix x to 0.
    y_offsets = jax.random.uniform(key, shape=(k - 1, 1),
                                   minval=-scale * r_factor,
                                   maxval= scale * r_factor)
    x_offsets = jnp.zeros_like(y_offsets)
    random_offsets = jnp.concatenate([x_offsets, y_offsets], axis=1)
    
    # Include the central offset [0, 0]
    offsets = jnp.concatenate([jnp.zeros((1, 2)), random_offsets], axis=0)
    return offsets



def get_dither_offsets_uniform(k, scale, r_factor=1.0):
    """
    Returns exactly k dither offsets (as a jnp.array of shape (k, 2)), where k is an odd integer.
    One offset is [0, 0], and the remaining (k-1) offsets are evenly distributed on a circle.
    
    Parameters:
      k (int): The total number of offsets (must be odd, e.g., 5, 7, 9, ..., 25).
      scale (float): The basic scale factor (e.g., detector pixel size).
      r_factor (float): A multiplier that sets the radius of the circle for the nonzero offsets.
    
    Returns:
      jnp.array: An array of shape (k, 2) containing the dither offsets.
    """
    if k % 2 == 0:
        raise ValueError("k must be an odd number.")
    # If k==1, just return the center offset.
    if k == 1:
        return scale * jnp.array([[0, 0]])
    
    m = k - 1  # Number of offsets on the circle.
    angles = jnp.linspace(0, 2 * jnp.pi, m, endpoint=False)
    # Compute the circle offsets: (cos(theta), sin(theta)) scaled appropriately.
    circle_offsets = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    circle_offsets = r_factor * scale * circle_offsets
    # Combine the center [0, 0] with the circle offsets.
    offsets = jnp.concatenate([jnp.zeros((1, 2)), circle_offsets], axis=0)
    return offsets

###############################################################################
# 7) Compute parameter loss (for analysis only)
###############################################################################
def compute_parameter_loss(true_params, predicted_params):
    """
    Computes both the sum-of-squared differences and sum-of-absolute differences
    between the true and predicted distortion parameters.
    """
    squared_loss = jnp.sum((true_params - predicted_params) ** 2)
    absolute_loss = jnp.sum(jnp.abs(true_params - predicted_params))
    return squared_loss, absolute_loss